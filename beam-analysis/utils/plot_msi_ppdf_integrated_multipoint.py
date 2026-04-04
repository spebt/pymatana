import os
import h5py
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------
FOV_MM = 10.0          # Field of View in mm
NX, NY = 200, 200      # Voxel grid dimensions
PURITY_THRESH = 0.7    # Minimum multiplexing fraction to consider a footprint
AND_CONDITION = True   # Require intersection from ALL sources
NORMALIZE = False      # For the SWSI output normalization
MIN_MPXI = 1           # THE FIX: Set to 1 for ALL beams, 2 for multiplexed-only

# -----------------------------------------------------------------------------
# 1. DATA LOADING & UTILS
# -----------------------------------------------------------------------------
def load_system_matrix(h5_path: str):
    with h5py.File(h5_path, 'r') as h5f:
        if "data" in h5f:
            data = h5f["data"][:]
            indices = h5f["indices"][:]
            indptr = h5f["indptr"][:]
            shape = tuple(h5f.attrs["shape"])
            return sp.csr_matrix((data, indices, indptr), shape=shape)
        elif "ppdfs" in h5f:
            return h5f["ppdfs"][:]
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")

def load_mask_matrix(h5_path: str):
    with h5py.File(h5_path, 'r') as h5f:
        if "data" in h5f:
            data = h5f["data"][:]
            indices = h5f["indices"][:]
            indptr = h5f["indptr"][:]
            shape = tuple(h5f.attrs["shape"])
            return sp.csr_matrix((data, indices, indptr), shape=shape).toarray()
        elif "beam_mask" in h5f:
            return h5f["beam_mask"][:]
        else:
            raise ValueError(f"Unknown mask format in {h5_path}")

def load_base_resources(base_dir, tensor_path):
    blob = torch.load(tensor_path, map_location="cpu", weights_only=False)
    det_verts = blob["layouts"]["position 000"]["detector units"]
    return det_verts

def mm_to_idx(x, y, nx, ny, fov):
    ix, iy = int((x + fov/2) / (fov/nx)), int((y + fov/2) / (fov/ny))
    ix = max(0, min(nx-1, ix))
    iy = max(0, min(ny-1, iy))
    return ix * ny + iy 

# -----------------------------------------------------------------------------
# 2. PHYSICS-AWARE ANALYSIS ENGINE
# -----------------------------------------------------------------------------
def analyze_multiplexing_ambiguity(source_points_mm, base_dir, n_layouts, 
                                   purity_thresh, nx, ny, fov, 
                                   AND_COND, NORMALIZE_OUTPUT, min_mpxi):
    n_voxels = nx * ny
    source_indices = [mm_to_idx(sx, sy, nx, ny, fov) for sx, sy in source_points_mm]
    
    global_source_fps = [set() for _ in source_indices]
    global_voxel_fps = [set() for _ in range(n_voxels)]
    
    global_source_act = [0] * len(source_indices) 
    global_voxel_act = [0] * n_voxels             
    sensitivity_accumulator = np.zeros(n_voxels, dtype=np.float32)
    
    print(f"--- Processing {n_layouts} Layouts | GLOBAL AND Logic: {AND_COND} | MPXI >= {min_mpxi} ---")
    
    for l_idx in range(n_layouts):
        m_path = os.path.join(base_dir, f"beams_masks_configuration_{l_idx:02d}.hdf5")
        p_path = os.path.join(base_dir, f"beams_properties_configuration_{l_idx:02d}.hdf5")
        ppdf_path = os.path.join(base_dir, f"position_{l_idx:03d}_ppdfs.hdf5")
        
        if not all(os.path.exists(p) for p in [m_path, p_path, ppdf_path]):
            continue

        masks = load_mask_matrix(m_path)
        ppdf_mat = load_system_matrix(ppdf_path)
        
        with h5py.File(p_path, 'r') as f_p:
            properties = f_p["beam_properties"][:]
            
        # Dynamically obeys MIN_MPXI
        mpxi_lookup = {
            (int(r[1]), int(r[2])): int(r[10]) 
            for r in properties if int(r[10]) >= min_mpxi
        }
        
        layout_sensitivity = np.asarray(ppdf_mat.sum(axis=0)).flatten()
        sensitivity_accumulator += layout_sensitivity
        
        active_voxel_indices = np.where(layout_sensitivity > 1e-6)[0]

        for i, s_idx in enumerate(source_indices):
            s_b_ids = masks[:, s_idx]
            s_act = np.where(s_b_ids > 0)[0]
            global_source_act[i] += len(s_act)
            
            for d in s_act:
                bid = int(s_b_ids[d])
                if (int(d), bid) in mpxi_lookup:
                    global_source_fps[i].add((l_idx, int(d), bid))

        for v_idx in active_voxel_indices:
            if v_idx in source_indices:
                continue
                
            v_b_ids = masks[:, v_idx]
            v_act = np.where(v_b_ids > 0)[0]
            global_voxel_act[v_idx] += len(v_act)
            
            for d in v_act:
                bid = int(v_b_ids[d])
                if (int(d), bid) in mpxi_lookup:
                    global_voxel_fps[v_idx].add((l_idx, int(d), bid))

    msi_accumulator = np.zeros(n_voxels, dtype=np.float32)
    swsi_accumulator = np.zeros(n_voxels, dtype=np.float32)
    
    valid_source_fps = []
    for fp, act in zip(global_source_fps, global_source_act):
        purity = len(fp) / act if act > 0 else 0
        if purity >= purity_thresh and len(fp) > 0:
            valid_source_fps.append(fp)
            
    if not valid_source_fps:
        print("WARNING: No sources survived the purity threshold.")
        max_sens = np.max(sensitivity_accumulator) if np.any(sensitivity_accumulator) else 1.0
        return msi_accumulator.reshape(nx, ny), swsi_accumulator.reshape(nx, ny), max_sens

    for v_idx in range(n_voxels):
        if v_idx in source_indices or sensitivity_accumulator[v_idx] < 1e-6:
            continue
            
        v_fp = global_voxel_fps[v_idx]
        v_act = global_voxel_act[v_idx]
        v_purity = len(v_fp) / v_act if v_act > 0 else 0
        
        if v_purity < purity_thresh or not v_fp:
            continue

        sim_scores = []
        ghost_support = len(v_fp) 
        
        for s_fp in valid_source_fps:
            intersection = len(s_fp.intersection(v_fp))
            sim_scores.append(intersection / ghost_support if ghost_support > 0 else 0.0)

        if not sim_scores:
            continue
            
        final_sim = min(sim_scores) if AND_COND else max(sim_scores)
        
        msi_accumulator[v_idx] = final_sim
        swsi_accumulator[v_idx] = final_sim * sensitivity_accumulator[v_idx]

    if NORMALIZE_OUTPUT:
        swsi_final = np.divide(swsi_accumulator, sensitivity_accumulator, 
                               out=np.zeros_like(swsi_accumulator), 
                               where=sensitivity_accumulator > 1e-9)
    else:
        swsi_final = swsi_accumulator

    max_sens = np.max(sensitivity_accumulator)
    return msi_accumulator.reshape(nx, ny), swsi_final.reshape(nx, ny), max_sens

# -----------------------------------------------------------------------------
# 3. VISUALIZATION
# -----------------------------------------------------------------------------
def visualize_map(data_2d, source_points_mm, out_dir, title_prefix, 
                  and_cond, fov, cbar_label, vmin=None, vmax=None):
    
    max_val = np.max(data_2d)
    min_val = np.min(data_2d)
    
    # Finding the maximum NON-ZERO value
    non_zero_elements = data_2d[data_2d > 0]
    max_non_zero = np.max(non_zero_elements) if non_zero_elements.size > 0 else 0.0
    
    if "MSI" in title_prefix:
        # 6 Decimal Places for precision hunting
        stats_str = f"Max: {max_val * 100:.6f}% | Min: {min_val * 100:.6f}%"
        print(f"\n[{title_prefix}] Absolute Max: {max_val * 100:.6f}%")
        print(f"[{title_prefix}] Max NON-ZERO: {max_non_zero * 100:.6f}%")
    else:
        stats_str = f"Max: {max_val:.6e} | Min: {min_val:.6e}"
        print(f"\n[{title_prefix}] Absolute Max: {max_val:.6e}")
        print(f"[{title_prefix}] Max NON-ZERO: {max_non_zero:.6e}")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    half_fov = fov / 2.0
    im = ax.imshow(data_2d.T, extent=[-half_fov, half_fov, -half_fov, half_fov], 
                   origin='lower', cmap='inferno', interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    
    plt.colorbar(im, label=cbar_label)
    
    sx, sy = zip(*source_points_mm)
    ax.scatter(sx, sy, color='cyan', marker='*', s=150, label="Sources", edgecolors='black')
    
    ax.set_title(f"{title_prefix}\n{stats_str}\nAND_Logic: {and_cond} | FOV: {fov}mm")
    
    clean_title = title_prefix.replace(" ", "_").replace("(", "").replace(")", "").lower()
    fname = f"{clean_title}_and_{and_cond}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    print(f"Saved: {fname}")
    plt.show()

def visualize_histogram(data_2d, out_dir, title_prefix, x_label):
    """
    Plots a histogram of all strictly non-zero values to automatically adapt
    to the active data distribution without being crushed by the zero-background.
    """
    # Isolate non-zero pixels
    non_zero_data = data_2d[data_2d > 0]
    
    if non_zero_data.size == 0:
        print(f"[{title_prefix}] Histogram skipped: No non-zero data found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # bins='auto' dynamically finds the best bin size based on the data spread
    counts, bins, patches = ax.hist(
        non_zero_data, 
        bins='auto', 
        color='royalblue', 
        edgecolor='black', 
        alpha=0.75
    )
    
    # Calculate some quick stats for the title
    mean_val = np.mean(non_zero_data)
    median_val = np.median(non_zero_data)
    
    ax.set_title(f"{title_prefix} Distribution (Non-Zero Only)\nMean: {mean_val:.4f} | Median: {median_val:.4f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency (Voxel Count)")
    ax.grid(axis='y', alpha=0.3)
    
    clean_title = title_prefix.replace(" ", "_").replace("(", "").replace(")", "").lower()
    fname = f"{clean_title}_histogram.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    print(f"Saved Histogram: {fname}")
    plt.show()

# -----------------------------------------------------------------------------
# 4. EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # FILE PATHS
    DATA_DIR = "../../../data/hybrid_sc_spect_mph_biconical_base_72_2.5_0.4/filtered_outputs/mpxi_2" 
    TENSOR = "../../../data/scanner_layouts/hybrid_sc_spect_mph_biconical_base_72_2.5_0.4.tensor"
    
    PHANTOM = [(1.75, 1.75), (1.75, -1.75), (-1.75, -1.75), (-1.75, 1.75)]
    
    det_verts = load_base_resources(DATA_DIR, TENSOR)
    
    pure_msi_map, swsi_map, max_sens = analyze_multiplexing_ambiguity(
        source_points_mm=PHANTOM, 
        base_dir=DATA_DIR, 
        n_layouts=2, 
        purity_thresh=PURITY_THRESH,
        nx=NX, 
        ny=NY, 
        fov=FOV_MM,
        AND_COND=AND_CONDITION, 
        NORMALIZE_OUTPUT=NORMALIZE,
        min_mpxi=MIN_MPXI
    )
    
    # --- 2D MAP VISUALIZATIONS ---
    visualize_map(
        data_2d=pure_msi_map, source_points_mm=PHANTOM, out_dir=DATA_DIR, 
        title_prefix="Pure MSI Map", and_cond=AND_CONDITION, 
        fov=FOV_MM, cbar_label="Similarity Ratio (0 to 1)",
        vmin=0.0, vmax=1.0 
    )
    
    visualize_map(
        data_2d=swsi_map, source_points_mm=PHANTOM, out_dir=DATA_DIR, 
        title_prefix="SWSI Map", and_cond=AND_CONDITION, 
        fov=FOV_MM, cbar_label="Sensitivity Weighted Intensity (a.u.)",
        vmin=0.0, vmax=max_sens 
    )

    # --- HISTOGRAM VISUALIZATIONS ---
    visualize_histogram(
        data_2d=pure_msi_map, 
        out_dir=DATA_DIR, 
        title_prefix="Pure MSI Map",
        x_label="Similarity Ratio"
    )

    visualize_histogram(
        data_2d=swsi_map, 
        out_dir=DATA_DIR, 
        title_prefix="SWSI Map",
        x_label="Sensitivity Weighted Intensity"
    )