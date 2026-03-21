import os
import h5py
import torch
import numpy as np
import scipy.sparse as sp  # <-- NEW: Required for sparse matrix support
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0. AUTO-DETECTING LOADERS (NEW)
# -----------------------------------------------------------------------------
def load_system_matrix(h5_path: str):
    """Auto-detects sparse/dense HDF5 and returns a SciPy CSR matrix or NumPy array."""
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
    """
    Auto-detects sparse/dense mask formats. 
    Always returns a dense NumPy array because the downstream loop 
    performs heavy column-slicing, which is extremely slow on CSR matrices.
    """
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


# -----------------------------------------------------------------------------
# 1. DATA LOADING & UTILS
# -----------------------------------------------------------------------------
def load_base_resources(base_dir, tensor_path):
    """Loads static geometry and determines voxel count, adaptable to sparse/dense."""
    blob = torch.load(tensor_path, map_location="cpu", weights_only=False)
    # Adjust key if your tensor structure differs
    det_verts = blob["layouts"]["position 000"]["detector units"]
    
    # Peek at first PPDF to get dims securely
    first_ppdf = os.path.join(base_dir, "position_000_ppdfs.hdf5")
    with h5py.File(first_ppdf, 'r') as f:
        if "data" in f:
            n_voxels = tuple(f.attrs["shape"])[1]
        elif "ppdfs" in f:
            n_voxels = f['ppdfs'].shape[1]
        else:
            raise ValueError("Unknown format in base resource PPDF.")
        
    return det_verts, n_voxels

def mm_to_idx(x, y, nx, ny, fov=70):
    ix, iy = int((x + fov/2) / (fov/nx)), int((y + fov/2) / (fov/ny))
    ix = max(0, min(nx-1, ix))
    iy = max(0, min(ny-1, iy))
    return ix * ny + iy # Standard row-major unfolding

def idx_to_mm(v_idx, nx, ny, fov=70):
    ix, iy = divmod(v_idx, ny)
    px_size = fov / nx # Assuming square pixels
    x = (ix - nx/2) * px_size + (px_size/2) # Center of pixel
    y = (iy - ny/2) * px_size + (px_size/2)
    return x, y

# -----------------------------------------------------------------------------
# 2. PHYSICS-AWARE SWSI ANALYSIS
# -----------------------------------------------------------------------------
def analyze_swsi_dot_product(source_points_mm, base_dir, n_layouts, 
                             purity_thresh=0.7, nx=512, ny=512, 
                             AND_COND=True, NORMALIZE_OUTPUT=False):
    """
    Calculates SWSI (Sensitivity Weighted Similarity Index).
    
    Args:
        NORMALIZE_OUTPUT (bool): 
            If False: Returns Sum(Sim * PPDF). Result is 'Ghost Intensity'. 
            If True:  Returns Sum(Sim * PPDF) / Sum(PPDF). Result is 'Ghost Probability'.
    """
    n_voxels = nx * ny
    source_indices = [mm_to_idx(sx, sy, nx, ny) for sx, sy in source_points_mm]
    
    # Master accumulation arrays
    swsi_accumulator = np.zeros(n_voxels, dtype=np.float32)
    sensitivity_accumulator = np.zeros(n_voxels, dtype=np.float32)
    
    print(f"--- Processing {n_layouts} Layouts | AND Logic: {AND_COND} ---")
    
    for l_idx in range(n_layouts):
        # 1. File Paths
        m_path = os.path.join(base_dir, f"beams_masks_configuration_{l_idx:02d}.hdf5")
        p_path = os.path.join(base_dir, f"beams_properties_configuration_{l_idx:02d}.hdf5")
        ppdf_path = os.path.join(base_dir, f"position_{l_idx:03d}_ppdfs.hdf5")
        
        if not all(os.path.exists(p) for p in [m_path, p_path, ppdf_path]):
            print(f"Skipping layout {l_idx} (files missing)")
            continue

        # 2. Load Data using auto-detecting loaders
        masks = load_mask_matrix(m_path) # Shape: (n_detectors, n_voxels)
        ppdf_mat = load_system_matrix(ppdf_path)
        
        with h5py.File(p_path, 'r') as f_p:
            properties = f_p["beam_properties"][:]
            
        # Create MPXI Lookup: (det_id, beam_id) -> mpxi
        # Only keep multiplexed beams (mpxi >= 2) to save lookups
        mpxi_lookup = {
            (int(r[1]), int(r[2])): int(r[10]) 
            for r in properties if int(r[10]) >= 2
        }
        
        # Load Sensitivity (Sum over all crystals for this layout)
        # np.asarray().flatten() ensures compatibility with SciPy sparse matrices
        layout_sensitivity = np.asarray(ppdf_mat.sum(axis=0)).flatten()
        
        # Accumulate total sensitivity for normalization later (if requested)
        if NORMALIZE_OUTPUT:
            sensitivity_accumulator += layout_sensitivity

        # 3. Analyze Footprints (Optimized Loop)
        # We only iterate over voxels that actually have sensitivity > 0
        active_voxel_indices = np.where(layout_sensitivity > 1e-6)[0]
        
        # Pre-calculate Source Footprints for this layout
        source_fps = []
        for s_idx in source_indices:
            s_b_ids = masks[:, s_idx]
            s_act = np.where(s_b_ids > 0)[0]
            fp = set()
            for d in s_act:
                bid = int(s_b_ids[d])
                # Check MPXI directly
                if (int(d), bid) in mpxi_lookup:
                    fp.add((int(d), bid))
            
            # Check purity
            purity = len(fp) / len(s_act) if len(s_act) > 0 else 0
            source_fps.append(fp if purity >= purity_thresh else set())

        # Loop only over active voxels for speed
        for v_idx in active_voxel_indices:
            # Exclusion Zone (Don't analyze the source itself)
            if v_idx in source_indices:
                continue
            
            # Voxel Footprint
            v_b_ids = masks[:, v_idx]
            v_act = np.where(v_b_ids > 0)[0]
            
            v_fp = set()
            for d in v_act:
                bid = int(v_b_ids[d])
                if (int(d), bid) in mpxi_lookup:
                    v_fp.add((int(d), bid))
            
            if not v_fp: continue
            
            # Purity check for target voxel
            v_purity = len(v_fp) / len(v_act)
            if v_purity < purity_thresh: continue

            # Calculate Jaccard Similarity
            sim_scores = []
            for s_fp in source_fps:
                if not s_fp:
                    sim_scores.append(0.0)
                    continue
                intersection = len(s_fp.intersection(v_fp))
                union = len(s_fp) # Jaccard denominator often usually Union, but user used len(source) in previous script (Containment Index). 
                # If you want Jaccard: union = len(s_fp.union(v_fp))
                # Using Containment (previous script logic):
                sim_scores.append(intersection / union)

            # Aggregate Logic
            if AND_COND:
                final_sim = min(sim_scores)
            else:
                final_sim = max(sim_scores)
            
            # --- THE DOT PRODUCT ---
            # Weight the similarity by the voxel's sensitivity in THIS layout
            swsi_accumulator[v_idx] += (final_sim * layout_sensitivity[v_idx])

        if (l_idx + 1) % 5 == 0:
            print(f"  Processed {l_idx + 1}/{n_layouts} layouts...")

    # 4. Final Calculation
    if NORMALIZE_OUTPUT:
        # Relative Probability (0.0 to 1.0)
        # Avoid divide by zero
        swsi_map = np.divide(swsi_accumulator, sensitivity_accumulator, 
                             out=np.zeros_like(swsi_accumulator), 
                             where=sensitivity_accumulator > 1e-9)
    else:
        # Absolute Intensity (Dot Product)
        swsi_map = swsi_accumulator
        # Optional: Normalize to 0-100 range based on max value for visualization
        if np.max(swsi_map) > 0:
             swsi_map = (swsi_map / np.max(swsi_map)) * 100

    return swsi_map.reshape(nx, ny)

# -----------------------------------------------------------------------------
# 3. VISUALIZATION
# -----------------------------------------------------------------------------
def visualize_swsi(swsi_2d, source_points_mm, out_dir, and_cond, normalized):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use 'inferno' or 'magma' for intensity maps (black is zero)
    im = ax.imshow(swsi_2d.T, extent=[-35, 35, -35, 35], origin='lower', 
                   cmap='inferno', interpolation='nearest')
    
    cbar_label = "Probability (Normalized)" if normalized else "Ghost Intensity (a.u.)"
    plt.colorbar(im, label=cbar_label)
    
    sx, sy = zip(*source_points_mm)
    ax.scatter(sx, sy, color='cyan', marker='*', s=150, label="Sources", edgecolors='black')
    
    mode_str = "Normalized" if normalized else "Absolute Intensity"
    ax.set_title(f"SWSI Analysis ({mode_str})\nAND_Logic: {and_cond}")
    
    fname = f"swsi_{'norm' if normalized else 'abs'}_and_{and_cond}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    print(f"Saved: {fname}")
    plt.show()

if __name__ == "__main__":
    # CONFIG
    DATA_DIR = "../../../data/mph_hourglass_single_position_base_2mm_18pinholes_rotated_elliptical_comp2/outputs"
    TENSOR = "../../../data/scanner_layouts/mph_hourglass_single_position_base_2mm_18pinholes_rotated_elliptical.tensor"
    
    # Square configuration
    PHANTOM = [(5.0, 5.0), (-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0)]
    
    AND_CONDITION = True
    
    # IMPORTANT: Set this to False to get the "Dot Product Intensity"
    NORMALIZE = False 
    
    det_verts, n_voxels = load_base_resources(DATA_DIR, TENSOR)
    
    swsi_map = analyze_swsi_dot_product(
        PHANTOM, DATA_DIR, n_layouts=10, # Ensure this matches your data
        AND_COND=AND_CONDITION, 
        NORMALIZE_OUTPUT=NORMALIZE
    )
    
    visualize_swsi(swsi_map, PHANTOM, DATA_DIR, AND_CONDITION, NORMALIZE)