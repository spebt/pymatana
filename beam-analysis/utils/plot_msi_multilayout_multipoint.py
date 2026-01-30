import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------
def load_base_resources(base_dir, tensor_path):
    """Loads static geometry and determines voxel count."""
    blob = torch.load(tensor_path, map_location="cpu", weights_only=False)
    det_verts = blob["layouts"]["position 000"]["detector units"]

    # Use first PPDF to get FOV dimensions
    first_ppdf = os.path.join(base_dir, "position_000_ppdfs.hdf5")
    with h5py.File(first_ppdf, 'r') as f:
        n_voxels = f['ppdfs'].shape[1]
    
    return det_verts, n_voxels

# -----------------------------------------------------------------------------
# 2. IMPROVED SWSI ANALYSIS (PPDF-WEIGHTED INTEGRATION)
# -----------------------------------------------------------------------------
def analyze_multi_layout_swsi_fixed(source_points_mm, base_dir, n_layouts, 
                                    purity_thresh=0.7, nx=280, ny=280, AND_COND=True):
    """
    Calculates SWSI by weighting EACH layout's similarity by its specific sensitivity.
    Restores the AND_COND toggle for SNMMI analysis.
    """
    source_indices = [mm_to_idx(sx, sy, nx, ny) for sx, sy in source_points_mm]
    n_voxels = nx * ny
    
    # Cumulative storage for weighted normalization
    total_weighted_sim = np.zeros(n_voxels, dtype=np.float32)
    total_sensitivity = np.zeros(n_voxels, dtype=np.float32)
    
    print(f"--- Processing {n_layouts} Layouts | AND Logic: {AND_COND} ---")
    
    for l_idx in range(n_layouts):
        # File paths per layout
        m_path = os.path.join(base_dir, f"beams_masks_configuration_{l_idx:02d}.hdf5")
        p_path = os.path.join(base_dir, f"beams_properties_configuration_{l_idx:02d}.hdf5")
        ppdf_path = os.path.join(base_dir, f"position_{l_idx:03d}_ppdfs.hdf5")
        
        if not all(os.path.exists(p) for p in [m_path, p_path, ppdf_path]):
            continue

        with h5py.File(m_path, 'r') as f_m, h5py.File(p_path, 'r') as f_p, h5py.File(ppdf_path, 'r') as f_ppdf:
            # 1. Load Layout Resources
            masks = f_m["beam_mask"][:]
            properties = f_p["beam_properties"][:]
            mpxi_lookup = {(int(r[1]), int(r[2])): int(r[10]) for r in properties}
            
            # Layout-specific sensitivity map
            layout_sensitivity = np.sum(f_ppdf['ppdfs'][:], axis=0)
            total_sensitivity += layout_sensitivity
            
            # 2. Footprint Calculation (Multiplexed Only)
            fov_footprints = []
            for v_idx in range(n_voxels):
                b_ids = masks[:, v_idx]
                act = np.where(b_ids > 0)[0]
                fp = {(int(d), int(b_ids[d])) for d in act if mpxi_lookup.get((int(d), int(b_ids[d])), 0) >= 2}
                purity = len(fp) / len(act) if len(act) > 0 else 0
                fov_footprints.append(fp if purity >= purity_thresh else None)

            source_fps = [fov_footprints[idx] for idx in source_indices]

            # 3. Geometric Ambiguity * Layout Sensitivity
            for v_idx, v_fp in enumerate(fov_footprints):
                if v_fp is None or layout_sensitivity[v_idx] == 0:
                    continue
                
                # Tight exclusion to see center hub clearly
                gx, gy = idx_to_mm(v_idx, nx, ny)
                if any(np.sqrt((gx-sx)**2 + (gy-sy)**2) < 2.5 for sx, sy in source_points_mm):
                    continue
                
                # Binary Similarity
                sim_scores = [len(s_fp.intersection(v_fp)) / len(s_fp) if s_fp else 0 for s_fp in source_fps]
                
                # Restore the toggle logic
                binary_sim = min(sim_scores) if AND_COND else max(sim_scores)
                
                # Accumulate sensitivity-weighted similarity
                total_weighted_sim[v_idx] += (binary_sim * layout_sensitivity[v_idx])

    # Final SWSI: Rotationally Integrated Sensitivity-Weighted Similarity
    swsi_map = np.divide(total_weighted_sim, total_sensitivity, 
                         out=np.zeros_like(total_weighted_sim), 
                         where=total_sensitivity != 0)
    
    return swsi_map.reshape(nx, ny)

# -----------------------------------------------------------------------------
# 3. UTILS & VISUALIZATION
# -----------------------------------------------------------------------------
def mm_to_idx(x, y, nx, ny, fov=70):
    ix, iy = int((x + fov/2) / (fov/nx)), int((y + fov/2) / (fov/ny))
    return max(0, min(nx-1, ix)) * ny + max(0, min(ny-1, iy))

def idx_to_mm(v_idx, nx, ny, fov=70):
    ix, iy = divmod(v_idx, ny)
    return (ix - nx/2) * (fov/nx), (iy - ny/2) * (fov/ny)

def visualize_swsi(swsi_2d, source_points_mm, out_dir, and_cond):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(swsi_2d.T, extent=[-35, 35, -35, 35], origin='lower', cmap='hot', interpolation='bilinear')
    plt.colorbar(im, label="SWSI Score")
    
    sx, sy = zip(*source_points_mm)
    ax.scatter(sx, sy, color='cyan', marker='*', s=250, label="Sources", edgecolors='black', zorder=10)
    
    ax.set_aspect('equal')
    ax.set_title(f"Master SWSI Map (AND={and_cond})\nIntegrated PPDF Weighting & Rotational Stability")
    plt.savefig(os.path.join(out_dir, f"master_swsi_and_{and_cond}.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "../../../data/system_layout_2mm_36pinholes_rotated/filtered_outputs/mpxi_2"
    TENSOR = "../../../data/scanner_layouts/system_layout_2mm_36pinholes_rotated.tensor"
    PHANTOM = [(5.0, 5.0), (-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0)]
    
    # Toggle this to False to see individual ghosts, True for the center Hub
    AND_CONDITION = True 
    
    geometry, n_voxels = load_base_resources(DATA_DIR, TENSOR)
    swsi_2d = analyze_multi_layout_swsi_fixed(PHANTOM, DATA_DIR, 10, AND_COND=AND_CONDITION)
    visualize_swsi(swsi_2d, PHANTOM, DATA_DIR, AND_CONDITION)