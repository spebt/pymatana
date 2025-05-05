import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from rich.progress import track # Use track for simple loops

INPUT_DIR = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/beam_volumes"
BEAM_DIR = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/npzs"
OUTPUT_DIR = "results/ppds_maps" # Save maps in a dedicated subdirectory
N_ROTATIONS = 24
FOV_X_PIXELS = 512
FOV_Y_PIXELS = 512
EXPECTED_DETECTORS_PER_FILE = 864 # Should match previous steps
EPSILON = 1e-12

t_sum_file = os.path.join(INPUT_DIR, "ppds_T_sums_allrots.npz")
volume_file = os.path.join(INPUT_DIR, "ppds_volumes_allrots.npz")
mask_map_base_path = os.path.join(BEAM_DIR, "scanner_ppdfs_{:02d}_beam_params.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Consolidated Data ---
print("Loading consolidated T_sums and Volumes...")
try:
    t_sum_data = np.load(t_sum_file)
    beam_T_sums_all = t_sum_data['beam_T_sums'] # Shape (N_ROTATIONS, n_detectors, 15)

    volume_data = np.load(volume_file)
    beam_volumes_all = volume_data['beam_volumes'] # Shape (N_ROTATIONS, n_detectors, 15)

    # Validate shapes
    if beam_T_sums_all.shape[0] != N_ROTATIONS or beam_volumes_all.shape[0] != N_ROTATIONS:
         print(f"Warning: Loaded data has {beam_T_sums_all.shape[0]} rotations, expected {N_ROTATIONS}.")
    if beam_T_sums_all.shape[1] != EXPECTED_DETECTORS_PER_FILE or beam_volumes_all.shape[1] != EXPECTED_DETECTORS_PER_FILE:
         print(f"Warning: Loaded data has {beam_T_sums_all.shape[1]} detectors, expected {EXPECTED_DETECTORS_PER_FILE}.")
    n_loaded_rotations = beam_T_sums_all.shape[0] # Use actual loaded count
    n_detectors = beam_T_sums_all.shape[1]

    print(f"Loaded T_sums shape: {beam_T_sums_all.shape}")
    print(f"Loaded Volumes shape: {beam_volumes_all.shape}")

except FileNotFoundError:
    print(f"Error: Could not find input files {t_sum_file} or {volume_file}. Exiting.")
    exit()
except Exception as e:
    print(f"Error loading consolidated data: {e}. Exiting.")
    exit()


# --- Initialize PPDS Map ---
cumulative_ppds_map = np.zeros((FOV_Y_PIXELS, FOV_X_PIXELS), dtype=np.float64)

# --- Loop Through Rotations and Calculate Cumulative PPDS ---
print(f"Calculating cumulative PPDS map across {n_loaded_rotations} rotations...")
for aid in track(range(n_loaded_rotations), description="Processing Rotations"):
    try:
        # --- Load Rotation-Specific Mask Map ---
        mask_map_file = mask_map_base_path.format(aid)
        mask_data = np.load(mask_map_file)
        mask_map_rot = mask_data['filtered ppdfs']
        if mask_map_rot.shape[0] != n_detectors or \
           mask_map_rot.shape[1] != FOV_Y_PIXELS or \
           mask_map_rot.shape[2] != FOV_X_PIXELS:
            print(f"Warning: Mask map shape mismatch for rotation {aid}. Skipping.")
            continue

        # --- Get T_sums and Volumes for this Rotation ---
        T_sums_rot = beam_T_sums_all[aid]   # Shape (n_detectors, 15)
        volumes_rot = beam_volumes_all[aid] # Shape (n_detectors, 15)

        # --- Calculate Sum_V_i per detector ---
        Sum_V_i = np.sum(volumes_rot, axis=1) # Shape (n_detectors,)
        valid_detector_mask_1d = Sum_V_i > EPSILON
        denominator_V_i = np.where(valid_detector_mask_1d, Sum_V_i, EPSILON)

        # --- Calculate Numerator T_i,b_j for each voxel ---
        beam_indices_0based = mask_map_rot.astype(int) - 1 # Shape (n_detectors, H, W)
        T_at_voxel = np.zeros_like(mask_map_rot, dtype=np.float64) # Shape (n_detectors, H, W)
        valid_beam_mask = beam_indices_0based >= 0
        det_indices, y_indices, x_indices = np.where(valid_beam_mask)
        beam_idxs_for_valid_voxels = beam_indices_0based[valid_beam_mask] # 1D array

        selected_T_sums = T_sums_rot[det_indices, beam_idxs_for_valid_voxels] # 1D array
        T_at_voxel[valid_beam_mask] = selected_T_sums

        # --- Calculate PPDS Contribution for this Rotation ---
        # Contribution = T_at_voxel / denominator_V_i (broadcasted)
        # Only calculate contribution for detectors with valid total volume
        contribution_term = np.zeros_like(T_at_voxel) # Initialize contribution array
        valid_detector_mask_3d = valid_detector_mask_1d[:, None, None] # Shape (n_detectors, 1, 1)

        # Calculate contribution only where detector volume is valid
        contribution_term = np.where(
            valid_detector_mask_3d, # Condition: Is detector volume > EPSILON?
            T_at_voxel / denominator_V_i[:, None, None], # True: Calculate T/V
            0.0 # False: Contribution is 0
        )

        current_rotation_ppds = np.sum(contribution_term, axis=0) # Shape (H, W)
        cumulative_ppds_map += current_rotation_ppds

    except FileNotFoundError:
        print(f"Error: Mask map file not found for rotation {aid}. Skipping rotation.")
        continue
    except Exception as e:
        print(f"Error processing rotation {aid}: {e}. Skipping rotation.")
        continue

print("\nSaving final cumulative PPDS map...")
final_map_npy_file = os.path.join(OUTPUT_DIR, 'ppds_map_final.npy')
np.save(final_map_npy_file, cumulative_ppds_map)
print(f"Final PPDS map saved to: {final_map_npy_file}")
try:
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cumulative_ppds_map, cmap='viridis', origin='lower',
                    interpolation='nearest')
    plt.title(f'Final PPDS Map ({n_loaded_rotations} Rotations Combined)')
    plt.colorbar(im, label='PPDS (arb. units)')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.tight_layout()
    final_map_png_file = os.path.join(OUTPUT_DIR, 'ppds_map_final.png')
    plt.savefig(final_map_png_file, dpi=200)
    plt.close()
    print(f"Final PPDS map image saved to: {final_map_png_file}")
except Exception as e:
    print(f"Error saving final map image: {e}")


print("\nPPDS map calculation complete.")