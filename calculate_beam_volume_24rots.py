import numpy as np
import h5py
import os
import scipy as sp
import skimage as ski
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
)

INPUT_DIR = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/sysmats"
OUTPUT_DIR = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/beam_volumes" # Saving final results here
BEAM_DIR = "/vscratch/grp-rutaoyao/Harsh/24_rots_again/npzs"
N_ROTATIONS = 24
FOV_X_PIXELS = 512
FOV_Y_PIXELS = 512
PIXEL_SIZE_MM = 0.25
EXPECTED_DETECTORS_PER_FILE = 864 
PROFILE_HALF_LENGTH = 15 # Half-length of line profile for FWHM in pixels (we can change this if we want)
PROFILE_INTERPOLATION_FACTOR = 4 # Interpolation factor for FWHM calculation

# --- Helper Functions ---

def get_intersections(slope, intercept):
    """
    Calculates intersection points of line x = slope*y + intercept
    with the FOV boundaries (0 to 511 for both x and y).
    Returns a list of valid [x, y] intersection points.
    """
    points = []
    # Check intersections with y=0 and y=511
    if np.isinf(slope): # Vertical line x = intercept
        if 0 <= intercept <= 511:
            points.append([intercept, 0])
            points.append([intercept, 511])
    else: # Non-vertical lines
        x_at_y0 = intercept
        x_at_y511 = slope * (FOV_Y_PIXELS - 1) + intercept
        if 0 <= x_at_y0 <= (FOV_X_PIXELS - 1):
            points.append([x_at_y0, 0])
        if 0 <= x_at_y511 <= (FOV_X_PIXELS - 1):
            points.append([x_at_y511, FOV_Y_PIXELS - 1])

        # Check intersections with x=0 and x=511 (only if slope != 0)
        if not np.isclose(slope, 0):
            y_at_x0 = -intercept / slope
            y_at_x511 = ((FOV_X_PIXELS - 1) - intercept) / slope
            if 0 <= y_at_x0 <= (FOV_Y_PIXELS - 1):
                points.append([0, y_at_x0])
            if 0 <= y_at_x511 <= (FOV_Y_PIXELS - 1):
                points.append([FOV_X_PIXELS - 1, y_at_x511])
        elif 0 <= intercept <= (FOV_X_PIXELS - 1): # Special case for horizontal line (slope=0): y = intercept_y
             pass
    if len(points) > 1:
        unique_points = np.unique(np.array(points), axis=0)
        return unique_points.tolist()
    return points

def get_beam_center(slope, intercept):
    """Calculates the geometric center of the beam segment within the FOV."""
    intersections = get_intersections(slope, intercept)
    if len(intersections) < 2:
        print(f"Warning: Beam with slope={slope}, intercept={intercept} has < 2 intersections. Using FOV center.")
        return [FOV_X_PIXELS / 2.0, FOV_Y_PIXELS / 2.0]
    center = np.mean(np.array(intersections), axis=0)
    return center.tolist() # [center_x, center_y]

def calculate_fwhm(profile):
    """Calculates FWHM from an intensity profile."""
    if profile is None or len(profile) < 2 or np.max(profile) == 0:
        return 0.0

    # Interpolate for potentially better accuracy
    profile_interpolated = sp.ndimage.zoom(profile, PROFILE_INTERPOLATION_FACTOR, order=1)

    half_max = np.max(profile_interpolated) / 2.0
    indices = np.where(profile_interpolated >= half_max)[0]

    if len(indices) < 2:
        # Handle cases where peak is too narrow or doesn't drop below half-max
        return 0.0

    fwhm_interpolated = indices[-1] - indices[0]
    fwhm_pixels = fwhm_interpolated / PROFILE_INTERPOLATION_FACTOR
    return fwhm_pixels


# --- Main Processing Function ---

def calculate_volumes_for_rotation(hdf5_fname, params_fname, pbar, task_detectors):
    """
    Loads data for one rotation, calculates tangential width and volume for each beam.
    Returns beam T_sums and beam volumes for this rotation.
    """
    fov_pixels_flat = FOV_X_PIXELS * FOV_Y_PIXELS
    beam_T_sums = None
    beam_volumes = None

    try:
        with h5py.File(hdf5_fname, "r") as f:
            ppdfs_flat = np.copy(f["ppdfs"][:])
            n_detectors, flat_size = ppdfs_flat.shape
            if flat_size != fov_pixels_flat:
                 raise ValueError(f"Flat PPDF size mismatch in {hdf5_fname}")
        params_data = np.load(params_fname)
        beam_params = params_data['beam params'] # (n_detectors, 15, 5) [s, i, r, L_pix, T_sum]
        # filtered_ppdfs_3d = params_data['filtered ppdfs'] # Not needed here

        if beam_params.shape[0] != n_detectors:
             raise ValueError(f"Detector count mismatch between {hdf5_fname} ({n_detectors}) and {params_fname} ({beam_params.shape[0]})")
        beam_T_sums = np.zeros((n_detectors, 15))
        beam_volumes = np.zeros((n_detectors, 15))
        beam_widths_mm = np.zeros((n_detectors, 15)) # Store widths temporarily for debugging
        for det_idx in range(n_detectors):
            ppdf_2d = ppdfs_flat[det_idx].reshape(FOV_Y_PIXELS, FOV_X_PIXELS)
            for beam_idx in range(15):
                params = beam_params[det_idx, beam_idx]
                slope, intercept, r, length_pixels, t_sum = params

                # Only process valid beams (where r!=0 was stored in previous step)
                # Note: We stored L and T even if r<0.75, but width/volume only make sense for actual beams.
                # Use r as the indicator of a fitted beam.
                if r == 0:
                    continue
                beam_T_sums[det_idx, beam_idx] = t_sum

                # --- Calculate Tangential Width ---
                # 1. Find beam center (cx, cy) using slope and intercept
                center_x, center_y = get_beam_center(slope, intercept)

                # 2. Determine perpendicular unit vector (puv_x, puv_y)
                if np.isinf(slope): # Vertical line (x=const)
                    puv_x, puv_y = 0, 1
                elif np.isclose(slope, 0): # Horizontal line (y=const)
                    puv_x, puv_y = 1, 0
                else:
                    norm = np.sqrt(slope**2 + (-1)**2)
                    puv_x = slope / norm
                    puv_y = -1 / norm

                # 3. Define profile line ends (row=y, col=x)
                r0 = np.clip(center_y - PROFILE_HALF_LENGTH * puv_y, 0, FOV_Y_PIXELS - 1)
                c0 = np.clip(center_x - PROFILE_HALF_LENGTH * puv_x, 0, FOV_X_PIXELS - 1)
                r1 = np.clip(center_y + PROFILE_HALF_LENGTH * puv_y, 0, FOV_Y_PIXELS - 1)
                c1 = np.clip(center_x + PROFILE_HALF_LENGTH * puv_x, 0, FOV_X_PIXELS - 1)

                # 4. Extract profile from the 2D PPDF
                try:
                    profile = ski.measure.profile_line(
                        ppdf_2d,
                        (r0, c0), # Start (row, col)
                        (r1, c1), # End (row, col)
                        linewidth=1,
                        mode="constant", 
                        cval=0,        
                        order=1
                    )
                except Exception as e:
                    print(f"\nError getting profile for det {det_idx}, beam {beam_idx} in {os.path.basename(hdf5_fname)}: {e}")
                    profile = None # Set profile to None if error occurs


                # 5. Calculate FWHM in pixels
                w_pixels = calculate_fwhm(profile)

                # --- Calculate Volume ---
                w_mm = w_pixels * PIXEL_SIZE_MM
                L_mm = length_pixels * PIXEL_SIZE_MM
                V_mm3 = w_mm * 1.0 * L_mm # FWHM_vertical = 1

                beam_volumes[det_idx, beam_idx] = V_mm3
                beam_widths_mm[det_idx, beam_idx] = w_mm # For debug/verification

            pbar.update(task_detectors, advance=1)

        print(f"Max width calculated for {os.path.basename(hdf5_fname)}: {np.max(beam_widths_mm):.2f} mm")

    except Exception as e:
        print(f"\nFATAL Error processing rotation file {os.path.basename(hdf5_fname)}: {e}")
        return None, None

    return beam_T_sums, beam_volumes


# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_T_sums = []
    all_volumes = []
    pbar = Progress(
        SpinnerColumn(), BarColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(), TextColumn("[{task.completed} of {task.total}]"),
        TimeElapsedColumn(), console=None,
    )
    task_rotations = pbar.add_task("[cyan]Processing rotations...", total=N_ROTATIONS)
    task_detectors = pbar.add_task("[green]Processing PPDFs.....", total=EXPECTED_DETECTORS_PER_FILE)

    with pbar:
        for i in range(N_ROTATIONS):
            pbar.reset(task_detectors, description="[green]Processing PPDFs.....")
            hdf5_fname = os.path.join(INPUT_DIR, f"scanner_ppdfs_{i:02d}.hdf5")
            params_fname = os.path.join(BEAM_DIR, f"scanner_ppdfs_{i:02d}_beam_params.npz")

            if not os.path.exists(hdf5_fname) or not os.path.exists(params_fname):
                print(f"\nInput files missing for rotation {i}, skipping.")
                pbar.update(task_detectors, advance=EXPECTED_DETECTORS_PER_FILE, description="[red]Input Missing! Skip")
                pbar.update(task_rotations, advance=1)
                all_T_sums.append(None)
                all_volumes.append(None)
                continue

            print(f"\nProcessing rotation {i}...")
            T_sums_rot, volumes_rot = calculate_volumes_for_rotation(
                hdf5_fname, params_fname, pbar, task_detectors
            )

            all_T_sums.append(T_sums_rot)
            all_volumes.append(volumes_rot)

            pbar.update(task_rotations, advance=1)

    print("\nCombining results...")
    valid_T_sums = [t for t in all_T_sums if t is not None]
    valid_volumes = [v for v in all_volumes if v is not None]

    if not valid_T_sums or not valid_volumes:
        print("Error: No valid rotation data processed. Cannot save final results.")
    else:
        final_T_sums = np.stack(valid_T_sums, axis=0)
        final_volumes = np.stack(valid_volumes, axis=0)

        print(f"Final T_sums shape: {final_T_sums.shape}")
        print(f"Final Volumes shape: {final_volumes.shape}")

        t_sum_outfile = os.path.join(OUTPUT_DIR, "ppds_T_sums_allrots.npz")
        volume_outfile = os.path.join(OUTPUT_DIR, "ppds_volumes_allrots.npz")

        np.savez_compressed(t_sum_outfile, beam_T_sums=final_T_sums)
        np.savez_compressed(volume_outfile, beam_volumes=final_volumes)

        print(f"Saved T_sums to: {t_sum_outfile}")
        print(f"Saved Volumes to: {volume_outfile}")

    print("\nBeam volume calculation complete.")