#!/usr/bin/env python3
"""
Generate ASCI angular sensitivity histogram for one layout.

Sticks to the ORIGINAL script structure/logic as much as possible:
- Uses Rich progress (optional, but kept minimal).
- Loads beam_properties + beam_masks HDF5 for ONE layout.
- Digitizes angles into EXACTLY 360 bins (0 to 2π).
- Filters beams by:
  - NaN angles removed
  - FWHM range [min, max] from config
  - Sensitivity > 1% of max sensitivity (after FWHM filtering)
- Populates asci_histogram[pixel, angle_bin] by counting beams covering pixel.

Config-driven paths/thresholds come from YAML provided by user.
"""

import os
import sys
import argparse
import yaml
import torch
import h5py
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

# --- FIX: PyTorch Beta-Bypass Loader for Masks ---
def load_mask_matrix_dict(h5_path: str) -> dict:
    """
    Returns a dictionary of 1D tensors to completely avoid instantiating 
    the leaky PyTorch SparseCsrTensor beta object.
    """
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            return {
                "is_sparse": True,
                # Force int64 for PyTorch indexing compatibility
                "indptr": torch.tensor(h5f["indptr"][:], dtype=torch.int64),
                "indices": torch.tensor(h5f["indices"][:], dtype=torch.int64),
                # Masks use integer IDs, so data is int32
                "data": torch.tensor(h5f["data"][:], dtype=torch.int32),
                "shape": tuple(h5f.attrs["shape"])
            }
        elif "beam_mask" in h5f:
            return {
                "is_sparse": False,
                "dense_tensor": torch.tensor(h5f["beam_mask"][:], dtype=torch.int32)
            }
        else:
            raise ValueError(f"Unknown mask format in {h5_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate ASCI histogram for a given layout.")
    parser.add_argument("layout_idx", type=int, help="Layout index to process (e.g., 0..23).")
    parser.add_argument("--config", default="configs/base_config.yml", help="Path to YAML config.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Config ---
    input_dir = cfg["paths"]["data_output_dir"]

    n_bins = int(cfg["analysis"]["n_bins"])
    if n_bins != 360:
        print(f"Error: This script is intended to run with n_bins=360. Got: {n_bins}")
        sys.exit(1)

    fwhm_min = float(cfg["analysis"]["fwhm_min_mm"])
    fwhm_max = float(cfg["analysis"]["fwhm_max_mm"])
    fwhm_col = int(cfg["analysis"]["fwhm_column_index"])

    # Histogram size derived from FOV
    n_pixels = tuple(cfg["fov"]["n_pixels"])
    if len(n_pixels) != 2:
        print(f"Error: fov.n_pixels must be 2D like [512,512]. Got: {n_pixels}")
        sys.exit(1)
    n_vox = int(n_pixels[0] * n_pixels[1])

    # --- Files ---
    prop_file = f"beams_properties_configuration_{args.layout_idx:02d}.hdf5"
    mask_file = f"beams_masks_configuration_{args.layout_idx:02d}.hdf5"

    prop_path = os.path.join(input_dir, prop_file)
    mask_path = os.path.join(input_dir, mask_file)

    if not os.path.exists(prop_path):
        print(f"Error: missing properties file: {prop_path}")
        sys.exit(1)
    if not os.path.exists(mask_path):
        print(f"Error: missing masks file: {mask_path}")
        sys.exit(1)

    # --- Bin boundaries: EXACTLY 360 bins across [0, 2π) ---
    angular_bin_boundaries = torch.linspace(0.0, 2.0 * torch.pi, steps=n_bins + 1)

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        refresh_per_second=10,
    ) as progress:
        task = progress.add_task(f"ASCI layout {args.layout_idx:02d}", total=4)

        # 1) Load data
        with h5py.File(prop_path, "r") as f:
            layout_beams_properties = torch.from_numpy(f["beam_properties"][:])  # shape: (N_beams, K)
            _header = f["beam_properties"].attrs.get("Header", None)

        # --- UPDATED: Load mask raw components to avoid PyTorch leaks ---
        beams_masks_dict = load_mask_matrix_dict(mask_path)

        progress.update(task, advance=1)

        # 2) Digitize angles
        angles = layout_beams_properties[:, 3]
        digitized_angles = torch.bucketize(angles, angular_bin_boundaries, right=False) - 1
        digitized_angles = digitized_angles.clamp(0, n_bins - 1)

        layout_beams_properties = torch.cat(
            (layout_beams_properties, digitized_angles.unsqueeze(1).float()), dim=1
        )

        progress.update(task, advance=1)

        # 3) Filtering
        not_nan = ~torch.isnan(layout_beams_properties[:, 3])
        filtered = layout_beams_properties[not_nan]

        fwhm_vals = filtered[:, fwhm_col]
        fwhm_mask = (fwhm_vals >= fwhm_min) & (fwhm_vals <= fwhm_max)
        filtered = filtered[fwhm_mask]

        if filtered.shape[0] > 0:
            max_sens = filtered[:, 7].max()
            filtered = filtered[filtered[:, 7] > max_sens * 0.01]

        progress.update(task, advance=1)

        # 4) Populate histogram
        asci_histogram = torch.zeros((n_vox, n_bins), dtype=torch.int32)
        angle_bin_col = filtered.shape[1] - 1

        # Wrapped in no_grad to guarantee no memory retention
        with torch.no_grad():
            for beam_props in filtered:
                detector_idx = int(beam_props[1])
                beam_idx = int(beam_props[2])
                angle_bin_idx = int(beam_props[angle_bin_col])

                if 0 <= angle_bin_idx < n_bins:
                    # --- FIX: Reconstruct dense row manually from dict ---
                    if beams_masks_dict["is_sparse"]:
                        detector_mask_row = torch.zeros(n_vox, dtype=torch.int32)
                        start_idx = int(beams_masks_dict["indptr"][detector_idx])
                        end_idx = int(beams_masks_dict["indptr"][detector_idx + 1])
                        
                        if start_idx < end_idx:
                            cols = beams_masks_dict["indices"][start_idx:end_idx]
                            vals = beams_masks_dict["data"][start_idx:end_idx]
                            detector_mask_row[cols] = vals
                    else:
                        detector_mask_row = beams_masks_dict["dense_tensor"][detector_idx]
                        
                    asci_histogram[detector_mask_row == beam_idx, angle_bin_idx] += 1

        # Save output
        out_file = os.path.join(input_dir, f"asci_histogram_{args.layout_idx:02d}.hdf5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("asci_histogram", data=asci_histogram.numpy())
            f["asci_histogram"].attrs["n_bins"] = n_bins
            f["asci_histogram"].attrs["layout_idx"] = args.layout_idx
            f["asci_histogram"].attrs["fwhm_min_mm"] = fwhm_min
            f["asci_histogram"].attrs["fwhm_max_mm"] = fwhm_max
            f["asci_histogram"].attrs["fwhm_column_index"] = fwhm_col

        progress.update(task, advance=1)

    print(f"Saved ASCI histogram to: {out_file}")

if __name__ == "__main__":
    main()