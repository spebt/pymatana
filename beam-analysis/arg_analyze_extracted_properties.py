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

    # --- Files (match your new naming convention) ---
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
    # Use 361 edges (0..2π inclusive), then clamp bin indices into [0,359].
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
            # header optional, kept for parity with original
            _header = f["beam_properties"].attrs.get("Header", None)

        with h5py.File(mask_path, "r") as f:
            beams_masks = torch.from_numpy(f["beam_mask"][:])  # shape: (N_detectors, N_pixels_flat)

        progress.update(task, advance=1)

        # 2) Digitize angles (ONLY ONCE — fixes original duplication bug)
        # Column 3 is angle, consistent with your original script usage.
        angles = layout_beams_properties[:, 3]
        digitized_angles = torch.bucketize(angles, angular_bin_boundaries, right=False) - 1
        digitized_angles = digitized_angles.clamp(0, n_bins - 1)

        layout_beams_properties = torch.cat(
            (layout_beams_properties, digitized_angles.unsqueeze(1).float()), dim=1
        )

        progress.update(task, advance=1)

        # 3) Filtering (keep original spirit, but config-driven)
        # Remove NaN angles
        not_nan = ~torch.isnan(layout_beams_properties[:, 3])
        filtered = layout_beams_properties[not_nan]

        # FWHM range filter from config
        # (Original used "<4" hardcoded; we now use [fwhm_min, fwhm_max] from YAML.)
        fwhm_vals = filtered[:, fwhm_col]
        fwhm_mask = (fwhm_vals >= fwhm_min) & (fwhm_vals <= fwhm_max)
        filtered = filtered[fwhm_mask]

        # Sensitivity filter: keep beams > 1% of max sensitivity (after FWHM filter)
        # Original used column 7.
        if filtered.shape[0] > 0:
            max_sens = filtered[:, 7].max()
            filtered = filtered[filtered[:, 7] > max_sens * 0.01]

        progress.update(task, advance=1)

        # 4) Populate histogram
        asci_histogram = torch.zeros((n_vox, n_bins), dtype=torch.int32)

        # Last appended column is the bin index
        angle_bin_col = filtered.shape[1] - 1

        for beam_props in filtered:
            detector_idx = int(beam_props[1])
            beam_idx = int(beam_props[2])
            angle_bin_idx = int(beam_props[angle_bin_col])

            # Safety (should already be clamped)
            if 0 <= angle_bin_idx < n_bins:
                asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

        # Save output (keep original filename style; layout-specific)
        out_file = os.path.join(input_dir, f"asci_histogram_{args.layout_idx:02d}.hdf5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("asci_histogram", data=asci_histogram.numpy())
            # optional: store metadata from config
            f["asci_histogram"].attrs["n_bins"] = n_bins
            f["asci_histogram"].attrs["layout_idx"] = args.layout_idx
            f["asci_histogram"].attrs["fwhm_min_mm"] = fwhm_min
            f["asci_histogram"].attrs["fwhm_max_mm"] = fwhm_max
            f["asci_histogram"].attrs["fwhm_column_index"] = fwhm_col

        progress.update(task, advance=1)

    print(f"Saved ASCI histogram to: {out_file}")


if __name__ == "__main__":
    main()
