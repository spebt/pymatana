import os
import h5py
import torch
import yaml
import argparse
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Plot ASCI map by automatically detecting all histograms in the folder.")
    parser.add_argument(
        "--config", 
        default="configs/base_config.yml", 
        help="Path to the centralized YAML configuration file."
    )
    # Optional override: if provided, only these layouts are used.
    parser.add_argument(
        "--layouts", 
        type=int, 
        nargs="+", 
        help="Specific layout indices to aggregate. If omitted, all files in the folder are used."
    )
    args = parser.parse_args()

    # --- 2. Load Configuration ---
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        return
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    input_dir = cfg['paths']['data_output_dir']
    plot_dir = os.path.join(input_dir, "plots")
    n_bins = cfg['analysis']['n_bins']
    
    fov_pixels = cfg['fov']['n_pixels']
    fov_res = cfg['fov']['mm_per_pixel']
    fov_side_mm = fov_pixels[0] * fov_res[0]
    
    os.makedirs(plot_dir, exist_ok=True)

    # --- 3. Dynamic File Detection ---
    if args.layouts:
        # Use specific layouts requested by the user
        file_list = [os.path.join(input_dir, f"asci_histogram_{idx:02d}.hdf5") for idx in args.layouts]
    else:
        # Automatically find all matching histogram files in the directory
        search_pattern = os.path.join(input_dir, "asci_histogram_*.hdf5")
        file_list = sorted(glob.glob(search_pattern))
        
    if not file_list:
        print(f"Error: No histogram files found in {input_dir}")
        return

    # --- 4. Aggregate Histograms ---
    asci_hist = torch.zeros(fov_pixels[0] * fov_pixels[1], n_bins, dtype=torch.int32)

    print(f"Aggregating {len(file_list)} layout histograms...")
    for h_path in file_list:
        if not os.path.exists(h_path):
            print(f"  Warning: File not found: {h_path}")
            continue
            
        print(f"  Processing: {os.path.basename(h_path)}")
        with h5py.File(h_path, "r") as f:
            asci_hist += torch.from_numpy(f["asci_histogram"][...])

    # Calculate ASCI score
    asci_map = torch.count_nonzero(asci_hist, dim=1).float() / n_bins

    # --- 5. Visualization ---
    print("Generating ASCI map plot...")
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    img_extent = (-fov_side_mm/2, fov_side_mm/2, -fov_side_mm/2, fov_side_mm/2)

    im = ax.imshow(
        asci_map.view(fov_pixels[0], fov_pixels[1]).T,
        extent=img_extent,
        origin="lower", 
        cmap="viridis", 
        vmin=0, 
        vmax=1
    )

    cbar = fig.colorbar(im, ax=ax, label="ASCI")
    cbar.set_label('ASCI (%)', size=18)
    cbar.ax.tick_params(labelsize=12)
    cbar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    cbar.update_ticks()

    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.set_title(f"Cumulative ASCI Map ({len(file_list)} layouts)\nmax {asci_map.max():.2%}, min {asci_map.min():.2%}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    out_path = os.path.join(plot_dir, "cumulative_asci_map.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    
    print(f"Success! Cumulative ASCI map saved to: {out_path}")

if __name__ == "__main__":
    main()