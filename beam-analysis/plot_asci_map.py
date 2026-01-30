import os
import h5py
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Plot ASCI map from angular sensitivity histograms.")
    parser.add_argument(
        "--config", 
        default="configs/analysis_config.yml", 
        help="Path to the centralized YAML configuration file."
    )
    # Allows overriding the layout sequence from CLI if needed (e.g., --layouts 0 24)
    parser.add_argument(
        "--layouts", 
        type=int, 
        nargs="+", 
        help="Specific layout indices to aggregate. If not provided, defaults to layout 0."
    )
    args = parser.parse_args()

    # --- 2. Load Configuration ---
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        return
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Extract parameters from config
    input_dir = cfg['paths']['data_output_dir']
    plot_dir = os.path.join(input_dir, "plots")
    n_bins = cfg['analysis']['n_bins']
    
    # Calculate FOV side from pixels and resolution
    fov_pixels = cfg['fov']['n_pixels']
    fov_res = cfg['fov']['mm_per_pixel']
    fov_side_mm = fov_pixels[0] * fov_res[0]
    
    # Set layout sequence (default to layout 0 if no CLI args provided)
    layout_seq = args.layouts if args.layouts else [0]
    
    os.makedirs(plot_dir, exist_ok=True)

    # --- 3. Aggregate Histograms ---
    # Initialize the master histogram (Pixels x Angular Bins)
    asci_hist = torch.zeros(fov_pixels[0] * fov_pixels[1], n_bins, dtype=torch.int32)

    print(f"Aggregating {len(layout_seq)} layout histograms...")
    for idx in layout_seq:
        h_path = os.path.join(input_dir, f"asci_histogram_{idx:02d}.hdf5")
        if not os.path.exists(h_path):
            print(f"  Warning: File not found, skipping: {h_path}")
            continue
            
        with h5py.File(h_path, "r") as f:
            asci_hist += torch.from_numpy(f["asci_histogram"][...])

    # Calculate ASCI score: percentage of angular bins with non-zero sensitivity
    asci_map = torch.count_nonzero(asci_hist, dim=1).float() / n_bins

    # --- 4. Visualization ---
    print("Generating ASCI map plot...")
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")

    # extent defines the (left, right, bottom, top) boundaries in mm
    img_extent = (-fov_side_mm/2, fov_side_mm/2, -fov_side_mm/2, fov_side_mm/2)

    im = ax.imshow(
        asci_map.view(fov_pixels[0], fov_pixels[1]).T,
        extent=img_extent,
        origin="lower", 
        cmap="viridis", 
        vmin=0, 
        vmax=1
    )

    # Colorbar configuration
    cbar = fig.colorbar(im, ax=ax, label="ASCI")
    cbar.set_label('ASCI (%)', size=18)
    cbar.ax.tick_params(labelsize=12)
    # Format colorbar as percentage
    cbar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    cbar.update_ticks()

    # Axis styling
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.set_title(f"ASCI Map: max {asci_map.max():.2%}, min {asci_map.min():.2%}", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the output
    out_path = os.path.join(plot_dir, "asci_map.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    
    print(f"Success! ASCI map saved to: {out_path}")

if __name__ == "__main__":
    main()