import torch
import h5py
import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate ASCI angular sensitivity histograms.")
    parser.add_argument("layout_idx", type=int)
    parser.add_argument("--config", default="configs/analysis_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"--- Starting ASCI generation for layout index: {args.layout_idx} ---")

    # Configuration from YAML
    n_bins = cfg['analysis']['n_bins']
    angular_bin_boundaries = torch.arange(n_bins + 1) / (n_bins/2) * torch.pi
    input_dir = cfg['paths']['data_output_dir']
    F_MIN, F_MAX = cfg['analysis']['fwhm_min_mm'], cfg['analysis']['fwhm_max_mm']

    # 1. Data Loading
    try:
        prop_file = f"beams_properties_configuration_{args.layout_idx:02d}.hdf5"
        mask_file = f"beams_masks_configuration_{args.layout_idx:02d}.hdf5"
        
        with h5py.File(os.path.join(input_dir, prop_file), "r") as f:
            layout_props = torch.from_numpy(f["beam_properties"][:])
        
        with h5py.File(os.path.join(input_dir, mask_file), "r") as f:
            beams_masks = torch.from_numpy(f["beam_mask"][:])
            
    except FileNotFoundError:
        print(f"Error: Missing HDF5 files in {input_dir}. Run extraction first.")
        sys.exit(1)

    # 2. Data Processing and Filtering
    # Digitize angles into bins
    digitized_angles = torch.bucketize(layout_props[:, 3], angular_bin_boundaries, right=False)
    layout_props = torch.cat((layout_props, (digitized_angles - 1).unsqueeze(1).float()), dim=1)
    
    # Filter by FWHM and sensitivity (keeping beams > 1% of max sensitivity)
    fwhm_col = cfg['analysis']['fwhm_column_index']
    valid_mask = (layout_props[:, fwhm_col] >= F_MIN) & (layout_props[:, fwhm_col] <= F_MAX)
    filtered_props = layout_props[valid_mask]
    
    if filtered_props.shape[0] > 0:
        max_sens = filtered_props[:, 7].max()
        filtered_props = filtered_props[filtered_props[:, 7] > max_sens * 0.01]

    # 3. Histogram Population
    asci_histogram = torch.zeros((512 * 512, n_bins), dtype=torch.int32)
    for beam in filtered_props:
        det_idx, beam_id = int(beam[1]), int(beam[2])
        angle_bin = int(beam[-1])
        
        if 0 <= angle_bin < n_bins:
            # Map pixels from mask to the angular histogram
            asci_histogram[beams_masks[det_idx] == beam_id, angle_bin] += 1

    # 4. Save Output
    out_file = os.path.join(input_dir, f"asci_histogram_{args.layout_idx:02d}.hdf5")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram.numpy())

    print(f"--- ASCI Histogram saved to: {out_file} ---")

if __name__ == "__main__":
    main()