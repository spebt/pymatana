import sys
import os
import yaml
import argparse
import torch
from torch import tensor, arange, cat
from scanner_modeling.beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_combined_mask,
    sample_ppdf_on_arc_2d_local,
)
from scanner_modeling.convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from scanner_modeling.geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from scanner_modeling.geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from scanner_modeling.ppdf_io import load_ppdfs_data_from_hdf5
from scanner_modeling.beam_property_io import (
    initialize_beam_masks_hdf5,
    append_to_hdf5_dataset,
)

def main():
    parser = argparse.ArgumentParser(description="Extract beam masks for a specific SPECT layout.")
    parser.add_argument("layout_idx", type=int, help="Index of the layout to process.")
    parser.add_argument("--config", default="configs/base_config.yml", help="Path to the YAML config.")
    args = parser.parse_args()

    # Load centralized configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"--- Starting Beam Mask Extraction for Layout Index: {args.layout_idx} ---")

    # 1. FOV and Directory Initialization
    fov_dict = fov_tensor_dict(
        n_pixels=tuple(cfg['fov']['n_pixels']),
        mm_per_pixel=tuple(cfg['fov']['mm_per_pixel']),
        center_coordinates=tuple(cfg['fov']['center_coordinates']),
    )
    fov_n_pixels_int = int(fov_dict["n pixels"].prod())
    out_dir = cfg['paths']['data_output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # 2. Load Scanner Data
    scanner_layouts_data, _ = load_scanner_layouts(
        cfg['paths']['scanner_layouts_dir'], 
        cfg['paths']['scanner_layouts_filename']
    )
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        args.layout_idx, scanner_layouts_data
    )

    # 3. Initialize HDF5 Output
    out_hdf5_filename = f"beams_masks_configuration_{args.layout_idx:02d}.hdf5"
    out_hdf5_file, beams_masks_dataset = initialize_beam_masks_hdf5(
        fov_n_pixels_int, out_hdf5_filename, out_dir
    )

    # 4. Load PPDF Data
    ppdfs_hdf5_filename = f"position_{args.layout_idx:03d}_ppdfs.hdf5"
    ppdfs = load_ppdfs_data_from_hdf5(out_dir, ppdfs_hdf5_filename, fov_dict)

    # 5. Coordinate Preparation
    detector_unit_centers = detector_units_vertices.mean(dim=1)
    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov_dict["size in mm"] * 0.5
    
    hull_points_batch = cat((
        fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
        detector_unit_centers.unsqueeze(1),
    ), dim=1)
    hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)
    fov_points_xy = pixels_coordinates(fov_dict)

    # 6. Main Extraction Loop
    n_detectors = detector_units_vertices.shape[0]
    for i in range(n_detectors):
        ppdf_2d = ppdfs[i].view(int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1]))
        hull_2d = convex_hull_2d(hull_points_batch[i])
        
        # Sample PPDF on arc
        sampled_ppdf, sampling_rads, _ = sample_ppdf_on_arc_2d_local(
            ppdf_2d, detector_unit_centers[i], hull_2d, fov_dict
        )
        
        # Detect boundaries using config thresholds
        boundaries = beams_boundaries_radians(
            sampled_ppdf, sampling_rads, 
            threshold=cfg['extraction']['relative_threshold']
        )
        
        # Mapping to FOV masks
        fov_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_centers[i])
        masks = get_beams_masks(fov_rads, boundaries)
        combined_mask = get_beams_combined_mask(masks)
        
        append_to_hdf5_dataset(beams_masks_dataset, combined_mask)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{n_detectors} detector units.")

    out_hdf5_file.close()
    print(f"--- Finished Mask Extraction for Layout Index: {args.layout_idx} ---")

if __name__ == "__main__":
    main()