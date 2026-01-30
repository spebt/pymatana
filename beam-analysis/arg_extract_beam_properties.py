import sys
import os
import yaml
import argparse
import torch
from torch import cat, tensor, arange
from scanner_modeling.beam_property_extract import *
from scanner_modeling.convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from scanner_modeling.geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from scanner_modeling.geometry_2d_utils import fov_tensor_dict, pixels_coordinates, pixels_to_detector_unit_rads
from scanner_modeling.ppdf_io import load_ppdfs_data_from_hdf5
from scanner_modeling.beam_property_io import (
    initialize_beam_properties_hdf5,
    append_to_hdf5_dataset,
    stack_beams_properties,
)

def main():
    parser = argparse.ArgumentParser(description="Extract physical beam properties for a SPECT layout.")
    parser.add_argument("layout_idx", type=int)
    parser.add_argument("--config", default="configs/analysis_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # FOV and IO Setup
    fov_dict = fov_tensor_dict(
        n_pixels=tuple(cfg['fov']['n_pixels']),
        mm_per_pixel=tuple(cfg['fov']['mm_per_pixel'])
    )
    out_dir = cfg['paths']['data_output_dir']
    
    # Initialize HDF5 for properties
    out_hdf5_filename = f"beams_properties_configuration_{args.layout_idx:02d}.hdf5"
    h5_file, prop_dataset = initialize_beam_properties_hdf5(out_hdf5_filename, out_dir)

    # Load Geometry and PPDFs
    scanner_layouts, _ = load_scanner_layouts(cfg['paths']['scanner_layouts_dir'], cfg['paths']['scanner_layouts_filename'])
    _, detector_units_vertices = load_scanner_layout_geometries(args.layout_idx, scanner_layouts)
    ppdfs = load_ppdfs_data_from_hdf5(out_dir, f"position_{args.layout_idx:03d}_ppdfs.hdf5", fov_dict)

    # Coordinate Setup
    detector_unit_centers = detector_units_vertices.mean(dim=1)
    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov_dict["size in mm"] * 0.5
    hull_points_batch = sort_points_for_hull_batch_2d(cat((
        fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
        detector_unit_centers.unsqueeze(1)
    ), dim=1))
    fov_points_xy = pixels_coordinates(fov_dict)

    n_detectors = int(detector_units_vertices.shape[0])
    print(f"Extracting properties for {n_detectors} detector units...")

    for i in range(n_detectors):
        ppdf_2d = ppdfs[i].view(int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1]))
        hull_2d = convex_hull_2d(hull_points_batch[i])
        
        # Sampling and Boundary detection
        sampled, rads, _ = sample_ppdf_on_arc_2d_local(ppdf_2d, detector_unit_centers[i], hull_2d, fov_dict)
        boundaries = beams_boundaries_radians(sampled, rads, threshold=cfg['extraction']['relative_threshold'])
        
        fov_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_centers[i])
        masks = get_beams_masks(fov_rads, boundaries)
        
        if masks.shape[0] == 0: continue

        # Property Calculations
        weighted_centers = get_beams_weighted_center(masks, fov_points_xy, ppdf_2d)
        fwhm, _, _, _ = get_beam_width(
            weighted_centers, detector_unit_centers[i], masks, ppdf_2d, fov_dict,
            line_n_samples=cfg['extraction']['line_n_samples']
        )
        angles = get_beams_angle_radian(weighted_centers, detector_unit_centers[i])
        sizes, rel_sens, abs_sens = get_beams_basic_properties(masks, ppdf_2d, fov_points_xy)

        # Stack into standard HDF5 format
        stacked_props = stack_beams_properties(
            args.layout_idx, i, angles, fwhm, sizes, rel_sens, abs_sens, weighted_centers
        )
        
        if stacked_props.numel():
            append_to_hdf5_dataset(prop_dataset, stacked_props)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{n_detectors} units.")

    h5_file.close()
    print(f"--- Beam properties saved for layout {args.layout_idx} ---")

if __name__ == "__main__":
    main()