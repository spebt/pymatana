import sys
import os
import yaml
import argparse
import h5py
import torch
import gc
from torch import cat, tensor, arange

from scanner_modeling.beam_property_extract import *
from scanner_modeling.convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from scanner_modeling.geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from scanner_modeling.geometry_2d_utils import fov_tensor_dict, pixels_coordinates, pixels_to_detector_unit_rads
from scanner_modeling.beam_property_io import (
    initialize_beam_properties_hdf5,
    append_to_hdf5_dataset,
    stack_beams_properties,
)

# --- FIX 1: Robust Boundary Detection (Prevents Broadcast Crash) ---
def robust_beams_boundaries_radians(
    arc_sampled_ppdf: torch.Tensor, arc_rads: torch.Tensor, 
    relative_threshold: float = 0.01, absolute_floor: float = 10e-7
) -> torch.Tensor:
    """Safe boundary detection that returns an empty tensor instead of crashing."""
    arc_rads_step = arc_rads[1] - arc_rads[0]
    n_samples = arc_rads.shape[0]

    appended_rads = cat((arc_rads, arc_rads[-1:] + arc_rads_step), dim=0)
    
    relative_sampled_ppdf = arc_sampled_ppdf / (arc_sampled_ppdf.max() + 1e-12)
    thresholded_relative_sampled_ppdf = torch.zeros_like(relative_sampled_ppdf).masked_fill_(relative_sampled_ppdf > relative_threshold, 1)

    forward_diff_abs = torch.diff(
        thresholded_relative_sampled_ppdf, prepend=tensor([0.0]), append=tensor([0.0])
    ).abs()

    radian_edges_indices = torch.argwhere(forward_diff_abs > 0.5).squeeze()
    
    if radian_edges_indices.dim() == 0 or radian_edges_indices.numel() < 2:
        return torch.empty(0, 2) 

    n_intervals = radian_edges_indices.shape[0] - 1
    indices_expanded = arange(n_samples).view(1, -1).expand(n_intervals, -1)
    interval_boundaries = torch.stack((radian_edges_indices[:-1], radian_edges_indices[1:]), dim=1)
    interval_boundaries_expanded = interval_boundaries.unsqueeze(1).expand(-1, n_samples, -1)

    interval_masks = (indices_expanded >= interval_boundaries_expanded[:, :, 0]) & (indices_expanded < interval_boundaries_expanded[:, :, 1])
    
    interval_sums = interval_masks.sum(dim=1)
    interval_sums[interval_sums == 0] = 1 
    
    relative_interval_means = relative_sampled_ppdf.unsqueeze(0).expand(n_intervals, -1).clone().masked_fill_(~interval_masks, 0).sum(dim=1) / interval_sums
    absolute_interval_means = arc_sampled_ppdf.unsqueeze(0).expand(n_intervals, -1).clone().masked_fill_(~interval_masks, 0).sum(dim=1) / interval_sums
    
    is_valid_beam = (relative_interval_means > relative_threshold) & (absolute_interval_means > absolute_floor)
    beams_boundaries_indices = interval_boundaries[is_valid_beam]
    
    return appended_rads[beams_boundaries_indices]

# --- FIX 2: PyTorch Beta-Bypass Loader ---
def load_system_matrix_dict(h5_path: str) -> dict:
    """
    Returns a dictionary of 1D tensors to completely avoid instantiating 
    the leaky PyTorch SparseCsrTensor beta object.
    """
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            return {
                "is_sparse": True,
                "indptr": torch.tensor(h5f["indptr"][:], dtype=torch.int64),
                "indices": torch.tensor(h5f["indices"][:], dtype=torch.int64),
                "data": torch.tensor(h5f["data"][:], dtype=torch.float32),
                "shape": tuple(h5f.attrs["shape"])
            }
        elif "ppdfs" in h5f:
            return {
                "is_sparse": False,
                "dense_tensor": torch.tensor(h5f["ppdfs"][:], dtype=torch.float32)
            }
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract physical beam properties for a SPECT layout.")
    parser.add_argument("layout_idx", type=int)
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"--- Starting Beam Property Extraction for Layout Index: {args.layout_idx} ---", flush=True)

    # FOV and IO Setup
    fov_dict = fov_tensor_dict(
        n_pixels=tuple(cfg['fov']['n_pixels']),
        mm_per_pixel=tuple(cfg['fov']['mm_per_pixel'])
    )
    fov_n_pixels_int = int(fov_dict["n pixels"].prod())
    out_dir = cfg['paths']['data_output_dir']
    
    # Initialize HDF5 for properties
    out_hdf5_filename = f"beams_properties_configuration_{args.layout_idx:02d}.hdf5"
    h5_file, prop_dataset = initialize_beam_properties_hdf5(out_hdf5_filename, out_dir)

    # Load Geometry
    scanner_layouts, _ = load_scanner_layouts(cfg['paths']['scanner_layouts_dir'], cfg['paths']['scanner_layouts_filename'])
    _, detector_units_vertices = load_scanner_layout_geometries(args.layout_idx, scanner_layouts)
    
    # Load PPDF Data via Dict Bypass
    ppdfs_hdf5_filepath = os.path.join(out_dir, f"position_{args.layout_idx:03d}_ppdfs.hdf5")
    if not os.path.exists(ppdfs_hdf5_filepath):
        print(f"Error: PPDF file {ppdfs_hdf5_filepath} not found.")
        sys.exit(1)
        
    ppdfs_dict = load_system_matrix_dict(ppdfs_hdf5_filepath)

    # Coordinate Setup
    detector_unit_centers = detector_units_vertices.mean(dim=1)
    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov_dict["size in mm"] * 0.5
    hull_points_batch = sort_points_for_hull_batch_2d(cat((
        fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
        detector_unit_centers.unsqueeze(1)
    ), dim=1))
    fov_points_xy = pixels_coordinates(fov_dict)

    n_detectors = int(detector_units_vertices.shape[0])
    print(f"Extracting properties for {n_detectors} detector units...", flush=True)

    with torch.no_grad():
        for i in range(n_detectors):
            
            # --- FIX 3: Reconstruct dense row manually from dict ---
            if ppdfs_dict["is_sparse"]:
                ppdf_row = torch.zeros(fov_n_pixels_int, dtype=torch.float32)
                start_idx = int(ppdfs_dict["indptr"][i])
                end_idx = int(ppdfs_dict["indptr"][i+1])
                
                if start_idx < end_idx:
                    cols = ppdfs_dict["indices"][start_idx:end_idx]
                    vals = ppdfs_dict["data"][start_idx:end_idx]
                    ppdf_row[cols] = vals
            else:
                ppdf_row = ppdfs_dict["dense_tensor"][i]
                
            ppdf_2d = ppdf_row.view(int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1]))
            hull_2d = convex_hull_2d(hull_points_batch[i])
            
            # Sampling
            sampled, rads, _ = sample_ppdf_on_arc_2d_local(ppdf_2d, detector_unit_centers[i], hull_2d, fov_dict)
            
            # Fast-skip empty beams
            if sampled.max() <= 1e-9:
                if (i + 1) % 200 == 0:
                    print(f"  Processed {i+1}/{n_detectors} units.", flush=True)
                    gc.collect()
                continue

            try:
                boundaries = robust_beams_boundaries_radians(sampled, rads, relative_threshold=cfg['extraction']['relative_threshold'])
            except RuntimeError:
                boundaries = torch.empty(0)

            if boundaries.numel() == 0:
                if (i + 1) % 200 == 0:
                    print(f"  Processed {i+1}/{n_detectors} units.", flush=True)
                    gc.collect()
                continue
                
            fov_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_centers[i])
            masks = get_beams_masks(fov_rads, boundaries)
            
            # Property Calculations
            weighted_centers = get_beams_weighted_center(masks, fov_points_xy, ppdf_2d)
            fwhm, _, _, _ = get_beam_width(
                weighted_centers, detector_unit_centers[i], masks, ppdf_2d, fov_dict,
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
                print(f"  Processed {i+1}/{n_detectors} units.", flush=True)
                gc.collect()

    h5_file.close()
    print(f"--- Beam properties saved for layout {args.layout_idx} ---", flush=True)

if __name__ == "__main__":
    main()