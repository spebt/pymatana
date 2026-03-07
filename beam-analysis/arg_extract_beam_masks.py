import sys
import os
import yaml
import argparse
import h5py
import torch
import gc
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

# --- FIX 1: PyTorch Beta-Bypass Loader ---
# This entirely avoids `torch.sparse_csr_tensor` which causes the memory leak and warning.
def load_system_matrix_dict(h5_path: str) -> dict:
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
    parser = argparse.ArgumentParser(description="Extract beam masks for a specific SPECT layout.")
    parser.add_argument("layout_idx", type=int, help="Index of the layout to process.")
    parser.add_argument("--config", default="configs/base_config.yml", help="Path to the YAML config.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"--- Starting Beam Mask Extraction for Layout Index: {args.layout_idx} ---", flush=True)

    # 1. FOV Setup
    fov_dict = fov_tensor_dict(
        n_pixels=tuple(cfg['fov']['n_pixels']),
        mm_per_pixel=tuple(cfg['fov']['mm_per_pixel']),
        center_coordinates=tuple(cfg['fov']['center_coordinates']),
    )
    fov_n_pixels_int = int(fov_dict["n pixels"].prod())
    out_dir = cfg['paths']['data_output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # 2. Geometry Setup
    scanner_layouts_data, _ = load_scanner_layouts(
        cfg['paths']['scanner_layouts_dir'], 
        cfg['paths']['scanner_layouts_filename']
    )
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        args.layout_idx, scanner_layouts_data
    )
    n_detectors = detector_units_vertices.shape[0]

    # --- FIX 2: Pre-allocate HDF5 Dataset ---
    # We bypass `initialize_beam_masks_hdf5` and `append_to_hdf5_dataset`.
    # Resizing an HDF5 dataset 1364 times inside a loop causes huge RAM overhead.
    # Pre-allocating directly fixes this entirely and defaults empty rows to zero.
    out_hdf5_filename = f"beams_masks_configuration_{args.layout_idx:02d}.hdf5"
    out_hdf5_filepath = os.path.join(out_dir, out_hdf5_filename)
    
    h5_file = h5py.File(out_hdf5_filepath, "w")
    beams_masks_dataset = h5_file.create_dataset(
        "beam_mask", 
        shape=(n_detectors, fov_n_pixels_int), 
        dtype="int32",
        chunks=(1, fov_n_pixels_int), # Optimizes read/write row by row
        compression="gzip"
    )

    # 3. Load Matrix
    ppdfs_hdf5_filepath = os.path.join(out_dir, f"position_{args.layout_idx:03d}_ppdfs.hdf5")
    if not os.path.exists(ppdfs_hdf5_filepath):
        print(f"Error: PPDF file {ppdfs_hdf5_filepath} not found.")
        sys.exit(1)
        
    ppdfs_dict = load_system_matrix_dict(ppdfs_hdf5_filepath)

    detector_unit_centers = detector_units_vertices.mean(dim=1)
    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov_dict["size in mm"] * 0.5
    
    hull_points_batch = cat((
        fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
        detector_unit_centers.unsqueeze(1),
    ), dim=1)
    hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)
    fov_points_xy = pixels_coordinates(fov_dict)

    # 4. Processing Loop
    with torch.no_grad():
        for i in range(n_detectors):
            
            # Extract row securely from the dict
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
            
            sampled_ppdf, sampling_rads, _ = sample_ppdf_on_arc_2d_local(
                ppdf_2d, detector_unit_centers[i], hull_2d, fov_dict
            )
            
            # --- FIX 3: Fast empty beam skipping ---
            # If the beam is empty, we don't need to write to HDF5 because 
            # the pre-allocated HDF5 dataset is ALREADY natively filled with zeros!
            if sampled_ppdf.max() <= 1e-9:
                if (i + 1) % 200 == 0:
                    print(f"  Processed {i+1}/{n_detectors} detector units.", flush=True)
                    gc.collect()
                continue
                
            try:
                boundaries = beams_boundaries_radians(
                    sampled_ppdf, sampling_rads, 
                    threshold=cfg['extraction']['relative_threshold']
                )
            except RuntimeError:
                # Catch the 'broadcast shape' PyTorch internal error if a beam is exceptionally noisy
                boundaries = torch.empty(0)
                
            if boundaries.numel() > 0:
                fov_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_centers[i])
                masks = get_beams_masks(fov_rads, boundaries)
                combined_mask = get_beams_combined_mask(masks)
                
                # Write directly to the pre-allocated row
                beams_masks_dataset[i, :] = combined_mask.numpy()
                
                del masks
                del combined_mask

            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{n_detectors} detector units.", flush=True)
                gc.collect()

    h5_file.close()
    print(f"--- Finished Mask Extraction for Layout Index: {args.layout_idx} ---", flush=True)

if __name__ == "__main__":
    main()