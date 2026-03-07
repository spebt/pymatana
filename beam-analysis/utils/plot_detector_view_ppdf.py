import os
import sys
import argparse
import yaml
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from torch import tensor, cat, arange

# --- Add local modules to system path ---
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scanner_modeling.beam_property_extract import (
    sample_ppdf_on_arc_2d_local,
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,
    get_beams_basic_properties,
    get_beams_angle_radian
)
from scanner_modeling.convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from scanner_modeling.geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from scanner_modeling.geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)

# --- NEW: Auto-detecting System Matrix Loader ---
def load_system_matrix(h5_path: str) -> torch.Tensor:
    """
    Auto-detects if the HDF5 file is in the optimized Sparse CSR format 
    or the legacy Dense format and returns the corresponding PyTorch tensor.
    """
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            indptr = torch.tensor(h5f["indptr"][:], dtype=torch.int32)
            indices = torch.tensor(h5f["indices"][:], dtype=torch.int32)
            data = torch.tensor(h5f["data"][:], dtype=torch.float32)
            shape = tuple(h5f.attrs["shape"])
            return torch.sparse_csr_tensor(indptr, indices, data, size=shape)
        elif "ppdfs" in h5f:
            return torch.tensor(h5f["ppdfs"][:], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")

def plot_polygons_from_vertices_mpl(vertices: torch.Tensor, ax: plt.Axes, **kwargs):
    """Draws a collection of polygons on the given axes."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p

def find_beam_boundaries_with_static_filter(
    arc_sampled_ppdf: torch.Tensor, arc_rads: torch.Tensor, 
    relative_threshold: float = 0.01, absolute_floor: float = 10e-7
) -> torch.Tensor:
    """This is the updated beam boundary detection logic."""
    arc_rads_step = arc_rads[1] - arc_rads[0]
    n_samples = arc_rads.shape[0]

    appended_rads = cat((
            arc_rads,
            arc_rads[-1:] + arc_rads_step,
        ), dim=0)
    
    relative_sampled_ppdf = arc_sampled_ppdf / arc_sampled_ppdf.max()
    thresholded_relative_sampled_ppdf = torch.zeros_like(relative_sampled_ppdf).masked_fill_(relative_sampled_ppdf > relative_threshold, 1)

    forward_diff_abs = torch.diff(
        thresholded_relative_sampled_ppdf,
        prepend=tensor([0.0]),
        append=tensor([0.0]),
    ).abs()

    radian_edges_indices = torch.argwhere(forward_diff_abs > 0.5).squeeze()
    if radian_edges_indices.dim() == 0 or radian_edges_indices.numel() < 2:
        return torch.empty(0, 2) # No intervals found

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


def main():
    parser = argparse.ArgumentParser(description="Plot detector view of PPDF.")
    parser.add_argument("layout_idx", type=int, help="Index of the scanner layout")
    parser.add_argument("detector_idx", type=int, help="Index of the detector unit to analyze")
    parser.add_argument("--config", default="configs/base_config.yml", help="Path to config file")
    parser.add_argument("--zoom", action="store_true", help="Zoom in on plates in the plot")
    parser.add_argument("--zoom_margin", type=float, default=20.0, help="Margin around plates when zooming (mm)")
    parser.add_argument("--out", default="detector_view_output.png", help="Output filename for the plot")
    args = parser.parse_args()

    # Load centralized configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- Setup Variables from Config ---
    SCANNER_LAYOUTS_DIR = cfg['paths']['scanner_layouts_dir']
    SCANNER_LAYOUTS_FILENAME = cfg['paths']['scanner_layouts_filename']
    PPDFS_DATASET_DIR = cfg['paths']['data_output_dir']

    FOV_DICT = fov_tensor_dict(
        n_pixels=tuple(cfg['fov']['n_pixels']),
        mm_per_pixel=tuple(cfg['fov']['mm_per_pixel']),
        center_coordinates=tuple(cfg['fov']['center_coordinates']),
    )

    # --- 2. Load Geometry and PPDF Data ---
    scanner_layouts_data, _ = load_scanner_layouts(SCANNER_LAYOUTS_DIR, SCANNER_LAYOUTS_FILENAME)
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        args.layout_idx, scanner_layouts_data
    )
    detector_unit_verts = detector_units_vertices[args.detector_idx]
    detector_unit_center = detector_unit_verts.mean(dim=0)

    # Load System Matrix with Auto-Detection
    ppdfs_hdf5_filename = f"position_{args.layout_idx:03d}_ppdfs.hdf5"
    ppdfs_filepath = os.path.join(PPDFS_DATASET_DIR, ppdfs_hdf5_filename)
    
    if not os.path.exists(ppdfs_filepath):
        print(f"Error: PPDF file {ppdfs_filepath} not found.")
        sys.exit(1)
        
    all_ppdfs = load_system_matrix(ppdfs_filepath)

    # --- NEW: Dynamic Row Decompression ---
    if all_ppdfs.is_sparse_csr:
        ppdf_data_1d = all_ppdfs[args.detector_idx].to_dense()
    else:
        ppdf_data_1d = all_ppdfs[args.detector_idx]
        
    ppdf_data_2d = ppdf_data_1d.view(int(FOV_DICT["n pixels"][0]), int(FOV_DICT["n pixels"][1]))

    print(f"Data loaded for Layout {args.layout_idx}, Detector Unit {args.detector_idx}")
    print(f"Detector Unit Center: {detector_unit_center.tolist()}")

    # --- 3. Run Beam Segmentation and Property Extraction ---
    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * FOV_DICT["size in mm"] * 0.5
    hull_points_for_sampling = cat((fov_corners, detector_unit_center.unsqueeze(0)))
    sorted_hull_points = sort_points_for_hull_batch_2d(hull_points_for_sampling.unsqueeze(0)).squeeze(0)
    hull_2d = convex_hull_2d(sorted_hull_points)

    sampled_ppdf, sampling_rads, sampling_points = sample_ppdf_on_arc_2d_local(
        ppdf_data_2d, detector_unit_center, hull_2d, FOV_DICT
    )

    beam_boundaries_rads = find_beam_boundaries_with_static_filter(sampled_ppdf, sampling_rads)

    if beam_boundaries_rads.numel() == 0:
        print("Error: No beams were found for the selected detector unit with the current thresholds.")
        sys.exit(1)

    fov_points_xy = pixels_coordinates(FOV_DICT)
    fov_points_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_center)

    beams_masks = get_beams_masks(fov_points_rads, beam_boundaries_rads)
    beams_weighted_centers = get_beams_weighted_center(beams_masks, fov_points_xy, ppdf_data_2d)

    beams_fwhm, x_bounds_batch, sampled_beams_data, beam_sp_distance = get_beam_width(
        beams_weighted_centers, detector_unit_center, beams_masks, ppdf_data_2d, FOV_DICT
    )

    print(f"Found {beams_masks.shape[0]} beams.")
    for i in range(beams_fwhm.shape[0]):
        print(f"  - Beam {i+1}: FWHM = {beams_fwhm[i]:.4f} mm, Center = {beams_weighted_centers[i].tolist()}")

    # --- 4. Plotting Setup ---
    fig, axs = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)

    # --- Left Plot: Scanner Geometry ---
    ax = axs[0]
    plot_polygons_from_vertices_mpl(
        plates_vertices, ax=ax, fc='purple', ec='purple', alpha=0.4, label='Collimator Plates'
    )
    plot_polygons_from_vertices_mpl(
        detector_units_vertices, ax=ax, fc='#FFDAB9', ec='black', alpha=0.6, label='All Detector Units'
    )

    fov_size = FOV_DICT["size in mm"]
    center_coords = FOV_DICT["center coordinates in mm"]
    ppdf_extent = [
        (center_coords[0] - fov_size[0] / 2).item(), (center_coords[0] + fov_size[0] / 2).item(),
        (center_coords[1] - fov_size[1] / 2).item(), (center_coords[1] + fov_size[1] / 2).item(),
    ]

    ax.imshow(
        ppdf_data_2d.T, origin="lower", extent=ppdf_extent, cmap="hot_r", aspect='equal'
    )

    x_min_ppdf, x_max_ppdf, y_min_ppdf, y_max_ppdf = ppdf_extent
    ppdf_boundary_verts = torch.tensor([
        [x_min_ppdf, y_min_ppdf], [x_max_ppdf, y_min_ppdf], 
        [x_max_ppdf, y_max_ppdf], [x_min_ppdf, y_max_ppdf]
    ])
    plot_polygons_from_vertices_mpl(
        ppdf_boundary_verts.unsqueeze(0), ax=ax, facecolor='none', edgecolor='green', linewidth=2, label='PPDF Boundary'
    )

    for i, beam_center in enumerate(beams_weighted_centers):
        line_x = [detector_unit_center[0].item(), beam_center[0].item()]
        line_y = [detector_unit_center[1].item(), beam_center[1].item()]
        ax.plot(line_x, line_y, '--', color=f'C{i}', lw=1.5)

    ax.scatter(
        beams_weighted_centers[:, 0], beams_weighted_centers[:, 1],
        c='cyan', marker='x', s=100, zorder=5, label='Beam Centers'
    )

    intensity_scaling_factor = 40.0
    direction_vectors = sampling_points - detector_unit_center
    norm_vectors = direction_vectors / torch.norm(direction_vectors, dim=1, keepdim=True)
    displacements = norm_vectors * sampled_ppdf.unsqueeze(1) * intensity_scaling_factor
    visualized_curve_points = sampling_points + displacements

    ax.plot(visualized_curve_points[:, 0], visualized_curve_points[:, 1], '-', color='cyan', lw=2, label='PPDF Intensity Profile')

    arc_start_point = sampling_points[0]
    arc_end_point = sampling_points[-1]
    ax.plot(
        [detector_unit_center[0].item(), arc_start_point[0].item()],
        [detector_unit_center[1].item(), arc_start_point[1].item()],
        '--', color='pink', lw=1.5, label='Sampling Arc Boundary'
    )
    ax.plot(
        [detector_unit_center[0].item(), arc_end_point[0].item()],
        [detector_unit_center[1].item(), arc_end_point[1].item()],
        '--', color='pink', lw=1.5
    )

    ax.set_title(f"Geometry and Intensity Profile (Layout {args.layout_idx}, Det {args.detector_idx})", fontsize=17)
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)

    if args.zoom:
        all_x = plates_vertices[..., 0]
        all_y = plates_vertices[..., 1]
        min_x, max_x = all_x.min().item(), all_x.max().item()
        min_y, max_y = all_y.min().item(), all_y.max().item()
        ax.set_xlim(min_x - args.zoom_margin, max_x + args.zoom_margin)
        ax.set_ylim(min_y - args.zoom_margin, max_y + args.zoom_margin)
    else:
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- Middle Plot: Line Profiles and FWHM ---
    ax = axs[1]
    for i in range(sampled_beams_data.shape[0]):
        line_profile = sampled_beams_data[i]
        color = f'C{i}'
        ax.plot(beam_sp_distance, line_profile, label=f'Beam {i+1}', color=color)
        half_max = line_profile.max() / 2
        ax.plot(x_bounds_batch[i], [half_max, half_max], '--', color=color, lw=2)
        ax.text(0, half_max, f'FWHM: {beams_fwhm[i]:.2f} mm', color=color, ha='left', va='bottom', fontsize=17, fontweight='bold')
        
    ax.set_title('Beam Spatial Profiles (FWHM)', fontsize=18)
    ax.set_xlabel('Distance along sampling line (mm)', fontsize=18)
    ax.set_ylabel('PPDF Intensity', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Right Plot: Angular Profile Visualization ---
    ax = axs[2]
    sampling_degrees = np.rad2deg(sampling_rads.numpy())
    ax.plot(sampling_degrees, sampled_ppdf, color='black', lw=1.5, label='Arc PPDF Profile')
    for i in range(beam_boundaries_rads.shape[0]):
        start_rad, end_rad = beam_boundaries_rads[i]
        color = f'C{i}'
        beam_mask = (sampling_rads >= start_rad) & (sampling_rads <= end_rad)
        ax.fill_between(
            sampling_degrees, sampled_ppdf, where=beam_mask.numpy(),
            color=color, alpha=0.4, label=f'Beam {i+1}'
        )
        beam_ppdf_slice = sampled_ppdf[beam_mask]
        if beam_ppdf_slice.numel() > 0:
            peak_intensity = beam_ppdf_slice.max()
            peak_angle_deg = sampling_degrees[beam_mask][torch.argmax(beam_ppdf_slice)]
            ax.text(
                peak_angle_deg, peak_intensity, f'{peak_intensity:.2e}',
                ha='center', va='bottom', color=color, fontsize=18, fontweight='bold'
            )
            
    ax.set_title('Beam Angular Profiles', fontsize=18)
    ax.set_xlabel('Angle (degrees)', fontsize=18)
    ax.set_ylabel('PPDF Intensity', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(args.out)
    print(f"Successfully saved plot to {args.out}")

if __name__ == "__main__":
    main()