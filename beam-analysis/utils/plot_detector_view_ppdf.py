import os
import sys
import torch
import h5py
import matplotlib.pyplot as plt
from torch import tensor, cat, arange

# --- Add local modules to system path ---
# This assumes the notebook is in a directory parallel to the one containing the scripts.
# Adjust the path if your directory structure is different.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# --- Import all necessary functions from our scripts ---
from beam_property_extract import (
    sample_ppdf_on_arc_2d_local,
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,
    get_beams_basic_properties,
    get_beams_angle_radian
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from ppdf_io import load_ppdfs_data_from_hdf5

print("All modules imported successfully!")

def plot_polygons_from_vertices_mpl(vertices: torch.Tensor, ax: plt.Axes, **kwargs):
    """Draws a collection of polygons on the given axes."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


# %% [markdown]
# ## 1. Configuration
# Set the file paths and select the `LAYOUT_INDEX` and `DETECTOR_UNIT_INDEX` you wish to inspect.

# %%
# --- User Configuration ---
LAYOUT_INDEX = 0         # Index of the scanner layout (e.g., 0-23)
DETECTOR_UNIT_INDEX = 0   # Index of the detector unit to analyze

# --- File Paths ---
# Assumes a directory structure like: <project_root>/data/...
SCANNER_LAYOUTS_DIR = "../../../data/scanner_layouts"
SCANNER_LAYOUTS_FILENAME = "mph_hourglass_single_position_base_3mm_18pinholes.tensor"
PPDFS_DATASET_DIR = "../../../data/mph_hourglass_single_position_base_3mm_18pinholes_stationary/outputs"

# --- FOV Definition ---
FOV_DICT = fov_tensor_dict(
    n_pixels=(512, 512),
    mm_per_pixel=(0.25, 0.25),
    center_coordinates=(0.0, 0.0),
)

# --- Define the modified beams_boundaries_radians function with the hardcoded filter ---
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
    # Avoid division by zero if an interval is empty
    interval_sums[interval_sums == 0] = 1
    
    # Calculate relative and absolute means
    relative_interval_means = relative_sampled_ppdf.unsqueeze(0).expand(n_intervals, -1).clone().masked_fill_(~interval_masks, 0).sum(dim=1) / interval_sums
    absolute_interval_means = arc_sampled_ppdf.unsqueeze(0).expand(n_intervals, -1).clone().masked_fill_(~interval_masks, 0).sum(dim=1) / interval_sums
    
    # A beam is valid if it meets BOTH conditions
    is_valid_beam = (relative_interval_means > relative_threshold) & (absolute_interval_means > absolute_floor)
    beams_boundaries_indices = interval_boundaries[is_valid_beam]
    
    return appended_rads[beams_boundaries_indices]

# %% [markdown]
# ## 2. Load Geometry and PPDF Data

# %%
# --- Load Scanner Layout ---
scanner_layouts_data, _ = load_scanner_layouts(SCANNER_LAYOUTS_DIR, SCANNER_LAYOUTS_FILENAME)
plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
    LAYOUT_INDEX, scanner_layouts_data
)
detector_unit_verts = detector_units_vertices[DETECTOR_UNIT_INDEX]
detector_unit_center = detector_unit_verts.mean(dim=0)

# --- Load PPDF Data ---
ppdfs_hdf5_filename = f"position_{LAYOUT_INDEX:03d}_ppdfs.hdf5"
all_ppdfs = load_ppdfs_data_from_hdf5(
    PPDFS_DATASET_DIR, ppdfs_hdf5_filename, FOV_DICT
)
ppdf_data_1d = all_ppdfs[DETECTOR_UNIT_INDEX]
ppdf_data_2d = ppdf_data_1d.view(int(FOV_DICT["n pixels"][0]), int(FOV_DICT["n pixels"][1]))

print(f"Data loaded for Layout {LAYOUT_INDEX}, Detector Unit {DETECTOR_UNIT_INDEX}")
print(f"Detector Unit Center: {detector_unit_center.tolist()}")

# %% [markdown]
# ## 3. Run Beam Segmentation and Property Extraction

# %%
# --- 1. Define Sampling Arc ---
fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * FOV_DICT["size in mm"] * 0.5
hull_points_for_sampling = cat((fov_corners, detector_unit_center.unsqueeze(0)))
sorted_hull_points = sort_points_for_hull_batch_2d(hull_points_for_sampling.unsqueeze(0)).squeeze(0)
hull_2d = convex_hull_2d(sorted_hull_points)

# --- 2. Sample PPDF on Arc ---
sampled_ppdf, sampling_rads, sampling_points = sample_ppdf_on_arc_2d_local(
    ppdf_data_2d, detector_unit_center, hull_2d, FOV_DICT
)

# --- 3. Find Beam Boundaries ---
beam_boundaries_rads = find_beam_boundaries_with_static_filter(sampled_ppdf, sampling_rads)

if beam_boundaries_rads.numel() == 0:
    raise ValueError("No beams were found for the selected detector unit with the current thresholds.")

# --- 4. Get Masks and Properties ---
fov_points_xy = pixels_coordinates(FOV_DICT)
fov_points_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_center)

beams_masks = get_beams_masks(fov_points_rads, beam_boundaries_rads)
beams_weighted_centers = get_beams_weighted_center(beams_masks, fov_points_xy, ppdf_data_2d)

# Get FWHM and other line-based properties
beams_fwhm, x_bounds_batch, sampled_beams_data, beam_sp_distance = get_beam_width(
    beams_weighted_centers, detector_unit_center, beams_masks, ppdf_data_2d, FOV_DICT
)

print(f"Found {beams_masks.shape[0]} beams.")
for i in range(beams_fwhm.shape[0]):
    print(f"  - Beam {i+1}: FWHM = {beams_fwhm[i]:.4f} mm, Center = {beams_weighted_centers[i].tolist()}")

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# --- User Configuration for Plotting ---
ZOOM_IN_ON_PLATES = True  # <--- SET THIS TO TRUE/FALSE
ZOOM_MARGIN_MM = 20.0     # Margin around the plates in mm

# --- Create the Combined Plot ---
fig, axs = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)

# -------------------------------------------------------------------
# --- Left Plot: Scanner Geometry with Overlaid Intensity Arc ---
# -------------------------------------------------------------------
ax = axs[0]

# Plot geometry and PPDF
plot_polygons_from_vertices_mpl(
    plates_vertices, ax=ax, fc='purple', ec='purple',
    alpha=0.4, label='Collimator Plates'
)
plot_polygons_from_vertices_mpl(
    detector_units_vertices, ax=ax, fc='#FFDAB9', ec='black',
    alpha=0.6, label='All Detector Units'
)

fov_size = FOV_DICT["size in mm"]
center_coords = FOV_DICT["center coordinates in mm"]
ppdf_extent = [
    (center_coords[0] - fov_size[0] / 2).item(), (center_coords[0] + fov_size[0] / 2).item(),
    (center_coords[1] - fov_size[1] / 2).item(), (center_coords[1] + fov_size[1] / 2).item(),
]

ax.imshow(
    ppdf_data_2d.T, origin="lower", extent=ppdf_extent,
    cmap="hot_r", aspect='equal'
)

x_min_ppdf, x_max_ppdf, y_min_ppdf, y_max_ppdf = ppdf_extent
ppdf_boundary_verts = torch.tensor([
    [x_min_ppdf, y_min_ppdf], [x_max_ppdf, y_min_ppdf], 
    [x_max_ppdf, y_max_ppdf], [x_min_ppdf, y_max_ppdf]
])
plot_polygons_from_vertices_mpl(
    ppdf_boundary_verts.unsqueeze(0), ax=ax, facecolor='none',
    edgecolor='green', linewidth=2, label='PPDF Boundary'
)
# plot_polygons_from_vertices_mpl(
#     detector_unit_verts.unsqueeze(0), ax=ax, fc='red',
#     ec='black', lw=1.5, label=f'Selected Unit ({DETECTOR_UNIT_INDEX})'
# )

# Draw lines from detector to beam centers and mark centers
for i, beam_center in enumerate(beams_weighted_centers):
    line_x = [detector_unit_center[0].item(), beam_center[0].item()]
    line_y = [detector_unit_center[1].item(), beam_center[1].item()]
    ax.plot(line_x, line_y, '--', color=f'C{i}', lw=1.5)

ax.scatter(
    beams_weighted_centers[:, 0], beams_weighted_centers[:, 1],
    c='cyan', marker='x', s=100, zorder=5, label='Beam Centers'
)

# 2. Calculate and plot the "polar" intensity profile curve
intensity_scaling_factor = 40.0
direction_vectors = sampling_points - detector_unit_center
norm_vectors = direction_vectors / torch.norm(direction_vectors, dim=1, keepdim=True)
displacements = norm_vectors * sampled_ppdf.unsqueeze(1) * intensity_scaling_factor
visualized_curve_points = sampling_points + displacements

ax.plot(visualized_curve_points[:, 0], visualized_curve_points[:, 1], '-', color='cyan', lw=2, label='PPDF Intensity Profile')

# 3. Draw dashed lines to connect the detector to the arc extremes
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

# 4. Finalize Left Plot with ZOOM LOGIC
ax.set_title(f"Scanner Geometry and Intensity Profile for Detector {DETECTOR_UNIT_INDEX}", fontsize=17)
ax.set_xlabel("X (mm)", fontsize=18)
ax.set_ylabel("Y (mm)", fontsize=18)

if ZOOM_IN_ON_PLATES:
    # Extract all X and Y coordinates from the plates vertices
    # plates_vertices shape is likely (N_plates, N_verts, 2)
    all_x = plates_vertices[..., 0]
    all_y = plates_vertices[..., 1]
    
    # Calculate bounds
    min_x, max_x = all_x.min().item(), all_x.max().item()
    min_y, max_y = all_y.min().item(), all_y.max().item()
    
    # Apply bounds with margin
    ax.set_xlim(min_x - ZOOM_MARGIN_MM, max_x + ZOOM_MARGIN_MM)
    ax.set_ylim(min_y - ZOOM_MARGIN_MM, max_y + ZOOM_MARGIN_MM)
    print(f"Zooming in on plates: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
else:
    # Default Full View
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)

# ax.legend(loc='upper right')
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------
# --- Middle Plot: Line Profiles and FWHM ---
# -------------------------------------------------------------------
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
# ax.legend()

# -------------------------------------------------------------------
# --- Right Plot: Angular Profile Visualization ---
# -------------------------------------------------------------------
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
# ax.legend()

# plt.show()
plt.savefig("output2.png")