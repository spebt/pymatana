#!/usr/bin/env python3
"""
Per-pixel design maps from the PPDFs, beam properties and beam masks of a design's layouts.

  wfwhm        weighted FWHM: over the strips (i, b) whose mask covers pixel j and whose PPDF
               is nonzero there, sum_ib (sum_k p_ibk) FWHM_ib / sum_ib (sum_k p_ibk), where
               p_ibk is the PPDF of strip b of detector i at pixel k
  asci         angular sampling completeness index: % of the n_bins strip-angle bins in which
               a strip covers pixel j (strips above 1% of the peak strip sensitivity, FWHM in
               (fwhm_min_mm, fwhm_max_mm])
  sensitivity  sum_i PPDF_ij
  ppds         sum over the detectors i whose strips cover pixel j of (sum_k PPDF_ik) / A_i,
               A_i = sum_b FWHM_ib x radial width_ib in mm^2 (radial width: 2.355 x the
               PPDF-weighted SD of the strip's pixels along the strip angle)

The PPDF files matched for a layout (e.g. the 8 T8 bed positions) are averaged, so sensitivity
and PPDS are per position, then averaged over layouts; ASCI and wFWHM pool the layouts' strips.
Inputs are the outputs of arg_extract_beam_masks.py and arg_extract_beam_properties.py; file
names come from the config's `maps` section.

  python arg_design_maps.py --config configs/base_config.yml --layouts 0 1 [--out maps.hdf5]
"""
import argparse
import glob
import math
import os
import sys

import h5py
import numpy as np
import yaml

SENSITIVITY_FLOOR = 0.01  # as arg_analyze_extracted_properties.py


def load_matrix(path, dense_key, dtype):
    """A dense `dense_key` dataset, or CSR (data, indices, indptr, attrs["shape"]), as a dense array."""
    with h5py.File(path, "r") as h5f:
        if dense_key in h5f:
            return h5f[dense_key][:].astype(dtype)
        if "data" not in h5f:
            raise ValueError(f"Unknown matrix format in {path}")
        n_rows, n_cols = (int(v) for v in h5f.attrs["shape"])
        indptr = h5f["indptr"][:]
        out = np.zeros((n_rows, n_cols), dtype)
        out[np.repeat(np.arange(n_rows), np.diff(indptr)), h5f["indices"][:]] = h5f["data"][:]
        return out


def pixel_xy(n_pixels, mm_per_pixel):
    """Pixel p = ix * ny + iy at (ix * dx, iy * dy) mm, the order of pixels_coordinates."""
    ix, iy = np.divmod(np.arange(n_pixels[0] * n_pixels[1]), n_pixels[1])
    return np.c_[ix * mm_per_pixel[0], iy * mm_per_pixel[1]]


def layout_maps(ppdf, props, mask, xy, n_bins, fwhm_window, cover):
    """Sensitivity and PPDS maps of one layout, its wFWHM sums, and its ASCI coverage OR'ed
    into `cover` (n_pixels x n_bins). Property columns as stack_beams_properties."""
    n_det, n_pix = ppdf.shape
    det, bid = props[:, 1].astype(np.int64), props[:, 2]
    ang, fwhm, sens = props[:, 3], props[:, 4], props[:, 7]
    valid = np.isfinite(ang) & (det >= 0) & (det < n_det)
    good = valid & np.isfinite(fwhm) & (fwhm > 0)
    s_max = np.nanmax(np.where(valid, sens, np.nan)) if valid.any() else 0.0
    angle_bin = np.floor(np.where(valid, ang, 0.0) / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
    in_asci = good & (sens > SENSITIVITY_FLOOR * s_max) & (fwhm > fwhm_window[0]) & (fwhm <= fwhm_window[1])

    w_num, w_den, area = np.zeros(n_pix), np.zeros(n_pix), np.zeros(n_det)
    for s in np.flatnonzero(good):
        i = det[s]
        pix = np.flatnonzero(mask[i] == bid[s])
        if pix.size == 0:
            continue
        p = ppdf[i, pix]
        if in_asci[s]:
            cover[pix, angle_bin[s]] = True
        lit = p > 0
        total = p[lit].sum()
        w_num[pix[lit]] += total * fwhm[s]
        w_den[pix[lit]] += total
        if pix.size >= 3 and p.sum() > 0:
            r = xy[pix] @ np.array([math.cos(ang[s]), math.sin(ang[s])])
            m = p @ r / p.sum()
            var = p @ (r - m) ** 2 / p.sum()
            if var > 0:
                area[i] += fwhm[s] * 2.355 * math.sqrt(var)

    density = np.divide(ppdf.sum(axis=1), area, out=np.zeros(n_det), where=area > 0)
    ppds = np.zeros(n_pix)
    for i in np.flatnonzero(density):
        ppds[mask[i] > 0] += density[i]
    return ppdf.sum(axis=0), ppds, w_num, w_den


def main():
    parser = argparse.ArgumentParser(description="Per-pixel wFWHM, ASCI, sensitivity and PPDS maps of a design.")
    parser.add_argument("--config", default="configs/base_config.yml")
    parser.add_argument("--layouts", type=int, nargs="+", required=True, help="Layout indices of the design.")
    parser.add_argument("--out", help="Output HDF5 (default: <data_output_dir>/design_maps.hdf5).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    data_dir, files = cfg["paths"]["data_output_dir"], cfg["maps"]
    n_pixels, mm_per_pixel = tuple(cfg["fov"]["n_pixels"]), tuple(cfg["fov"]["mm_per_pixel"])
    n_bins = int(cfg["analysis"]["n_bins"])
    fwhm_window = (float(cfg["analysis"]["fwhm_min_mm"]), float(cfg["analysis"]["fwhm_max_mm"]))
    xy = pixel_xy(n_pixels, mm_per_pixel)
    n_pix = xy.shape[0]

    sens, ppds = np.zeros(n_pix), np.zeros(n_pix)
    w_num, w_den = np.zeros(n_pix), np.zeros(n_pix)
    cover = np.zeros((n_pix, n_bins), dtype=bool)
    n_files = []
    for layout in args.layouts:
        ppdf_paths = sorted(glob.glob(os.path.join(data_dir, files["ppdf_files"].format(layout=layout))))
        if not ppdf_paths:
            sys.exit(f"Error: no PPDF file for layout {layout} ({files['ppdf_files']}) in {data_dir}")
        ppdf = sum(load_matrix(p, "ppdfs", np.float64) for p in ppdf_paths) / len(ppdf_paths)
        if ppdf.shape[1] != n_pix:
            sys.exit(f"Error: PPDFs have {ppdf.shape[1]} pixels; fov.n_pixels gives {n_pix}")
        with h5py.File(os.path.join(data_dir, files["properties_file"].format(layout=layout)), "r") as h5f:
            props = h5f["beam_properties"][:]
        mask = load_matrix(os.path.join(data_dir, files["masks_file"].format(layout=layout)), "beam_mask", np.int32)
        s, d, wn, wd = layout_maps(ppdf, props, mask, xy, n_bins, fwhm_window, cover)
        sens += s / len(args.layouts)
        ppds += d / len(args.layouts)
        w_num += wn
        w_den += wd
        n_files.append(len(ppdf_paths))
        print(f"  layout {layout}: {len(ppdf_paths)} PPDF file(s), {props.shape[0]} strips", flush=True)
        del ppdf, mask

    maps = {
        "wfwhm": (np.divide(w_num, w_den, out=np.full(n_pix, np.nan), where=w_den > 0), "mm"),
        "asci": (cover.sum(axis=1) / n_bins * 100.0, "%"),
        "sensitivity": (sens, ""),
        "ppds": (ppds, "mm^-2"),
    }
    out = args.out or os.path.join(data_dir, "design_maps.hdf5")
    with h5py.File(out, "w") as h5f:
        h5f.attrs.update(layouts=args.layouts, ppdf_files_per_layout=n_files, n_pixels=n_pixels,
                         mm_per_pixel=mm_per_pixel, n_bins=n_bins, fwhm_window_mm=fwhm_window)
        for name, (values, units) in maps.items():
            h5f.create_dataset(name, data=values.reshape(n_pixels)).attrs["units"] = units
            print(f"  {name}: mean {np.nanmean(values):.6g} {units}, max {np.nanmax(values):.6g}")
    print(f"--- Design maps saved to {out} ---", flush=True)


if __name__ == "__main__":
    main()
