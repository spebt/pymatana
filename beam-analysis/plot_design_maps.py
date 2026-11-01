#!/usr/bin/env python3
"""
Plot design maps (arg_design_maps.py) side by side: one column per design, one row per map,
x right and y up. Each row has one colour scale over the designs, up to the 99.5th percentile,
from 0 (wFWHM: from the 0.5th percentile, as its spread can be a few % of its mean).

  python plot_design_maps.py bench.hdf5 mobo.hdf5 --labels Benchmark MOBO --out design_maps.png
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROWS = (("wfwhm", "wFWHM"), ("asci", "ASCI"), ("sensitivity", "Sensitivity"), ("ppds", "PPDS"))


def main():
    parser = argparse.ArgumentParser(description="Plot design maps side by side.")
    parser.add_argument("maps", nargs="+", help="HDF5 files written by arg_design_maps.py")
    parser.add_argument("--labels", nargs="+", help="Column titles (default: the file names)")
    parser.add_argument("--out", default="design_maps.png")
    args = parser.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.maps]

    designs = []
    for path in args.maps:
        with h5py.File(path, "r") as h5f:
            half = np.asarray(h5f.attrs["n_pixels"]) * np.asarray(h5f.attrs["mm_per_pixel"]) / 2
            designs.append({key: (h5f[key][:], h5f[key].attrs["units"]) for key, _ in ROWS})
            designs[-1]["extent"] = (-half[0], half[0], -half[1], half[1])

    fig, axes = plt.subplots(len(ROWS), len(designs), figsize=(3.4 * len(designs) + 1.2, 3.1 * len(ROWS)),
                             squeeze=False, layout="constrained")
    for r, (key, title) in enumerate(ROWS):
        vmin = min(np.nanpercentile(d[key][0], 0.5) for d in designs) if key == "wfwhm" else 0
        vmax = max(np.nanpercentile(d[key][0], 99.5) for d in designs)
        units = designs[0][key][1].replace("mm^-2", "mm$^{-2}$")
        for c, d in enumerate(designs):
            values = d[key][0]
            im = axes[r, c].imshow(values.T, origin="lower", extent=d["extent"], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"{labels[c]}: {title}, mean {np.nanmean(values):.4g} {units}", fontsize=10)
            axes[r, c].set_xlabel("x (mm)")
            axes[r, c].set_ylabel("y (mm)")
        fig.colorbar(im, ax=axes[r, :], label=f"{title} ({units})" if units else title)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
