#!/usr/bin/env python3
import argparse
import os
import glob
import h5py
import yaml
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch

def load_config(config_path):
    """Loads settings from the centralized YAML config."""
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_beam_data(h5_path, columns=["FWHM (mm)", "detector unit id"]):
    """Extracts specified columns from a beam_properties HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in f["beam_properties"].attrs["Header"]]
        data = torch.from_numpy(f["beam_properties"][:])
        
    results = {}
    for col in columns:
        if col not in header:
            raise RuntimeError(f"Column '{col}' not found in {h5_path}")
        results[col] = data[:, header.index(col)]
    return results

# --- NEW: Auto-detecting Mask Matrix Loader ---
def load_mask_matrix(h5_path: str) -> torch.Tensor:
    """
    Auto-detects sparse/dense mask formats. 
    Returns a dense PyTorch tensor for easy unique() counting.
    """
    with h5py.File(h5_path, 'r') as h5f:
        if "data" in h5f:
            # Load Sparse CSR format via SciPy to avoid PyTorch beta bugs
            data = h5f["data"][:]
            indices = h5f["indices"][:]
            indptr = h5f["indptr"][:]
            shape = tuple(h5f.attrs["shape"])
            csr = sp.csr_matrix((data, indices, indptr), shape=shape)
            return torch.from_numpy(csr.toarray()).to(torch.int32)
        elif "beam_mask" in h5f:
            # Load Legacy Dense format
            return torch.tensor(h5f["beam_mask"][:], dtype=torch.int32)
        else:
            raise ValueError(f"Unknown mask format in {h5_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified Beam Statistics: Histogram, Multiplicity, and Aggregation.")
    # Input modes
    parser.add_argument("--props", help="Path to a single beam_properties_*.hdf5 file.")
    parser.add_argument("--props-dir", help="Directory containing multiple beam_properties_*.hdf5 files.")
    parser.add_argument("--masks", help="Optional: single beams_masks_*.hdf5 (for multiplicity bar chart).")
    
    # Configuration
    parser.add_argument("--config", default="configs/base_config.yml", help="Path to centralized config.")
    parser.add_argument("--out", help="Output directory for plots (overrides config).")
    
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1. Determine Output Path
    out_dir = args.out or (cfg['paths']['data_output_dir'] + "/plots" if cfg else "plots")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Data Gathering
    fwhm_list = []
    det_ids_list = []
    
    if args.props_dir:
        # AGGREGATION MODE (from Script 2)
        search_pattern = os.path.join(args.props_dir, "beams_properties_*.hdf5")
        files = sorted(glob.glob(search_pattern))
        print(f"Aggregating {len(files)} files from {args.props_dir}...")
    elif args.props:
        # SINGLE FILE MODE (from Script 1)
        files = [args.props]
    else:
        print("Error: You must provide either --props or --props-dir.")
        return

    for f_path in files:
        try:
            data = read_beam_data(f_path)
            fwhm_raw = data["FWHM (mm)"].numpy()
            det_id_raw = data["detector unit id"].numpy()
            
            # Filter NaNs
            mask = ~np.isnan(fwhm_raw)
            fwhm_list.extend(fwhm_raw[mask])
            det_ids_list.extend(det_id_raw[mask])
        except Exception as e:
            print(f"Skipping {f_path}: {e}")

    cumulative_fwhm = np.array(fwhm_list)
    cumulative_det_id = np.array(det_ids_list)

    if len(cumulative_fwhm) == 0:
        print("No valid data found.")
        return

    # 3. Statistics & Printouts
    print("-" * 50)
    print(f"Total beams processed: {len(cumulative_fwhm)}")
    
    # Use thresholds from config if available
    f_min = cfg['analysis']['fwhm_min_mm'] if cfg else 2.0
    f_max = cfg['analysis']['fwhm_max_mm'] if cfg else 5.0
    
    good_mask = (cumulative_fwhm >= f_min) & (cumulative_fwhm <= f_max)
    n_good_det = len(np.unique(cumulative_det_id[good_mask]))
    
    print(f"Beams in {f_min}-{f_max}mm window: {np.sum(good_mask)}")
    print(f"Detectors with >=1 'good' beam: {n_good_det}")

    # 4. Figure 1: FWHM Histogram
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.hist(cumulative_fwhm, bins=200 if not args.props_dir else 500, color="#4c72b0", alpha=0.85)
    
    mean_v, med_v = np.mean(cumulative_fwhm), np.median(cumulative_fwhm)
    ax.axvline(mean_v, color="red", ls="--", lw=1.5, label=f"Mean {mean_v:.2f} mm")
    ax.axvline(med_v, color="green", ls=":", lw=1.5, label=f"Median {med_v:.2f} mm")
    
    ax.set_xlabel("Beam FWHM (mm)", fontsize=14)
    ax.set_ylabel("Number of beams", fontsize=14)
    ax.set_title("Distribution of Beam Widths", fontsize=16)
    ax.set_xlim([0, 9]) 
    ax.legend()
    
    fname = "cumulative_fwhm.png" if args.props_dir else "single_fwhm.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300)
    print(f"Saved Histogram → {os.path.join(out_dir, fname)}")

    # 5. Figure 2: Multiplicity
    if args.masks:
        print(f"Generating Multiplicity Plot from {args.masks}...")
        
        # --- FIX APPLIED HERE ---
        masks_data = load_mask_matrix(args.masks)
        
        counts = torch.tensor([(row.unique().numel() - 1) for row in masks_data])
        if counts.numel() > 0 and counts.max() > 0:
            max_k = int(counts.max().item())
            det_per_k = torch.bincount(counts, minlength=max_k+1)[1:]
            ks = np.arange(1, max_k+1)

            fig2, ax2 = plt.subplots(figsize=(7, 4), layout="constrained")
            bars = ax2.bar(ks, det_per_k.numpy(), color="#55a868", alpha=0.9)
            for bar, val in zip(bars, det_per_k.tolist()):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(val), ha="center", va="bottom")
            
            ax2.set_xlabel("Number of beams per detector (k)", fontsize=14)
            ax2.set_ylabel("Number of detectors", fontsize=14)
            ax2.set_title("Beam Multiplicity Distribution", fontsize=16)
            
            fig2.savefig(os.path.join(out_dir, "detector_multiplicity.png"), dpi=300)
            print(f"Saved Multiplicity → {os.path.join(out_dir, 'detector_multiplicity.png')}")

if __name__ == "__main__":
    main()