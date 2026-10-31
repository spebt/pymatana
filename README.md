# Imaging System Response Matrix Analysis

| **Review Date**| *2025-08-08*|
|----------------|:-------------|
| **Author** | [Fang Han (fhanonline@gmail.com)](mailto:fhanonline@gmail.com)|
| **Description**| Python-based analysis tools for imaging system response matrix analysis|

## :rocket: Quick Start

### PPDF Beam Analysis Demo/Test

Run the `debug_beams_property` jupyter notebook at `ppdf-analysis/beam-analysis/develop/jupyter_notebooks/debug_beams_property.ipynb`
- This version is compatible with the python scripts used to generate the result for the SNMMI 2025 presentation results

## :gear: Results Production

### Produce the Beams Properties

- `ppdf-analysis/beam-analysis/develop/python_scripts/extract_beams_props_and_masks_2.py`
- `ppdf-analysis/beam-analysis/develop/python_scripts/extract_beams_props_and_masks.py`

### Generate the ASCI Histogram

- `ppdf-analysis/beam-analysis/develop/python_scripts/generate_asci_map_hisotgram.py`

### Produce the ASCI Map

- `ppdf-analysis/beam-analysis/develop/jupyter_notebooks/asci_map_generation.ipynb`

## Design Maps: wFWHM, ASCI, Sensitivity, PPDS

Per-pixel maps of a design, as on the MIC 2026 slides comparing the benchmark and MOBO-optimized SC-SPECT designs. In `beam-analysis/`:

- `arg_design_maps.py` computes the four maps from the PPDFs and the outputs of `arg_extract_beam_masks.py` and `arg_extract_beam_properties.py`, and writes one HDF5 file per design (datasets `wfwhm`, `asci`, `sensitivity`, `ppds`, each `fov.n_pixels` in shape).
- `plot_design_maps.py` plots one or more designs side by side, each map on one colour scale.

```bash
cd beam-analysis
python arg_extract_beam_masks.py 0 --config configs/base_config.yml
python arg_extract_beam_properties.py 0 --config configs/base_config.yml
python arg_design_maps.py --config configs/base_config.yml --layouts 0 1 --out bench.hdf5
python plot_design_maps.py bench.hdf5 mobo.hdf5 --labels Benchmark MOBO --out design_maps.png
python tests/test_design_maps.py
```

For pixel j, strip b of detector i, and p_ibk the strip's PPDF at pixel k:

| Map | Definition |
|-----|------------|
| wFWHM_j | sum (sum_k p_ibk) FWHM_ib / sum (sum_k p_ibk), over the strips whose mask covers j |
| ASCI_j | % of the `analysis.n_bins` strip-angle bins in which a strip covers j: strips above 1% of the peak strip sensitivity, FWHM in (`fwhm_min_mm`, `fwhm_max_mm`] |
| Sensitivity_j | sum_i PPDF_ij |
| PPDS_j | sum over the detectors i whose strips cover j of (sum_k PPDF_ik) / A_i, with A_i = sum_b FWHM_ib x radial width_ib (mm²) |

- `fov` must match the PPDFs, which need one file per layout with every detector as a row (stack per-panel files first). The `maps` section of the config names the input files.
- The PPDF files matched for a layout are averaged: with T8 (`ppdf_files: "position_{layout:03d}_ppdfs_t8_*.hdf5"`), sensitivity and PPDS are per bed position. Layouts are averaged.
- Sensitivity and PPDS are 2-D (in-plane) values; PPDS depends on the pixel size.
- ASCI applies the 1% floor before the FWHM window, `arg_analyze_extracted_properties.py` after it. The MIC maps used `fwhm_max_mm: 0.45` on the 10 mm FOV.
- A strip with no half maximum along the 64 mm sampling line of `get_beam_width` has FWHM 64 mm; such strips are kept, as in the MIC maps.
