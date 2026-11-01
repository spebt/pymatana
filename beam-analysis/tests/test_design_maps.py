"""Two detectors with one strip each on a 4 x 4 grid of 1 mm pixels, every map checked by hand.

  python tests/test_design_maps.py   (or pytest)
"""
import math
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arg_design_maps import layout_maps, load_matrix, pixel_xy

# detector 0: strip along y over pixels (0, 0..3), PPDF 1 each, FWHM 2 mm
# detector 1: strip along x over pixels (0..3, 0), PPDF 2 each, FWHM 4 mm
PPDF = np.zeros((2, 16))
PPDF[0, [0, 1, 2, 3]] = 1.0
PPDF[1, [0, 4, 8, 12]] = 2.0
MASK = (PPDF > 0).astype(np.int32)
#                 layout det beam angle      FWHM x  y  sens rel npix n
PROPS = np.array([[0, 0, 1, math.pi / 2, 2.0, 0, 0, 1.0, 1, 4, 1],
                  [0, 1, 1, 0.0,         4.0, 0, 0, 1.0, 1, 4, 1]])
RADIAL = 2.355 * math.sqrt(1.25)  # SD of 0, 1, 2, 3 mm


def run(fwhm_window):
    cover = np.zeros((16, 4), dtype=bool)
    sens, ppds, w_num, w_den = layout_maps(PPDF, PROPS, MASK, pixel_xy((4, 4), (1.0, 1.0)), 4, fwhm_window, cover)
    return sens, ppds, np.divide(w_num, w_den, out=np.full(16, np.nan), where=w_den > 0), cover.sum(1) / 4 * 100


def test_maps():
    sens, ppds, wfwhm, asci = run((0.0, 9.0))
    assert np.allclose(sens[[0, 1, 4]], [3.0, 1.0, 2.0])
    assert np.isclose(wfwhm[0], (4 * 2.0 + 8 * 4.0) / 12) and wfwhm[1] == 2.0 and wfwhm[4] == 4.0
    assert np.isnan(wfwhm[5])
    density = [4 / (2.0 * RADIAL), 8 / (4.0 * RADIAL)]
    assert np.allclose(ppds[[0, 1, 4, 5]], [sum(density), density[0], density[1], 0.0])
    assert np.allclose(asci[[0, 1, 4, 5]], [50.0, 25.0, 25.0, 0.0])  # angle bins 1 and 0
    _, _, _, asci = run((0.0, 3.0))  # detector 1's 4 mm strip out of the window
    assert np.allclose(asci[[0, 1, 4]], [25.0, 25.0, 0.0])


def test_csr_matches_dense():
    with tempfile.TemporaryDirectory() as tmp:
        dense, csr = os.path.join(tmp, "dense.hdf5"), os.path.join(tmp, "csr.hdf5")
        with h5py.File(dense, "w") as h5f:
            h5f["ppdfs"] = PPDF
        rows, cols = np.nonzero(PPDF)
        with h5py.File(csr, "w") as h5f:
            h5f["data"], h5f["indices"] = PPDF[rows, cols], cols
            h5f["indptr"] = np.r_[0, np.cumsum(np.bincount(rows, minlength=2))]
            h5f.attrs["shape"] = PPDF.shape
        assert np.array_equal(load_matrix(dense, "ppdfs", float), load_matrix(csr, "ppdfs", float))


if __name__ == "__main__":
    test_maps()
    test_csr_matches_dense()
    print("ok")
