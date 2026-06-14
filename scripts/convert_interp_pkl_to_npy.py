"""scripts/convert_interp_pkl_to_npy.py — migrate legacy interp/*.pkl to *.npy.

Older runs stored each field as a pickled scipy RegularGridInterpolator.  Those
pickles break across scipy versions (the internal grid attribute was renamed), and
the pipeline now stores plain .npy grids that are re-interpolated on load.  This
extracts the raw value array out of each pickle (attribute access only -- it never
calls the interpolator, so it works even when the pickle is otherwise unusable) and
writes the matching .npy, letting an existing run be re-run without regenerating the
slices from FLASH.

Usage:  python scripts/convert_interp_pkl_to_npy.py <run_dir> [<run_dir> ...]
"""
import pickle
import sys
from pathlib import Path

import numpy as np


def convert(run_dir):
    interp = Path(run_dir) / "interp"
    pkls = sorted(interp.glob("*.pkl"))
    if not pkls:
        print(f"{run_dir}: no interp/*.pkl found")
        return
    for p in pkls:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        values = np.asarray(obj.__dict__["values"])   # the grid as stored at build
        out = p.with_suffix(".npy")
        np.save(out, values)
        print(f"  {p.name} -> {out.name}  shape {values.shape}")
    print(f"{run_dir}: converted {len(pkls)} field(s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: convert_interp_pkl_to_npy.py <run_dir> [<run_dir> ...]")
    for d in sys.argv[1:]:
        convert(d)
