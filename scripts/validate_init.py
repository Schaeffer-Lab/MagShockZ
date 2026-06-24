"""scripts/validate_init.py — does the OSIRIS initial condition match FLASH?

The flash2osiris generation pipeline writes interp/*.npy slices of the FLASH
midplane plus a py-script that OSIRIS calls to fill the fields/particles at t=0.
A silent x/y-order or rotation mistake in that py-script does not crash anything;
it just initializes the wrong profile.  This script makes that failure loud.

For each field component it imports the run's OWN py-script (the exact code OSIRIS
runs), drives its `set_fld_int` with a STATE that mirrors the OSIRIS t=0 call, and
overlays the result on the OSIRIS t=0 field dump (MS/FLD/<comp>-savg-000000.h5).
The py-script curve is, by construction, the FLASH interpolant sampled along the
lineout (rotated into OSIRIS components), so the two curves must coincide.  A
mismatch means the deck and the py-script disagree about how OSIRIS coordinates
map onto the FLASH plane.

Usage:  python scripts/validate_init.py <run_dir>
"""
import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import plot_style

FIELD_COMPONENTS = ["b1", "b2", "b3", "e1", "e2", "e3"]


def _import_pyscript(run_dir):
    """Import the run's py-script as a module, with CWD set to run_dir so its
    relative ``interp/<name>.npy`` loads resolve."""
    matches = sorted(run_dir.glob("py-script-*.py"))
    if not matches:
        raise FileNotFoundError(f"no py-script-*.py in {run_dir}")
    path = matches[0]
    os.chdir(run_dir)                       # py-script reads interp/ relatively
    spec = importlib.util.spec_from_file_location("osiris_pyscript", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def _read_t0_dump(run_dir, comp):
    """Return (data, axis_min, axis_max) for the t=0 dump of `comp`, or None."""
    import h5py
    hits = glob.glob(str(run_dir / "MS" / "FLD" / f"{comp}-savg" / f"{comp}-savg-000000.h5"))
    if not hits:
        return None
    with h5py.File(hits[0], "r") as f:
        key = next(k for k in f.keys() if k not in ("AXIS", "SIMULATION"))
        data = np.asarray(f[key])
        ax1 = np.asarray(f["AXIS"]["AXIS1"])    # [min, max]
    return data, float(ax1[0]), float(ax1[1])


def _pyscript_field(mod, comp, shape, x_bnd):
    """Run the py-script's set_fld_int for one component on a fresh STATE."""
    STATE = {"fld": comp, "data": np.zeros(shape), "x_bnd": np.asarray(x_bnd, float)}
    mod.set_fld_int(STATE)
    return STATE["data"]


def validate(run_dir):
    run_dir = Path(run_dir).resolve()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mod, path = _import_pyscript(run_dir)
    print(f"py-script: {path}  (dims={mod.dims}, theta={mod.theta:.4f})")

    if mod.dims != 1:
        raise NotImplementedError("validate_init currently overlays 1D lineouts only")

    print(f"\n{'comp':>5} {'max|rel dev|':>14} {'max|abs dev|':>14}   status")
    worst = 0.0
    for comp in FIELD_COMPONENTS:
        dump = _read_t0_dump(run_dir, comp)
        if dump is None:
            continue
        osiris, x0, x1 = dump
        # OSIRIS' 1D domain runs [0, distance]; the t=0 dump fixes the grid.
        x_bnd = [[0.0, mod.distance]]
        py = _pyscript_field(mod, comp, osiris.shape, x_bnd)

        denom = np.maximum(np.abs(osiris).max(), 1e-30)
        rel = np.abs(py - osiris) / denom
        max_rel, max_abs = float(rel.max()), float(np.abs(py - osiris).max())
        worst = max(worst, max_rel)
        status = "OK" if max_rel < 0.02 else "MISMATCH"
        print(f"{comp:>5} {max_rel:14.3e} {max_abs:14.3e}   {status}")

        s = np.linspace(0.0, mod.distance, osiris.shape[0])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
        a1.plot(s, osiris, "k-", lw=2, label="OSIRIS t=0 dump")
        a1.plot(s, py, "r--", lw=1.6, label="py-script set_fld_int (= FLASH lineout)")
        a1.set_ylabel(comp)
        a1.set_title(f"{comp}: OSIRIS init vs FLASH  (max rel dev {max_rel:.2e}, {status})")
        a1.legend(); a1.grid(True)
        a2.plot(s, py - osiris, "b-", lw=1.2)
        a2.set_xlabel(r"distance along lineout [$c/\omega_{pe}$]")
        a2.set_ylabel("py - OSIRIS"); a2.grid(True)
        plt.tight_layout()
        out = fig_dir / f"validate_{comp}.png"
        plt.savefig(out, dpi=150); plt.close()

    print(f"\nfigures -> {fig_dir}/validate_<comp>.png")
    print(f"worst max relative deviation across components: {worst:.3e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="OSIRIS run directory (contains py-script-*.py and MS/)")
    plot_style.add_publication_arg(ap)
    args = ap.parse_args()
    plot_style.apply(args.publication)
    validate(args.run_dir)
