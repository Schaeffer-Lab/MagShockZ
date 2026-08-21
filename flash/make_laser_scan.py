# -*- coding: utf-8 -*-
"""flash/make_laser_scan.py — build the 2D laser ray-trace convergence scan.

Reduces a MagShockZ **3D** ``flash.par`` to a 2D deck (the 3D deck stays the single
source of truth for every physical number: materials, field, pulse, beam geometry,
domain extent in x and y) and then writes one deck per scan point, plus a Slurm
script that runs them.

The scan legs, and what each is testing:

  ``rays``     ``ed_numberOfRays_1`` — whether the ray bundle samples the beam finely
               enough.  The absorbed *total* converges long before the deposition
               *pattern* does: the reflected/refracted halo off the ablation plume is
               resolved by discrete ray tracks, so it is what this leg moves.

  ``grad``     ``ed_gradOrder`` — the interpolation order for nₑ and Tₑ that the ray
               tracer uses both for refraction and for the deposition in a cell.  With
               nₑ/n_c ~ 3e-3 the refraction is negligible, so a large sensitivity here
               would mean the *deposition* is interpolation-limited, not the ray paths.

  ``refine``   the beam channel's own refinement.  The production deck refines on
               ``dens`` and ``magz`` only, so nothing forces resolution onto a channel
               whose Tₑ rises by two orders of magnitude across it; this leg adds
               ``tele`` (and optionally ``depo``) to ``refine_var`` and raises
               ``lrefine_min``.  Par-file only — no source change is needed to refine
               the beam.

Every deck turns on the laser diagnostics the production run left off (``ed_useLaserIO``,
``ed_saveOutOfDomainRays``) and adds ``depo``/``lase`` to ``plot_var``, so
scripts/flash_laser_audit.py can run on the output directly.

Usage
-----
    python flash/make_laser_scan.py \\
        --par-3d /pscratch/sd/d/dschnei/FLASH_MagShockZ3D-corrected/flash.par \\
        --output-dir runs/flash_laser_scan \\
        [--legs rays grad refine] [--tmax 2.0e-9] [--nodes 4] [--hours 6]

Then, in a FLASH tree that has the MagShockZ2D unit installed
(``flash/MagShockZ2D`` copied to ``source/Simulation/SimulationMain/MagShockZ2D``)::

    ./setup -auto MagShockZ2D -2d -nxb=16 -nyb=16 +cartesian +hdf5typeio \\
            species=cham,targ,vac +mtmmmt +laser +usm3t +mgd mgd_meshgroups=1 \\
            -maxblocks=500 -site perlmutter.nersc.gov
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from magshockz.common import flash_utils as fu


# Parameters that only exist in 3D.  The beam's z coordinates are NOT here: FLASH
# still reads ed_lensZ/ed_targetZ in 2D and they are zero, which is what we want.
DROP_KEYS = (
    "zmin", "zmax", "nblockz",
    "zl_boundary_type", "zr_boundary_type",
    "rt_mgdzlboundarytype", "rt_mgdzrboundarytype",
    "diff_elezlboundarytype", "diff_elezrboundarytype",
)

# Overrides applied to every deck in the scan.
BASE_OVERRIDES = {
    # A 2D beam has a 1D cross section: the 3D deck's square2D ray grid and
    # gaussian2D weight have no 2D counterparts, and FLASH aborts on them.
    "ed_gridType_1": '"regular1D"',
    "ed_crossSectionFunctionType_1": '"gaussian1D"',

    # Diagnostics the production run left off.  LaserIO writes per-ray trajectories
    # and power, which is the only way to see *where along its path* a ray lost its
    # energy; saveOutOfDomainRays accounts for the rays that leave.
    "ed_useLaserIO": ".true.",
    "ed_saveOutOfDomainRays": ".true.",
    "ed_printBeams": ".true.",
    "ed_printPulses": ".true.",
    "ed_printMain": ".true.",

    # DEPO must not be reused: with reuse on it holds a previous step's specific
    # energy while FLASH rescales only its own counters, which silently breaks the
    # per-material energy sums in IO_writeIntegralQuantities.
    "ed_depoReuseMaxSteps": "-1",

    # The cumulative laser columns are accumulated once per call, so the routine has
    # to be called on every step.
    "io_integralFreq": "1",

    # Fresh start, and dense output through the pulse: the interesting window is the
    # first ~0.3 ns, when tau in the ambient is largest.
    "restart": ".false.",
    "checkpointFileNumber": "0",
    "plotFileNumber": "0",
    "plotFileIntervalTime": "0.05e-9",
    "checkpointFileIntervalTime": "0.25e-9",
    "rolling_checkpoint": "100",
}

# depo and lase are in unk but the production plot_var list omits them, which is why
# no plot file of the 3D run carries a deposition field at all.
EXTRA_PLOT_VARS = ("depo", "lase")


# The deck's EOS/opacity tables are named by path, and the 3D deck's paths
# (~/FLASH_TABLES/...) no longer exist on Perlmutter.  These are the keys that name
# one, so they can be checked and optionally retargeted.
TABLE_KEYS = ("eos_targTableFile", "eos_chamTableFile", "eos_VacTableFile",
              "op_chamFileName", "op_targFileName", "op_VacFileName")


def retarget_tables(par: dict, table_dir) -> dict:
    """Check that every EOS/opacity table the deck names exists; optionally move it.

    With ``table_dir`` set, each entry keeps its **basename** and is re-rooted there.
    That is not always a like-for-like swap — the 3D deck's paths sit under a
    ``HIGHBOUNDS`` directory, and a file of the same name elsewhere may be a
    differently bounded table — so this is opt-in and reports every substitution
    rather than doing it silently.
    """
    for key in TABLE_KEYS:
        match = [k for k in par if k.lower() == key.lower()]
        if not match:
            continue
        k = match[0]
        val = par[k].strip().strip('"').strip("'")
        if table_dir:
            new = os.path.join(table_dir, os.path.basename(val))
            print("  %s: %s -> %s" % (k, val, new))
            par[k] = '"%s"' % new
            val = new
        if not os.path.exists(os.path.expanduser(val)):
            print("  WARNING: %s points at a missing file: %s" % (k, val))
    return par


def reduce_to_2d(par_path: str, tmax: float) -> dict:
    """Read the 3D deck and return the 2D parameter dict (original key spellings)."""
    raw = _raw_par(par_path)
    out = {k: v for k, v in raw.items() if k.lower() not in DROP_KEYS}

    n_plot = max([int(k.split("_")[-1]) for k in out if k.lower().startswith("plot_var_")]
                 or [0])
    for i, var in enumerate(EXTRA_PLOT_VARS, start=n_plot + 1):
        out["plot_var_%d" % i] = '"%s"' % var

    out.update(BASE_OVERRIDES)
    out["tmax"] = "%.6g" % tmax
    return out


def _raw_par(path: str) -> dict:
    """Parse a deck keeping the *verbatim* right-hand sides and key spellings.

    flash_utils.parse_flash_par coerces values to Python types, which is right for
    analysis and wrong for round-tripping a deck: FLASH cares about ``.true.`` vs
    ``True`` and about quoting, so the text is preserved here instead.
    """
    out = {}
    with open(path) as fh:
        for line in fh:
            body = line.split("#")[0].split("!")[0].strip()
            if "=" not in body:
                continue
            key, _, val = body.partition("=")
            out[key.strip()] = val.strip()
    return out


def write_par(path: str, par: dict, header: str) -> None:
    with open(path, "w") as fh:
        fh.write("# %s\n" % header.replace("\n", "\n# "))
        fh.write("# Generated by flash/make_laser_scan.py — edit the generator, not this.\n\n")
        for k, v in par.items():
            fh.write("%-32s = %s\n" % (k, v))


def scan_points(legs: list, base_rays: int, base_grad: int) -> list:
    """(name, {parameter: value}) for each scan point, deduplicated on the baseline."""
    pts = [("baseline", {})]
    if "rays" in legs:
        for n in (base_rays // 4, base_rays * 4, base_rays * 16):
            pts.append(("rays%d" % n, {"ed_numberOfRays_1": str(n)}))
    if "grad" in legs:
        for g in (1, 2, 3):
            if g != base_grad:
                pts.append(("grad%d" % g, {"ed_gradOrder": str(g)}))
    if "refine" in legs:
        # Refining on tele puts cells where the beam has heated the ambient, which is
        # exactly the channel the production refine_var list ignores.
        pts.append(("refine_tele", {"refine_var_3": '"tele"'}))
        pts.append(("refine_tele_depo", {"refine_var_3": '"tele"',
                                         "refine_var_4": '"depo"'}))
        # A floor on refinement removes the coarse (>100 um) cells the beam crosses
        # far from the target, where nothing triggers the density criterion.
        pts.append(("lrefine_min4", {"lrefine_min": "4"}))
    return pts


SBATCH = """#!/bin/bash
#SBATCH -A m5032
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N {nodes}
#SBATCH -t {hours}:00:00
#SBATCH -J flash_laser_scan
#SBATCH -o %x-%j.out

# One FLASH run per scan point, sequentially: each 2D run is small, and running them
# in sequence keeps the per-run rank count (and therefore the domain decomposition)
# identical across the scan, which a convergence comparison needs.

set -euo pipefail
FLASH4=${{FLASH4:?set FLASH4 to the MagShockZ2D flash4 binary}}

for par in {pars}; do
    name=$(basename "$par" .par)
    mkdir -p "$name"
    cp "$par" "$name/flash.par"
    (cd "$name" && srun -n {ranks} "$FLASH4" -par_file flash.par)
done
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--par-3d", required=True, help="the 3D MagShockZ flash.par")
    p.add_argument("--output-dir", default="runs/flash_laser_scan")
    p.add_argument("--legs", nargs="+", default=["rays", "grad", "refine"],
                   choices=["rays", "grad", "refine"])
    p.add_argument("--tmax", type=float, default=2.0e-9,
                   help="stop time [s]; the default covers the whole pulse")
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--ranks", type=int, default=512)
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--table-dir", default=None,
                   help="re-root every EOS/opacity table on this directory, keeping "
                        "basenames (e.g. ~/EOS_and_opacity_tables/PrOpacEOS). Without "
                        "it the 3D deck's paths are kept and only checked.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = reduce_to_2d(args.par_3d, args.tmax)
    print("EOS / opacity tables:")
    base = retarget_tables(base, args.table_dir)

    typed = fu.parse_flash_par(args.par_3d)
    base_rays = int(typed["ed_numberofrays_1"])
    base_grad = int(typed.get("ed_gradorder", 2))

    written = []
    for name, over in scan_points(args.legs, base_rays, base_grad):
        par = dict(base)
        par.update(over)
        par["run_comment"] = '"MagShockZ2D laser scan: %s"' % name
        path = os.path.join(args.output_dir, "%s.par" % name)
        desc = "\n".join(["MagShockZ2D laser convergence scan point: %s" % name,
                          "reduced from %s" % os.path.abspath(args.par_3d)]
                         + ["%s = %s" % kv for kv in over.items()])
        write_par(path, par, desc)
        written.append(path)
        print("wrote %s" % path)

    sb = os.path.join(args.output_dir, "run_scan.sbatch")
    with open(sb, "w") as fh:
        fh.write(SBATCH.format(nodes=args.nodes, hours=args.hours, ranks=args.ranks,
                               pars=" ".join(os.path.basename(w) for w in written)))
    os.chmod(sb, 0o755)
    print("wrote %s (%d scan points)" % (sb, len(written)))


if __name__ == "__main__":
    main()
