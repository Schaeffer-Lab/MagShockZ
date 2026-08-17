# MagShockZ user guide

Everything you can run, and what you need active to run it. This is the reference for
using the repository directly — no agent required.

## Install

The library is a real installable package, so `import magshockz` works from a script, a
notebook, a REPL or an sbatch file with no `sys.path` manipulation:

```bash
conda activate analysis && pip install -e .
conda activate osiris2  && pip install -e .
```

Import modules by their real path — the package `__init__`s are deliberately thin:

```python
from magshockz.common import run_spec, plot_style, piston_profile
from magshockz.analysis.flash import experiment_image
from magshockz.init.warpx import units
```

## Two environments, used for disjoint stages

Two conda environments, used for disjoint stages — do not mix them:

- **`osiris2`** — FLASH→OSIRIS initialization and deck generation. The converter itself
  is the standalone **`flash2osiris`** package (pip-installed here; repo at
  `/pscratch/sd/d/dschnei/flash2osiris`); `init_python/` holds only the MagShockZ run
  drivers. Has `yt` + `unyt` for reading FLASH HDF5 plot files and `jinja2` for templating.
- **`analysis`** — OSIRIS analysis (`scripts/`). Has `osh5` / pyVisOS (`osh5io`,
  `osh5def`, `osh5vis`) and `osiris_utils` for reading OSIRIS HDF5 output.  It also
  has `yt` + `unyt`, so the FLASH-side analysis scripts (e.g.
  `run_flash_pressure_partition.py`) run here too — they import `analysis_utils`
  (→ `osiris_utils`) for config/`RunSpec` loading, which only exists in `analysis`.

The package itself depends only on numpy/scipy/unyt + astropy/plasmapy/yt, so anything
in `magshockz/` that does not touch OSIRIS output imports cleanly in either env.

## Shared flags

Every plotting script takes these:

| flag | effect |
|---|---|
| `--config <path>` | the run/analysis config; `$MAGSHOCKZ_SIM_DIR` / `$MAGSHOCKZ_FLASH_DIR` override the directory it names |
| `--publication` (`--pub`) | bump all text to paper/slide sizes; off by default, so saved figures are otherwise unchanged |
| `--units electron\|ion` | `electron` (default) keeps native OSIRIS `c/ωpe` & `1/ωpe`; `ion` rescales lengths by `d_i` and times by the upstream `T_ci` |
| `--output-dir <path>` | override where results land |

`--units ion` needs the upstream `T_ci`. It is read from the config's cached top-level
`t_ci` when present, else measured from the field — and `osiris_tune_shock.py` writes that
key for free, so tune the shock first and every later figure shares one consistent value.

## Commands

Every script below is `--config` driven and prints `--help`. The full inventory is
`ls scripts/` — the prefix says which code it reads.

### Tests

```bash
pytest                                  # full suite (testpaths=tests)
pytest --cov=magshockz --cov-report=term-missing
pytest tests/test_shock.py              # single file
pytest tests/test_shock.py::test_name   # single test
```

### Generate an OSIRIS deck  (`osiris2` env)

```bash
conda activate osiris2
python -m flash_osiris.generator --config runs/perlmutter_1d.run.yaml   # from the repo root
```

Wrappers: `runs/runme_perlmutter_1d.sh`, `runs/runme_perlmutter_2d.sh`,
`runs/run_dx_scan.sh`. Details in [osiris_pipeline.md](osiris_pipeline.md).

### Analyse an OSIRIS run  (`analysis` env)

```bash
conda activate analysis
python scripts/osiris_overview.py --config config/perlmutter_1.3.1d.yaml [--stride 16]
```

Also `osiris_pressure_partition`, `osiris_temperature_ratios`,
`osiris_heating_decomposition`, `osiris_energy_flux`, `osiris_synthetic_diagnostics`,
`osiris_dimensionless_params`, `osiris_convergence_scan`, `osiris_rh_prediction`,
`osiris_validate_init`.

### Tune the shock parameters, then write them back

Interactive; the config keeps its comments. Each command re-renders
`results/<run>/tune_*.png`, so the figure refreshes in your editor as you go.

```bash
# OSIRIS
python scripts/osiris_tune_shock.py --config config/perlmutter_1.3.1d.yaml      # v_shock / x_shock_0
python scripts/osiris_tune_shock.py --config ...yaml --mode regions --dump 400  # per-dump regions

# FLASH: place the front by hand on physical-unit (µm/ns) line-outs, then feed
# flash_rh_prediction.py
python scripts/flash_tune_shock.py --config config/flash_3d_noshield.yaml
python scripts/flash_tune_shock.py --config ...yaml --mode regions --snapshot-idx -1
```

`trajectory` mode writes all three parameters of the straight front
`x(t) = x0 + v·(t − t0)`, including the anchor `t0` (`t <ns>`) — so a shock that forms
mid-run is fitted from its formation time rather than back-extrapolated; `t0` defaults to
the IC dump time when the key is absent. It also caches `t_ci`, which every later
`--units ion` figure then reads.

`regions` mode writes per-dump region boundaries, and on the FLASH side also draws a 2D
`n_e` slice sharing the LOS-distance axis with the line-outs, so the markers fall over the
2D density jump (`--slice-axis {x,y,z}`, `--slice-halfwidth-um`, `--no-slice`).

### Compare to the experiment

```bash
python scripts/flash_experiment_compare.py --config config/flash_3d_2026-07.yaml
python scripts/flash_experiment_compare.py --config ...yaml --fit    # fit the shift, + an r map
python scripts/flash_experiment_compare.py --config ...yaml \
    --t-offset-ns 2.5 --x-offset-mm -1.2 --t-window 0 25
```

The registration is hand-tuned — read `flash_experiment_fit.png` before quoting a fitted
offset. See [flash_analysis.md](flash_analysis.md).

### Movies

```bash
# a diagnostic, quickly
python scripts/osiris_make_movie.py -d <run>/MS --units ion --config config/<run>.yaml
python scripts/osiris_make_movie.py -d <run>/MS/FLD/b2-savg --no-interactive \
    --units ion --config config/<run>.yaml --xlim 80 120 --log -s 4 -o b2

# 3D volume-rendered FLASH (COMPUTE node -- see scripts/flash_3d_movie.sbatch)
python scripts/flash_3d_movie.py --config config/flash_3d_2026-07.yaml \
    --preset trantham2026-07 --fields ne te ti bx by bz bmag
python scripts/flash_3d_movie.py --config ...yaml --preset ... --grids-only  # cache first
```

`--preset` picks that run's tuned camera and transfer functions and must match the
`--config`'s data. Sampling each AMR dump is the slow half and is cached, so `--grids-only`
primes the cache and later renders take seconds.

### WarpX heater-driven piston

Measure the FLASH piston, render the deck, run it, compare. The constraints that shaped
this deck are in [warpx_pipeline.md](warpx_pipeline.md) — read them before changing it.

```bash
bash init_warpx/build_warpx_gpu_2d.sh   # once: the 2D CUDA app carrying the operators

conda activate analysis
python scripts/flash_piston_profile.py --config config/flash_3d_corrected.yaml --t-window 3 12
python scripts/warpx_make_deck.py --config runs/magshockz_2d_heater.warpx.yaml --smoke --no-heater
sbatch init_warpx/run_heater_2d.sbatch  # 4 GPU nodes; HEATER_EXE=<cpu app> to fall back

python scripts/warpx_make_deck.py --config ...yaml --verify   # vs the post-run echo
python scripts/warpx_heater_compare.py --config runs/magshockz_2d_heater.warpx.yaml
python scripts/warpx_flash_evolution.py --config runs/magshockz_2d_heater.warpx.yaml

# the poster figure alone: n / v / T at one matched time + the shocked-layer scorecard
python scripts/warpx_flash_evolution.py --config runs/magshockz_2d_heater.warpx.yaml \
    --figures profiles --cache [--profile-time 0.95]

# piston_comparison.mp4: the same two-species panels over EVERY WarpX dump, FLASH
# beside WarpX. --jobs renders frames in parallel, so run it on a compute node --
# a frame costs ~50 s (the FLASH slice dominates) and there are ~70 of them.
salloc -N 1 -C cpu -A m5032 -t 1:00:00
python scripts/warpx_flash_evolution.py --config runs/magshockz_2d_heater.warpx.yaml \
    --figures compare --cache --movie --jobs 24 [--fps 10]
```

The heater-off null control answers "is the ambient heating numerical?". Give it its own
run directory or it clobbers the production diagnostics:

```bash
HEATER_RUNDIR=input_files/warpx/magshockz_2d_heater_noheater \
HEATER_DECK=input_files/warpx/magshockz_2d_heater/inputs_2d_heater_noheater \
  sbatch init_warpx/run_heater_2d.sbatch
```

Both directions of the round-trip. Neither writes anything, so both are safe against a
running job; both exit 1 on drift:

```bash
python scripts/warpx_make_deck.py --config ...yaml --check    # deck on disk vs the spec
python scripts/warpx_make_deck.py --config ...yaml --verify   # warpx_used_inputs vs the spec
```

## Where to go next

| document | covers |
|---|---|
| [osiris_pipeline.md](osiris_pipeline.md) | run specs, `RunSpec` resolution, deck generation, collisions |
| [flash_analysis.md](flash_analysis.md) | FLASH config resolution, results directories, experimental streak images |
| [warpx_pipeline.md](warpx_pipeline.md) | both WarpX schemas: the hybrid runs and the heater-driven piston |
| [physics_notes.md](physics_notes.md) | OSIRIS normalized units, the FLASH `sqrt(4π)`, the PyYAML 1.1 float trap |
