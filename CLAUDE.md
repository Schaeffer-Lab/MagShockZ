# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MagShockZ analyzes magnetized collisionless shock simulations for the Magnetized
Collisionless Shocks on Z (MagShockZ) experiment. Its core job is converting **FLASH**
MHD simulation output into initialized **OSIRIS** PIC input decks, then analyzing the
resulting OSIRIS runs (and the source FLASH data). Most work runs on NERSC Perlmutter.

## Environments

Two conda environments, used for disjoint stages — do not mix them:

- **`osiris2`** — FLASH→OSIRIS initialization and deck generation (`init_python/`).
  Has `yt` + `unyt` for reading FLASH HDF5 plot files and `jinja2` for deck templating.
- **`analysis`** — OSIRIS analysis (`scripts/`). Has `osh5` / pyVisOS (`osh5io`,
  `osh5def`, `osh5vis`) and `osiris_utils` for reading OSIRIS HDF5 output.

The installable package (`src/`, see `pyproject.toml`) depends only on numpy/scipy/unyt
so the pure-function modules import and test without the heavy OSIRIS/yt/astropy stack.

## Commands

```bash
# Tests (CI runs this; pip install numpy scipy unyt pyyaml pytest pytest-cov first)
pytest                                  # full suite (testpaths=tests)
pytest --cov=src --cov-report=term-missing
pytest tests/test_shock.py              # single file
pytest tests/test_shock.py::test_name   # single test

# Generate an OSIRIS deck from a run spec (single source of truth)
conda activate osiris2
cd init_python
python FLASH_OSIRIS_define.py --config ../runs/perlmutter_1d.run.yaml
# wrappers exist: runme_perlmutter_1d.sh, runme_perlmutter_2d.sh, run_dx_scan.sh

# Run an analysis script (config-driven; see each script's module docstring)
conda activate analysis
python scripts/overview.py --config config/perlmutter_1.3.1d.yaml [--stride 16 ...]

# Quick MP4 movie of a diagnostic (analysis env; --units electron|ion sets axis/time
# normalization read from the run dir; crop bounds are physical values in that unit)
python scripts/make_movie.py -d <run>/MS --units ion             # interactive menu
python scripts/make_movie.py -d <run>/MS/FLD/b2-savg --no-interactive \
    --units ion --xlim 80 120 --log -s 4 -o b2   # headless (sbatch: make_movie.sbatch)
```

The generator's CLI is **terminal-only with no hidden defaults**: argparse enforces
required-ness once, every parameter is explicit in the run spec, and CLI flags override
individual spec keys.

## Architecture

### Single source of truth: the run spec

Each run's parameters live **once**, in a `run.yaml` in the run's own directory.
`runs/*.run.yaml` are the version-controlled inputs; `FLASH_OSIRIS_define.py` freezes a
resolved copy to `<run_dir>/run.yaml`. Analysis reads parameters back through
`src/run_spec.py::RunSpec` instead of re-copying them into analysis configs.

`RunSpec.from_sim_dir()` resolves in priority order (first hit wins): `run.yaml` →
`run_manifest.yaml` (parse its `cli_command`) → legacy `runme*.sh` (parse python flags).
In a `run.yaml`, the `geometry` / `solver` / `diagnostics` groups exist purely for
readability and are flattened to top-level keys (matching the original CLI flags);
`charge_states` stays nested as metadata. `RunSpec` is deliberately dependency-light
(stdlib + PyYAML, astropy imported lazily) so it is unit-testable in isolation.

### Generation: `init_python/`

`FLASH_OSIRIS_define.py` (class `FLASH_OSIRIS_Base`, with 1D/2D subclasses) reads FLASH
data via `yt`, derives OSIRIS normalizations and per-species `rqm` from FLASH (mean
1836/ye over each species' mask within the OSIRIS domain), and renders the
`*_TEMPLATE.jinja` decks. `dt` is set from the CFL condition (`dx*0.95/sqrt(dims)`).
Units stay yt-native + `unyt` here (no astropy on the generation path).

### Analysis: library-first

Pure, testable functions live in **`src/`** and are re-exported from `src/__init__.py`
(`moment`, `temperature_profile`, `species_energy_profiles`, FLASH energy-partition
helpers, etc.). The thin, plotting/IO-heavy **`scripts/`** orchestrate them. Each script:
is `--config` driven (with a `$MAGSHOCKZ_SIM_DIR` override), uses
`analysis_utils.MagShockZRun` for unit/field context, `analysis_utils.detect_layout` /
`RunLayout` for dimension-agnostic (1D/2D) axis handling, reads with `osh5io`, plots with
`osh5vis` (metadata-sourced labels/units), and saves under `results/<run_name>/`.

`MagShockZRun` wraps an OSIRIS deck (via `osiris_utils.Simulation`) for field access and
astropy-unit conversions (cyclotron/ion frequencies, gyrotime). FLASH-side analysis uses
yt + unyt and does **not** go through `MagShockZRun`.

Scripts add `src/` to `sys.path` at import time (`sys.path.insert(0, .../src)`); they are
run as files, not as an installed module.

### Tests

`tests/conftest.py` puts `tests/` *ahead of* `src/` on `sys.path` so lightweight stubs
(`tests/osh5def.py`, `tests/analysis_utils.py`) shadow the real modules that would pull in
osiris/astropy. This is why the pure-function modules can be tested in CI without the
analysis env — keep new testable logic dependency-light and in `src/`.

## OSIRIS normalized units

OSIRIS normalizes to the electron plasma frequency `ω_p` and the reference density `n_0`.
Primed quantities are what live in the HDF5 files (Gaussian-based normalization):

| quantity | normalization | note |
|----------|---------------|------|
| time     | `t' = t·ω_p`                              | frequencies `ω' = ω/ω_p` |
| length   | `x' = ω_p·x / c`                          | i.e. units of `c/ω_p` (electron skin depth) |
| momentum | `u' = p/(m_sp·c) = γv/c`                  | **per-species** mass `m_sp`, so `u' ≈ v/c` for every species |
| E field  | `E' = (e·c/ω_p)/(m_e c²)·E`               | |
| B field  | `B' = (e·c/ω_p)/(m_e c²)·B`               | so `ω_ce = e·B/(m_e c) = B'·ω_p` |
| density  | `n' = n/n_0`                              | |
| energy   | per particle in `m_e c²`; densities in `n_0 m_e c²` | |

Consequences used throughout the analysis (`src/energy_partition.py`,
`src/temperature_anisotropy.py`):

- Because momentum is per-species (`u' = v/c`), bulk velocities are directly comparable
  across species and to `v_shock` (also in `c`); no per-species rescaling of velocities.
- The 2nd central moment of a phase space is `σ² = uth'² = T/(m_sp c²)`, so temperature in
  `m_e c²` is `T = |rqm|·σ²` (with `|rqm| = m_sp/m_e` for charge state 1).
- Kinetic energy densities (in `n_0 m_e c²`): ram `= ½·n·|rqm|·(⟨u'⟩−v_sh)²`, thermal
  `= ½·n·|rqm|·Σ_d σ_d²` (= `(3/2) n T_iso` isotropic). Both carry the ½ of `½mv²`.
- EM energy densities (in `n_0 m_e c²`) are `B'²/2` and `E'²/2` — the Gaussian
  `B²/(8π)` with `B² = B'²·B_0²`, `B_0 = m_e c ω_p/e`, reduces to exactly `B'²/2`.
  So field and particle energies share the same `n_0 m_e c²` units and are directly
  comparable. Fields look small only because `u_ram/u_B = v²/v_A² = M_A²` (≈100 here).

## Conventions

- FLASH analysis: yt-native + `unyt` units. No astropy, and not via the OSIRIS code path.
- 2D-capable analysis treats `x2` as the shock-normal axis; use `detect_layout` /
  `transverse_profile` rather than hardcoding dimensionality.
- Plan before implementing analysis changes; prefer adding a pure function to `src/`
  (with a test) over embedding logic in a script.
