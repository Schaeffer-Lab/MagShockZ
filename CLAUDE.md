# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MagShockZ analyzes magnetized collisionless shock simulations for the Magnetized
Collisionless Shocks on Z (MagShockZ) experiment. Its core job is converting **FLASH**
MHD simulation output into initialized **OSIRIS** PIC input decks, then analyzing the
resulting OSIRIS runs (and the source FLASH data). Most work runs on NERSC Perlmutter.

## Environments

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

The installable package (`src/`, see `pyproject.toml`) depends only on numpy/scipy/unyt
so the pure-function modules import and test without the heavy OSIRIS/yt/astropy stack.

## Commands

```bash
# Tests (CI runs this; pip install numpy scipy unyt pyyaml pytest pytest-cov first)
pytest                                  # full suite (testpaths=tests)
pytest --cov=src --cov-report=term-missing
pytest tests/test_shock.py              # single file
pytest tests/test_shock.py::test_name   # single test

# Generate an OSIRIS deck from a run spec (single source of truth). The generator is
# the external flash2osiris package; run it from the repo root so input_files/ lands here.
conda activate osiris2
python -m flash_osiris.generator --config runs/perlmutter_1d.run.yaml
# wrappers: init_python/runme_perlmutter_1d.sh, runme_perlmutter_2d.sh, run_dx_scan.sh

# Run an analysis script (config-driven; see each script's module docstring)
conda activate analysis
python scripts/overview.py --config config/perlmutter_1.3.1d.yaml [--stride 16 ...]

# Interactively tune the config's shock params, then write them back (comments
# preserved). Each command re-renders results/<run>/tune_*.png to refresh in your IDE.
python scripts/tune_shock.py --config config/perlmutter_1.3.1d.yaml          # v_shock/x_shock_0
python scripts/tune_shock.py --config ...yaml --mode regions --dump 400      # per-dump x_shock/x_downstream_start

# FLASH analog of tune_shock: place the FLASH front by hand on physical-unit (µm/ns)
# line-outs, then feed it to flash_rh_prediction.py. trajectory mode writes
# flash: v_shock_est_cms/x_shock_0_cm/t_shock_0_s — all three parameters of the straight
# front x(t) = x0 + v*(t - t0) move, including the anchor t0 ('t <ns>'), so a shock that
# forms mid-run is fitted from its formation time rather than back-extrapolated; t0
# defaults to the IC dump time when the key is absent. regions mode writes
# flash_dump_params.<idx>:
# x_shock_cm/x_downstream_start_cm (cm; separate from the OSIRIS c/ωpe dump_params).
# regions mode also shows a 2D n_e SlicePlot through the LOS, sharing the LOS-distance
# axis with the line-outs so the shock/downstream markers fall over the 2D density jump
# (--slice-axis {x,y,z}, --slice-halfwidth-um <transverse window>, --no-slice to skip).
python scripts/tune_flash_shock.py --config config/flash_3d_noshield.yaml                  # v_shock_est/x_shock_0/t_shock_0 on n_e/|B| streak
python scripts/tune_flash_shock.py --config ...yaml --mode regions --snapshot-idx -1       # per-dump x_shock_cm/x_downstream_start_cm (+ slice)

# Experimental streaked shadowgraphy vs FLASH n_e, drawn in the IMAGE's own ns/mm axes
# (never rescaled — the shorter simulation is translated onto them and the image cropped):
# side-by-side streaks, a two-colormap overlay (experiment in grey under n_e whose
# opacity ramps with density) and profiles at --times. The experiment/FLASH registration
# is HAND-TUNED (config experiment.registration; --t-offset-ns/--x-offset-mm/--flip-space
# override); --t-window/--x-window crop the view, --full-range shows the whole record.
python scripts/flash_experiment_compare.py --config config/flash_3d_2026-07.yaml
python scripts/flash_experiment_compare.py --config ...yaml --fit          # fit the shift, + an r map
python scripts/flash_experiment_compare.py --config ...yaml --t-offset-ns 2.5 --x-offset-mm -1.2 --t-window 0 25

# Quick MP4 movie of a diagnostic (analysis env; --units electron|ion sets axis/time
# normalization read from the run dir; crop bounds are physical values in that unit;
# --config uses the tuned upstream region for ion T_ci instead of the whole box)
python scripts/make_movie.py -d <run>/MS --units ion --config config/<run>.yaml  # interactive
python scripts/make_movie.py -d <run>/MS/FLD/b2-savg --no-interactive \
    --units ion --config config/<run>.yaml --xlim 80 120 --log -s 4 -o b2   # headless

# 3D volume-rendered FLASH movies (analysis env; compute node — see the .sbatch).
# --preset picks that run's tuned camera + transfer functions; it must match the
# --config's data. Sampling each AMR dump is the slow half and is cached, so
# --grids-only primes the cache and later renders are seconds each.
python scripts/flash_3d_movie.py --config config/flash_3d_2026-07.yaml \
    --preset trantham2026-07 --fields ne te ti bx by bz bmag
python scripts/flash_3d_movie.py --config ...yaml --preset ... --grids-only  # cache first
```

The generator's CLI is **terminal-only with no hidden defaults**: argparse enforces
required-ness once, every parameter is explicit in the run spec, and CLI flags override
individual spec keys.

## Architecture

### Single source of truth: the run spec

Each run's parameters live **once**, in a `run.yaml` in the run's own directory.
`runs/*.run.yaml` are the version-controlled inputs; the flash2osiris generator freezes a
resolved copy to `<run_dir>/run.yaml`. Analysis reads parameters back through
`src/run_spec.py::RunSpec` instead of re-copying them into analysis configs.

`RunSpec.from_sim_dir()` resolves in priority order (first hit wins): `run.yaml` →
`run_manifest.yaml` (parse its `cli_command`) → legacy `runme*.sh` (parse python flags).
In a `run.yaml`, the `geometry` / `solver` / `diagnostics` groups exist purely for
readability and are flattened to top-level keys (matching the original CLI flags);
`species_names` / `charge_states` stay nested as metadata. `RunSpec` is deliberately
dependency-light (stdlib + PyYAML, astropy imported lazily) so it is unit-testable.

### FLASH analysis configs: which data, resolved two ways

A FLASH-side config (`config/flash_*.yaml`) says which FLASH data it analyses in one of
two ways; `src/flash_source.py::resolve(cfg, config_path)` normalises both to one
`FlashSource` (`flash_dir`, `line_start`, `line_end`, `ic_index`, and — when knowable —
`reference_density` / `rqm_factor`), so no script branches on the mode:

- **via-run** — `sim_dir:` names an OSIRIS run; the FLASH `data_path` and the LOS
  (`start_point` / `end_point`) come from its `run.yaml` through `RunSpec`, never
  duplicated in the config. `ic_index` is the dump that seeded the deck, so FLASH and
  OSIRIS times line up. This is `config/flash_3d_noshield.yaml`.
- **direct** — `flash_data_dir:` points straight at a FLASH output directory and the
  config states its own `line_of_sight: {start_point, end_point}` (cm) plus an optional
  `ic_index` (default 0). For FLASH runs with no OSIRIS deck. This is
  `config/flash_3d_2026-07.yaml`.

`flash_data_dir` wins if both are set; `$MAGSHOCKZ_FLASH_DIR` overrides the directory in
either mode (the FLASH analogue of `$MAGSHOCKZ_SIM_DIR`, which `load_config` applies to
`sim_dir` — now optional, since a direct-mode config names no OSIRIS run). `flash_source`
resolves *paths and numbers only* — it never opens a dump, so it stays stdlib-light and
unit-tested; listing the plot files stays with `flash_utils.find_plot_files`, which owns
the filename convention. `scripts/flash_osiris_compare.py` deliberately keeps `sim_dir`:
it compares the two codes, so it genuinely needs both sides.

### Experimental streak images: the `experiment:` config block

`scripts/flash_experiment_compare.py` compares a **streaked-shadowgraphy** image
(`experiment/<shot>/shadowgraphy_streaked/`) to the FLASH nₑ streak. A streak has the same
layout as the FLASH one — time [ns] horizontal, one spatial axis vertical.

**The image's own axes are the frame.** The experimental streak is never stretched,
resampled or mapped into simulation units; every figure is drawn in its ns and mm. The
simulation — ~15 ns of a ~69 ns record, 6.3 mm of a 10 mm slit — is *translated* onto
those axes and the image is *cropped* to the window in view. `src/experiment_image.py`
owns both halves and is numpy-only (matplotlib's `imread` is imported lazily), so it is
unit-tested in CI like the rest of `src/`:

- **Load.** Prefer the **raw** streak: `experiment.csv` (a 2048² CSV of camera counts) +
  `experiment.calib` (`px_to_mm`, `px_to_ns`). The pixel grid *is* the measurement, so
  `load_streak_csv` gets the axes from the calibration alone — 2048 px × 0.033482716 =
  68.573 ns and × 0.005026907 = 10.295 mm — with `origin` placing mm = 0 (default
  `center`) and `row0_is_top` (default true) flipping the camera's top-down rows. The
  CSV is parsed once and cached beside it as `.npy`. `load_streak` (PNG + `axes:` +
  `crop_px`, with `detect_plot_box` finding the plot box inside a *decorated* figure)
  is the fallback for when only a rendered figure exists; its `axes` spans have to be
  read off the burned-in ticks by hand and must never be shrunk to "zoom".
- **Crop.** To look at part of the record use `experiment.view` / `--t-window` /
  `--x-window`, which `crop_window` turns into a whole-pixel crop that leaves every
  feature at the same ns and mm — never a rescale.
- **Register.** `t_exp = t_flash + t_offset_ns`, `mm = ±(los_µm/1000) + x_offset_mm` — a
  rigid translation of FLASH onto the image, plus a flip when the slit's +mm opposes the
  LOS. The camera trigger and the mm zero have no known FLASH counterpart, so this is
  **not derived**: either slide `--t-offset-ns` / `--x-offset-mm` / `--flip-space` by hand
  or fit it with `--fit`, which resamples FLASH onto the image's pixel pitch (no
  rescaling) and takes the whole-pixel shift with the highest normalised
  cross-correlation, evaluating every placement at once by FFT. Record the answer in
  `experiment.registration`. The script prints where FLASH landed and warns (falling
  back to the full record) when the registered data misses the image entirely.
  **Always read `flash_experiment_fit.png` before quoting a fitted offset**: for shot 3
  the r map is a broad *ridge*, flat to ±0.03 in r across the whole 53 ns of trial time
  offsets, because the FLASH leading edge (≈870 km/s over 3.8–7.3 ns) and the measured
  front (≈220 km/s) have such different slopes that no translation can make them
  coincide. The space offset is well determined; the time offset is not.

Caveat worth repeating in any write-up: shadowgraphy brightness responds to `∇²∫nₑ dl`
through the probe, not to a point sample of nₑ — front *positions* are comparable, signal
*amplitudes* are not.

### Generation: the `flash2osiris` package (external)

Deck generation lives in the standalone **`flash2osiris`** package
(`/pscratch/sd/d/dschnei/flash2osiris`, pip-installed into `osiris2`); MagShockZ keeps
only the run specs (`runs/`) and thin drivers (`init_python/`). `flash_osiris.generator`
(class `FLASH_OSIRIS_Base`, 1D/2D subclasses) reads FLASH via `yt`, derives OSIRIS
normalizations and per-population `rqm` (edens-weighted 1836/ye), and renders the deck +
py-init templates; `dt` is the CFL condition (`dx*0.95/sqrt(dims)`). Ion populations are
separated by **FLASH material** (target/chamber) via the yt plugin
(`~/.config/yt/my_plugins.py` → `flash_osiris/yt_plugin.py`), not by ion mass; the run
spec's `species_names: {cham: al, targ: si}` renames them. Units stay yt-native + `unyt`.

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

Every plotting script takes a shared `--publication` (alias `--pub`) flag: off by default
(matplotlib's own sizes, so saved figures are unchanged), on it bumps all text to large
paper/slide sizes. The look lives once in `src/plot_style.py`; a script calls
`plot_style.add_publication_arg(parser)` before `parse_args` and `plot_style.apply(args.publication)`
after. It is rcParams-only (set before any figure is drawn, so it also restyles `make_movie`'s
forked render workers) and imports matplotlib lazily, so it stays out of the CI-pure layer.

`src/plot_style.py` also owns the **display-unit** mapping (the second shared flag,
`--units electron|ion`): a script calls `plot_style.add_units_arg(parser)`, then
`disp = plot_style.build_units(args.units, cfg=cfg, config_path=...)`, and threads the
returned `DisplayUnits` into its `plot(...)`. `electron` (default) keeps native OSIRIS
`c/ωpe` & `1/ωpe`; `ion` rescales every *length* axis by the ion skin depth `d_i =
sqrt(|rqm_i|)` and shows time in the upstream ion gyroperiod `T_ci` (momentum/velocity
axes stay native — only lengths and times are rescaled). `disp.x()` / `disp.t()` rescale
coordinates, `disp.xlabel()` / `disp.tlabel()` / `disp.time_title()` give labels; only the
*figures* change (the saved `.npz` stays in native units). The `DisplayUnits` dataclass core
is numpy-only and unit-tested (`tests/test_plot_style.py`); `build_units`' ion path imports
`analysis_utils` lazily. `T_ci` needs the upstream `|B'|`: it is read from the config's
cached top-level `t_ci` key when present, else measured from the field. That `t_ci` is
written **for free** by `scripts/tune_shock.py` (trajectory mode) — it already loads the t=0
upstream field for `v_A`/`M_A`, so it computes `T_ci = ion_gyroperiod(|rqm_i|, |B'|)` there
and includes it in the `save` write-back — so every `--units ion` run reads one consistent
cached value (`make_movie.py` honours it too).

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

## FLASH magnetic field (unitsystem = none): the sqrt(4π) is REAL — do not strip it

These FLASH runs set the runtime parameter `unitsystem = "none"`, i.e. the rationalized
MHD convention where the magnetic pressure is `B_code²/2` (the 4π is absorbed into the
field variable). The **physical Gaussian field**, whose `v_A = B/sqrt(4π ρ)` reproduces
the Alfvén speed FLASH actually evolved, is therefore `B_Gauss = sqrt(4π)·B_code ≈
3.545·B_code`. yt's FLASH frontend knows this: for `unitsystem="none"` it sets
`ds.magnetic_unit = sqrt(4π) G`, so a plain `yt.load(...)` + `.to("G")` already returns
the correct physical Gauss. **Do not override `magnetic_unit`.**

History (so it is not re-introduced): a 2026-06-25 change wrongly diagnosed the sqrt(4π)
as a yt bug and overrode `magnetic_unit → 1 G` in `load_for_osiris`, which stripped the
factor and made every B-derived quantity (`v_A`, `M_A`, `β`, `T_ci`, **and the B written
into the OSIRIS deck**) wrong by sqrt(4π)/4π. It was reverted 2026-06-26 after three
independent confirmations: (1) the yt frontend applies sqrt(4π) *by design* only for
`unitsystem="none"`; (2) the dump's `unitsystem` parameter is literally `'none'`; (3) the
measured perpendicular-shock compression (dump 9: r≈3.12, *below* the gas-dynamic ceiling
3.29) matches the RH prediction only with the physical, sqrt(4π)-larger field (M_A≈8.5,
β≈6 → r=3.14), not the stripped one (M_A≈30, β≈78 → r=3.28). The correct numbers are the
original ones: `M_A ≈ 6–8.5`, `β ≈ 2–6`, upstream `|B| ≈ 15–25 T`. **Any OSIRIS deck
regenerated while the override was in place has its B too small by sqrt(4π) and must be
rebuilt.**

## Collisions (Monte Carlo, CPU-only)

Binary collisions are an **optional** feature of the deck generator, off by default
(every existing deck is byte-for-byte unchanged). They run **only in the standard CPU
solver** — OSIRIS's `cuda`/`tiles` modes do not implement Monte Carlo collisions (the
collide step lives in the standard `sort_collide` path). So a collisional run needs
`algorithm: cpu`; the generator asserts this and aborts otherwise.

Enable via a `collisions:` block in the run spec (see `runs/perlmutter_1d_collisions.run.yaml`):

```yaml
algorithm: cpu
charge_states: {al: 13, si: 14}   # also supplies each ion's q_real (Z); electrons = 1
collisions:
  enabled: true
  n_collide: 1                    # cadence: collide every n_collide steps (NOT a frequency)
  model: perez                    # perez (default) | sentoku | takizuka | isotropic
  nx_collision_cells: 1
  coulomb_log: 10.0               # "auto" for physical ln(Λ), or a fixed value to calibrate
  species: [e, al, si]            # species that collide
  like_collide: [e, al, si]       # subset that also self-collides
```

What the generator does when enabled: (1) emits `n0 = reference_density` into
`nl_simulation` (the collision module reads `sim_options%n0` and **aborts if it is ≤ 0**);
(2) adds `q_real`/`if_collide`/`if_like_collide` to each species, sourcing `q_real` from
`charge_states` (so Z lives in one place); (3) appends the `collisions` namelist. You do
**not** set a collision frequency — OSIRIS computes the physical rate from the plasma
state + `n0`; `n_collide` is only the cadence. The whole block survives the frozen
`run.yaml` round-trip (`collisions` is a `_METADATA_KEYS` group, not flattened).

**Reduced-mass caveat (important for our reduced-mass decks):** OSIRIS builds the real
ion mass as `m_real = q_real · rqm` from the **deck** `rqm`, which is reduced by
`rqm_factor`, and the collision rate scales as `(q_a q_b / μ_ab)²`. So at `rqm_factor ≠ 1`
the ion collisionality is distorted by ~`rqm_factor²`. `coulomb_log` is the single scalar
knob to recalibrate the overall rate (the generator warns when `coulomb_log: auto` is used
with `rqm_factor ≠ 1`), but it cannot independently fix different collision pairs — for a
faithful collisional run prefer `rqm_factor: 1`.

## Conventions

- FLASH analysis: yt-native + `unyt` units. No astropy, and not via the OSIRIS code path.
- Load FLASH dumps via `load_for_osiris`/`flash_utils` (it registers the OSIRIS-derived
  fields). It uses a plain `yt.load` — the `unitsystem="none"` sqrt(4π) on B is correct
  and must NOT be overridden (see the FLASH magnetic field section above).
- 2D-capable analysis treats `x2` as the shock-normal axis; use `detect_layout` /
  `transverse_profile` rather than hardcoding dimensionality.
- Plan before implementing analysis changes; prefer adding a pure function to `src/`
  (with a test) over embedding logic in a script.
