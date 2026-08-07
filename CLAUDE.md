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

The installable package (`src/`, see `pyproject.toml`) depends on numpy/scipy/unyt +
astropy/plasmapy/yt, all of which CI installs — so `src/` may use them freely, and a new
module that needs units, plasma formulary or a FLASH reader does not have to be contorted
to avoid them. The one stack CI does **not** have is OSIRIS's (`osiris_utils`, `osh5*`),
which is not pip-installable; those imports are stubbed in `tests/` (see Tests below).

## Commands

```bash
# Tests (CI runs this; pip install numpy scipy unyt astropy plasmapy yt pyyaml pytest pytest-cov first)
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
python scripts/osiris_overview.py --config config/perlmutter_1.3.1d.yaml [--stride 16 ...]

# Interactively tune the config's shock params, then write them back (comments
# preserved). Each command re-renders results/<run>/tune_*.png to refresh in your IDE.
python scripts/osiris_tune_shock.py --config config/perlmutter_1.3.1d.yaml          # v_shock/x_shock_0
python scripts/osiris_tune_shock.py --config ...yaml --mode regions --dump 400      # per-dump x_shock/x_downstream_start

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
python scripts/flash_tune_shock.py --config config/flash_3d_noshield.yaml                  # v_shock_est/x_shock_0/t_shock_0 on n_e/|B| streak
python scripts/flash_tune_shock.py --config ...yaml --mode regions --snapshot-idx -1       # per-dump x_shock_cm/x_downstream_start_cm (+ slice)

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
python scripts/osiris_make_movie.py -d <run>/MS --units ion --config config/<run>.yaml  # interactive
python scripts/osiris_make_movie.py -d <run>/MS/FLD/b2-savg --no-interactive \
    --units ion --config config/<run>.yaml --xlim 80 120 --log -s 4 -o b2   # headless

# 3D volume-rendered FLASH movies (analysis env; compute node — see the .sbatch).
# --preset picks that run's tuned camera + transfer functions; it must match the
# --config's data. Sampling each AMR dump is the slow half and is cached, so
# --grids-only primes the cache and later renders are seconds each.
python scripts/flash_3d_movie.py --config config/flash_3d_2026-07.yaml \
    --preset trantham2026-07 --fields ne te ti bx by bz bmag
python scripts/flash_3d_movie.py --config ...yaml --preset ... --grids-only  # cache first

# WarpX heater-driven piston (schema: heater_pic_2d — full PIC, no FLASH extraction).
# Measure the FLASH piston, render the ParmParse deck, run it, compare. See the
# "WarpX heater-driven piston runs" section for the constraints that shaped it.
bash init_warpx/build_warpx_gpu_2d.sh        # once: the 2D CUDA app carrying the operators
conda activate analysis
python scripts/flash_piston_profile.py --config config/flash_3d_corrected.yaml --t-window 3 12
python init_warpx/gen_heater_deck.py --config runs/magshockz_2d_heater.warpx.yaml --smoke --no-heater
sbatch init_warpx/run_heater_2d.sbatch       # 4 GPU nodes; HEATER_EXE=<cpu app> to fall back
# heater-off null control (is the ambient heating numerical?) -- own rundir, or it clobbers diags
HEATER_RUNDIR=input_files/warpx/magshockz_2d_heater_noheater \
HEATER_DECK=input_files/warpx/magshockz_2d_heater/inputs_2d_heater_noheater \
  sbatch init_warpx/run_heater_2d.sbatch
python init_warpx/gen_heater_deck.py --config ...yaml --verify   # vs the post-run echo
python scripts/warpx_heater_compare.py --config runs/magshockz_2d_heater.warpx.yaml

# Both directions of the round-trip. Neither writes anything, so both are safe against a
# running job; both exit 1 on drift.
python init_warpx/gen_heater_deck.py --config ...yaml --check    # deck on disk vs the spec
python init_warpx/gen_heater_deck.py --config ...yaml --verify   # warpx_used_inputs vs the spec
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
dependency-light (stdlib + PyYAML + astropy) so it is unit-testable without the OSIRIS stack.

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

### YAML floats: PyYAML is YAML 1.1, so exponents need a dot AND a sign

`5.0e18` and `1e-9` load as **strings**, not floats. PyYAML's 1.1 float resolver wants a
`.` in the mantissa *and* an explicit sign on the exponent — `5.0e+18` and `1.0e-09` are
floats, and `1.0e-9` happens to be fine because the negative exponent already carries its
sign. Nothing errors at load time and most consumers call `float(...)`, so a string sits
in the config until something compares or does arithmetic with it.

`yaml_edit._fmt` therefore pads the mantissa when rendering (`%g`'s `1e-09` → `1.0e-09`;
`%g` always signs the exponent), and `assert_roundtrip` compares numbers numerically
(`rel_tol=1e-5`, just above `%g`'s 6-significant-digit rounding) while still rejecting a
numeric-looking *string* — that check is what caught `tune_flash_shock`'s `t 1` writing
`t_shock_0_s: 1e-9`. `tests/test_config_yaml_scalars.py` walks every `config/*.yaml` and
`runs/*.yaml` and fails on any scalar that is a string Python can parse as a number.

### Where results go: one dataset tree, one sub-directory per variant config

`src/yaml_edit.py::out_dir(base_dir, override, cfg=, config_path=)` is the single
resolver every config-driven FLASH script calls (`flash_overview`, `flash_rh_prediction`,
`run_flash_pressure_partition`, `tune_flash_shock`, `flash_experiment_compare`). Default:
`results/<basename(flash_dir)>` — keyed on the **dataset**, so one run's outputs stay in
one tree and the expensive dataset-level caches are shared.

That breaks when **two configs analyse the same dataset differently** (a second line of
sight). They overwrite each other's figures, and because `flash_rh_prediction` reads
`flash_overview_*.npz` back out of this directory, it would silently pick up the *other*
config's line-out — a wrong-answer bug, not just clutter. So a **variant config claims its
own directory**, cheapest first:

- `results_subdir: auto` — `results/<dataset>/<config-stem>/`. Preferred: one line, no
  naming decision, renaming the config renames the directory, and it sits alongside
  `movie3d/` instead of duplicating it. A literal name works too (`results_subdir: offaxis`).
- `results_dir: <path>` — absolute or repo-relative; output decoupled from the dataset.
- `--output-dir` beats both.

The canonical config for a dataset leaves both keys unset and keeps the flat
`results/<dataset>/` path, so existing output is untouched. `flash_3d_movie.py`
deliberately builds `results/<dataset>/movie3d/` itself rather than going through
`out_dir`: its sampled AMR grids are LOS-independent, so every variant shares that cache
instead of re-sampling.

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

- **Mark features.** `experiment.trajectories` is a list of straight lines
  `x(t) = x0_mm + v_kms·(t − t0_ns)` drawn on every panel, hand-fitted in the units the
  axes use (km/s, mm, ns). `frame: experiment` (default) is a feature measured *in the
  data* — the observed shock front, the piston plasma — and so does not move when the
  registration changes; `frame: flash` is a simulated feature, translated onto the image
  like the FLASH data. The `flash:` block's shock front is added automatically as a
  `flash`-frame line, which is how the ~300 km/s measured front and the ~1000 km/s
  simulated one get compared on one picture.

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

### WarpX generation: the `flash2warpx` package (external)

The same FLASH dumps also feed **WarpX hybrid-PIC** (Ohm's-law) runs via the standalone
**`flash2warpx`** package (`/pscratch/sd/d/dschnei/flash2warpx`, pip-installed / on
`PYTHONPATH`). `flash_warpx.extractor.extract_slice` slices a FLASH dump into SI `.npy`
arrays + `meta.yaml`; `flash_warpx.run.HybridRunConfig`/`build_and_run` step WarpX over
them. MagShockZ keeps only the MagShockZ-specific run assets, mirroring the OSIRIS split:
- **configs** (the `extract:` + `run:` blocks fed to flash2warpx) → `runs/*.warpx.yaml`
  (distinct suffix from the OSIRIS `*.run.yaml`; **not** a `RunSpec` schema).
- **thin drivers** → `init_warpx/` (`stage_prod.py` = PHASE 1 extract+deflate in the
  `analysis` env; `run_prod.py` + `run_{production,deeptime}.sbatch` = PHASE 2 WarpX run).
- **generated trees + diags + movies** → `input_files/warpx/…` (gitignored, regenerable).

The generic package (extractor/run/`viz/` diagnostics, the `examples/magshockz_2d.yaml`
usage example, docs) stays in flash2warpx. Check a slice resolves the ion skin depth with
`python -m flash_warpx.resolution input_files/warpx/<tree> [--dx <sim_dx_m>]` (pure numpy;
`analysis` env). The MagShockZ-side Spitzer resistivity tool (`scripts/warpx_spitzer_resistivity.py`,
`src/spitzer_resistivity.py`) reads these trees to pick `run.plasma_resistivity`.

### WarpX heater-driven piston runs: `schema: heater_pic_2d` (no FLASH extraction)

`runs/*.warpx.yaml` holds **two different schemas**, dispatched on a top-level `schema:`
key. The flash2warpx specs above (`magshockz_2d_{production,deeptime}`) have `extract:` +
`run:` blocks and read a FLASH slice off disk. `runs/magshockz_2d_heater.warpx.yaml`
(`schema: heater_pic_2d`) is a different animal:

- **Full electromagnetic PIC**, not the Ohm's-law hybrid — required, because the heating
  operator deposits energy into *electrons*, and the hybrid solver has none.
- **No FLASH data is read.** The piston is *grown* by WarpX's `ParticleHeater` +
  `TargetInjector` (the PSC / Fox et al. 2018 laser-ablation surrogate, on the lab fork's
  `feature/particle-heater` branch at `/pscratch/sd/d/dschnei/warpx`): the injector keeps a
  dense cold slab topped up towards `n_t`, the heater drives momentum-space diffusion
  `d<u_i²>/dt = H` in its electrons, and the slab expands as a self-consistent, kinetically
  smooth piston. This sidesteps rather than solves the grid-sharp ideal-MHD interface in
  `docs/piston_interface_smoothing_plan.md` — both approaches stay useful.
- **Reduced mass ratio + arbitrary reference density**, so FLASH is matched in
  *dimensionless* terms only. `src/heater_piston_scaling.py` owns that mapping and is
  explicit about both halves: preserved are `M_A`, `M_ms`, `beta_e/beta_i`,
  `n_piston/n_amb`, `r_spot/d_i`, `omega_ci·t`; deliberately broken are `m_i/m_e`
  (~5e4 → 100), `Z` (→ 1), `omega_pe/omega_ci`, and absolute ns/µm. The generator prints
  the FLASH-vs-deck table; read it. `ReducedScaling.to_ns` / `to_um` bridge back through
  the matched *ion* scales (`d_i`, `T_ci`), never through `1/omega_pe`.

**Where the code lives.** `init_warpx/` holds only CLIs and sbatch scripts — no logic, and
nothing else imports from it. The three library modules are flat in `src/`, so they are
inside `testpaths` and `--cov=src`:

| module | role | omegashock analogue |
|---|---|---|
| `src/heater_piston_scaling.py` | FLASH targets → deck constants; the invariants | `units.py` |
| `src/heater_deck.py` | render the deck; parse/resolve/`key_params`/`verify` it back | `deck.py` |
| `src/heater_spec.py` | `load` (raises) / `scaling` / `validate` (warns) / `freeze` | `config.py` |

The spec block names are `meta:` / `flash_target:` / `scaling:` / `flow:` / `operators:` /
`numerics:` / `diagnostics:` / `runtime:` / `targets:` / `smoke:`. `flash_target:` and
`scaling:` stay distinct on purpose: this pipeline has **two tiers of primaries** — what
FLASH *measured* and what the reduced-mass bridge *chose* — and merging them would erase
the most important distinction in the file.

**The spec → deck mapping is one-way, and the deck round-trips back to prove it.**
`parse_inputs` → `resolve_constants` → `key_params` → `verify` resolves a deck's
`my_constants` in a restricted `eval` and diffs the resulting numbers against a freshly
rendered one, so the comparison is independent of formatting, comments, and whether a
length was written as `20.*de` or `2.*di`. Two uses: the generator self-verifies whatever
it writes, and `--verify` closes the loop against the `warpx_used_inputs` WarpX echoes
after the run. `rtol` is `1e-6`, not exact, because the WarpX build ships CODATA-2022
constants against `heater_piston_scaling`'s CODATA-2018 — a ~1e-9 shift in every resolved
length. `verify` ignores `amr.restart` and names `HEATER_EXTRA_ARGS` when `max_step`
diverges, since the sbatch appends both.

**`my_constants` are symbolic, so the deck states its own physics.** `slab = 2.0*di`,
`B0 = vA*sqrt(mu0*namb*mi)`, `u_std = sqrt(theta_e_amb)`, and the ambient temperatures as
the pressure balance they came from (`theta_e_amb = beta_e*B0^2/(2*mu0*namb*m_e*clight^2)`)
— which puts **four of the seven matched invariants** in the deck as named constants,
checkable without running Python. The floats that remain carry **10 significant digits**,
not the more readable 6: the symbolic expressions chain off them, and `theta_e_amb` reaches
`vA` through `B0` *squared*, so 7 digits would land it ~2e-7 from what `derive()` computed.

**`load` raises; `validate` warns.** Structural problems the generator cannot render —
wrong `schema`, missing block, backwards `t_window_ns`, a 1D `ppc_each_dim`, a
**non-periodic boundary** — raise in `load`. Everything physical warns and renders anyway,
because a deliberately off-target deck (the null control, a resolution probe, a
frame-consistency run) is legitimate. `validate` covers the derived invariants against
`targets.invariants:` (a regression guard on `derive` itself, since they are matched by
construction — hence `rtol=1e-6`), Debye resolution, domain wrap, spot-image overlap,
blocking factor and box-vs-rank count, `dt*omega_pe`, **the diagnostic cadences** (see the
"cadences are in STEPS" bullet below), the drive window, and — given `out_dir` — stale
checkpoints, which `run_heater_2d.sbatch` otherwise only discovers after six hours of queue.

Pipeline (`analysis` env for 1/2/4/5, the WarpX GPU app for 3):

```bash
# 1. measure the FLASH piston -> paste the printed flash_target: block into the spec
python scripts/flash_piston_profile.py --config config/flash_3d_corrected.yaml --t-window 3 12
# 2. render the deck (+ smoke/null variants, a frozen run.yaml, run_env.sh, a report).
#    Self-verifies; prints validate() warnings to stderr and the exact sbatch line.
python init_warpx/gen_heater_deck.py --config runs/magshockz_2d_heater.warpx.yaml --smoke
# 3. run it
sbatch init_warpx/run_heater_2d.sbatch
# 4. close the loop against what WarpX actually ran
python init_warpx/gen_heater_deck.py --config runs/magshockz_2d_heater.warpx.yaml --verify
# 5. compare to FLASH and iterate theta_e_heater / contrast / slab width
python scripts/warpx_heater_compare.py --config runs/magshockz_2d_heater.warpx.yaml
```

Constraints that shaped the deck, all of them things that bite silently:

- **Text deck, not PICMI.** The two operators are ParmParse-only; no PICMI binding exists
  and no pre-built `pywarpx` contains them. The sbatch greps the binary for
  `particle_heater` and aborts if absent, because an older binary silently ignores the
  block and produces a physics-free run that looks fine.
- **GPU needs its own build, and needed a source fix.** Every pre-existing GPU build in
  the fork predates the operator commits. `init_warpx/build_warpx_gpu_2d.sh` builds the
  2D CUDA app (`-DWarpX_APP=ON -DWarpX_PYTHON=OFF`, ~20 min) and refuses to finish unless
  the binary carries the symbol. The first CUDA build **failed to compile**
  `ParticleHeater.cpp`: its kernel is an extended `__device__` lambda inside
  `applyHeaterToSpecies`, and NVCC forbids those in a function with private/protected
  class access. Fixed by making that one method public (with a comment saying why) in the
  fork — it compiles fine as private under CPU/OMP, which is why the operator's whole
  validation campaign never hit it. `TargetInjector` is unaffected: its `ParallelFor`
  lambda sits in a free function. The sbatch runs GPU by default (`-C gpu -N 2`, one rank
  per GPU, `MPICH_GPU_SUPPORT_ENABLED=1`) and falls back to the prebuilt CPU app via
  `HEATER_EXE=…`, keying the MPICH/OpenMP env off which binary it was given.
  - Both scripts verify the symbol with `strings … | grep -c … || true`, **not**
    `grep -q`. Under `set -o pipefail` a short-circuiting `grep -q` SIGPIPEs `strings`
    partway through the 594 MB CUDA binary and pipefail reports the pipeline as failed, so
    a correct build gets rejected as "predates the operator". The 15 MB CPU binary is
    small enough that `strings` finishes first, which is why the pattern looked fine until
    the GPU build existed.
- **Fully periodic, so the slab is symmetric and expands both ways.** A uniform applied
  E/B on the grid requires periodic boundaries (*"do not use any other boundary condition
  than periodic"*, WarpX `parameters.rst`), so there is no one-sided piston off a wall:
  two fronts propagate in ±z and the run must stop before either wraps. The generator
  sizes the domain from `v_piston·t_run` and warns when it would not. This is not a knob:
  `heater_spec.load` **raises** if the spec grows a `geometry.boundary` that is anything
  else, because the symmetric slab, the domain sizing and the run-length budget all follow
  from it. A one-sided/wall variant needs a new `schema:` value and its own renderer —
  which is what the `schema:` dispatch exists for.
- **The heater currently drives for the whole run, and FLASH's laser does not.**
  `operators.drive_stop_t_ci` gates both operators in the run's own normalized time
  (`T_ci`, so it survives changes to `cell_size_de` and `cfl`), rendered as a stock
  `IntervalsParser` string `0:<stop>:<period>` — no engine change. The injector stops
  *with* the heater: otherwise it keeps refilling the slab with cold material and the
  contrast holds while the drive dies, which is neither FLASH nor a clean ballistic coast.
  The whole run is only `0.127 T_ci`, so the value is ~0.03, not omegashock's ~0.87.
  Currently `null`, and `validate` warns about it.
- **Provenance is per queue-segment, not per run.** `run_heater_2d.sbatch` stamps
  `provenance_<jobid>.txt` (binary + sha256, both git SHAs and dirty counts, deck and spec
  sha256, grid, restart target, GPUs) *before* `srun`, and appends the exit code after.
  One `provenance.txt` would be wrong here: the run **chains** across queue slots, so the
  later segment would overwrite the record of the binary that produced the earlier half of
  the data. The sbatch stays hand-written — its `--gpu-bind=none` + `3-SLURM_LOCALID`
  device ordering, the `strings | grep -c` SIGPIPE guard and the checkpoint grid-match loop
  are dearly bought, and generating them per run directory would fork that logic and leave
  stale copies. What the spec *does* drive is `run_env.sh` (`export HEATER_EXE=${HEATER_EXE:-…}`,
  so the CPU-app fallback on the command line still wins).
- **`TargetInjector` has no drift knob** (`u_std` only). With `flow.v_flow_kms ≠ 0` the
  initial plasma is boosted correctly *and* the motional `Ez = −v_flow·B0` is imposed —
  an exact equilibrium, but therefore a pure Galilean boost, useful only as a
  frame-consistency test. Injected piston particles then enter at rest in that frame and
  gyrate. Keep it 0 for physics runs.
- **Upstream = the initially unperturbed background** (`--upstream initial`, the default):
  `flash.par`'s chamber IC read off the t=0 dump, with no laser channel to exclude because
  the pulse has not fired yet (it ramps from 0.1 ns). The dump is read rather than the
  `flash.par` numbers trusted, because `flash.par` gives `ms_chamZ = 13`, the *atomic*
  number, while the EOS returns **Zbar = 3.66** at 9.83 eV — and Zbar sets `n_e`, `beta_e`
  and `d_i`. The config's `unperturbed_background:` block exists only to cross-check the
  dump and warn on a mismatch. `--upstream measured` samples ahead of the front instead;
  the script always prints both, because they differ enormously (`T_e` 30×, Zbar 3.5×,
  `|B|` 0.5×) and the choice changes every dimensionless number:

  | | unperturbed IC | measured ahead of front |
  |---|---|---|
  | `n_e` [m⁻³] | 3.04e24 | 1.82e24 |
  | `|B|` [T] | 7.00 | 3.44 |
  | `T_e` [eV] | 9.83 | 296 |
  | `M_A` / `M_ms` | 23.8 / 21.9 | 20.0 / 6.2 |
  | `beta_e` / `beta_i` | 0.245 / 0.067 | 18.3 / 0.52 |

  The IC is the low-β, strongly magnetized background the experiment was set up with; the
  measured column is a conduction/radiation-preheated **precursor**.
- **`diag_type = Full` writes raw particles, and at 1.6e8 macroparticles that dominates
  everything.** Measured: **4.6 GB per plotfile, of which 4.4 GB is particle data** and
  only 0.27 GB is the grid fields the analysis reads — ~0.3 TB over 71 dumps, and the I/O
  throttled the run harder than the physics did (step rate fell ~3× once dumping began).
  The generator therefore sets `diag1.write_species = 0` and relies on the per-species
  `rho_<sp>` / `T_<sp>` / `usq_<sp>` **grid** fields, which carry every profile the
  comparison scripts need. Raw particles survive as a separate sparse `phase` diagnostic
  (`phase_space_interval`, `phase_space_fraction: 0.02`) so the `(z, u_z)` reflected-ion
  signature is still available at ~1% of the cost.
- **Checkpoints are a diagnostic, so they land in `diags/chk*`, and their completion marker
  is `WarpXHeader`** — not the bare `Header` an AMReX plotfile carries. The sbatch's
  auto-resume originally globbed `$RUN/chk*` and required `Header`, so it found nothing and
  reported `FRESH start` even with seven valid checkpoints on disk; the failure mode is a
  silent restart from step 0, not an error. Both are fixed, but check the
  `=== RESUMING from checkpoint … ===` line rather than assuming.
- **`kappa` was measured, not assumed, and it was wrong by 2.5x.** `v_piston = kappa*c_s`
  had `KAPPA_DEFAULT = 2.5` (the classic rarefaction-front value, plus reading the 1D run's
  *shock* speed as the piston contact speed). The first 2D run aimed for 0.050 c and
  delivered **0.0202 c**, so kappa is **1.01** and every quantity derived through the front
  speed — `B0`, both ambient temperatures, `M_A` — was off by 2.5x. `M_A` came out 9.6
  against a 23.8 target purely from this.
- **Correcting kappa cannot be absorbed at fixed `mass_ratio` without paying for it, and the
  reason is structural.** `theta_heater` and `theta_ambient` are BOTH proportional to
  `mass_ratio * v_piston_c^2`, so their ratio is pinned at `2*M_A^2/(beta_e*kappa^2)` by the
  matched invariants alone (4531 here). Holding `v_piston_c = 0.05` at the true kappa gives
  `theta_e = 0.245`, i.e. piston electrons at **v_th = 0.5 c**, where the heater's
  non-relativistic `u = gamma*v` kicks stop imposing the `theta_e` they are handed. Cooling
  them by lowering `v_piston_c` cools the ambient by the same factor, so `lambda_De` shrinks
  and `cell_size_de` must shrink with it: cost goes as `v_piston_c^-3` (cells `^-2` in 2D,
  steps `^-1`). Lowering `mass_ratio` instead is nearly FREE — `cells/d_i =
  sqrt(mass_ratio)/(dx/d_e)` and the Debye-limited `dx` also goes as `sqrt(mass_ratio)`, so
  it cancels — which makes it the only knob that buys sub-relativistic electrons without
  buying grid. The spec keeps `mass_ratio: 100` by choice and pays instead:
  `v_piston_c: 0.03`, `cell_size_de: 0.12`, 944x4720 cells, 99326 steps, ~7.7x the first
  run (~2.2 h on 4 GPU nodes). `derive()` now guards on `sqrt(theta_e_heater)`, not
  `v_piston_c` — they differ by `sqrt(mass_ratio)/kappa`, a factor of 10, which is why the
  old guard stayed silent at 0.5 c.
- **Diagnostic cadences are in STEPS, so they must be rescaled whenever the step count
  changes.** Going 35760 -> 99326 steps at a larger grid would have made `diag_interval: 500`
  write ~199 plotfiles of ~750 MB (~150 GB) and throttle the run the way raw particles did;
  the spec now uses 1400 / 11000 / 14000 to hold ~71 plotfiles, ~9 phase dumps and
  ~7 checkpoints.
- **A checkpoint is only resumable by a deck with the same grid, and nothing enforced that.**
  Changing `v_piston_c` or `cell_size_de` resizes the grid, and the resume glob would happily
  hand a 944x4720 deck a 568x2832 checkpoint. Every checkpoint records its `amr.n_cell` in
  `warpx_job_info`, so the sbatch now requires it to match the deck and logs each
  `SKIPPING checkpoint …: grid [a b] != deck [c d]`. Archive superseded output
  (`diags` -> `diags_v1_kappa2.5`) rather than leaving it: plotfile names collide too.
- **The ambient-heating question needs a null control, not an argument.** Shock preheating
  and numerical grid heating are indistinguishable in every diagnostic, and the first run's
  103x rise in ambient `<u^2>` happened while the piston had swept only ~20% of the box.
  `gen_heater_deck.py --no-heater` writes a deck differing from production ONLY in the
  heater block (injector still on, so macroparticle count and load match); run it with
  `HEATER_RUNDIR=…_noheater HEATER_DECK=…_noheater`. With no energy input the slab cannot
  expand, so any `<u^2>` rise there is numerical. This is why `cell_size_de: 0.12` targets
  `lambda_De/dx = 0.0368` — the first run's exact value — rather than the cheaper 0.147 that
  merely clears the 0.03 threshold: matching it is what makes the control controlled.
- **A cold upstream can under-resolve the Debye length.** `beta_e` = 0.245 gives an
  ambient `T_e` of only 27.7 eV in the deck, so `lambda_De/dx` fell to 0.015 at
  `cell_size_de: 0.5` — below the ~0.03 threshold the OSIRIS convergence scan puts
  numerical grid heating at. `derive()` computes `debye_per_cell`, warns, and names the dx
  that would fix it; the spec now runs `cell_size_de: 0.12` at `v_piston_c: 0.03` (0.0368,
  clear — see the kappa bullets above for why both moved together). Grid heating of
  a cold ambient looks exactly like shock heating in the diagnostics, so
  `warpx_heater_compare.py` plots the ambient `<u²>` history alongside the operator
  balance.
- **The piston comparison is target species against target species, never a summed `n_e`.**
  FLASH side: its own `targ` (Si) mass fraction masking the electron density. WarpX side:
  `rho_piston_ions`, which at the deck's `Z = 1` is that species' electron density too. So
  `flash_piston_profile`'s front / drive density / contrast, `warpx_flash_evolution`'s
  piston row and *both* 2D slice rows, and `warpx_heater_compare`'s scorecard all measure
  one species. Two things this rules out: summing species (which hides *which* material
  moved — the ambient gets its own row instead), and reaching the target ion density
  through `rho*X/(A m_u)` and multiplying by the EOS `Zbar`, which folds an ionization
  state that runs 3.7 in the ambient to ~11 in the piston into a comparison whose deck has
  `Z = 1` by construction. Switching off the `Zbar` route moved the measured numbers < 0.5%
  (`n_piston_drive_per_m3` 2.0757e+25 → 2.0815e+25, front 768.6 → 768.5 km/s) because the
  dense piston is nearly pure Si; it is the mixed cells and the tail that differ, hence
  `l_piston_m` +9%. `flash_utils.flash_slice` takes `mask_field=` for the 2D half.
- **The FLASH measurement is easy to get wrong**, so `scripts/flash_piston_profile.py`
  reports per-dump diagnostics and refuses bad windows:
  - the front needs an **absolute** density threshold (a multiple of the far-field
    ambient), not a fraction of the profile peak. The dense stagnated material next to
    the target overtakes the leading edge in amplitude partway through the run, and a
    peak-relative threshold then jumps backwards — giving a front that "moves" at
    −390 km/s while every individual position increases.
  - `n_piston_drive_per_m3` is the density just **behind the front** (what drives the
    shock), not that global peak, which is ~17× higher and never reaches the front.
  - each dump is checked for a **pristine upstream**; once the diamagnetic cavity
    swallows the whole line-out there is nothing left to measure, and averaging past that
    point returns the cavity interior (`beta_e` ≈ 160, sub-critical `M_ms`) instead of the
    upstream. This is why `config/flash_3d_corrected.yaml` runs its LOS out to
    **y = 1.40 cm** rather than the 0.70 cm the other configs use.
  - the ambient band sits **1500 µm ahead** of the front: closer in it is still inside the
    compressed, conduction-preheated shell (at 400 µm: `n_e` 1.7×, `|B|` 1.9×, `T_e` 3.5×
    the converged values). Re-run the offset scan on any new dataset.
  - `l_piston_m` is **provenance only, not a matched invariant** — the ideal-MHD piston
    edge is grid-sharp, so no fitted e-folding length describes the interface.

Current numbers for `FLASH_MagShockZ3D-corrected`: piston front **769 km/s** (fit rms
114 µm) over 3–12 ns, against the unperturbed background above → `M_A` = 23.8,
`M_ms` = 21.9 (strongly super-critical), `beta_e` = 0.245, `beta_i` = 0.067, contrast 6.8,
`d_i` = 355 µm, `T_ci` = 69.0 ns (so the 8.75 ns window is only 0.127 `T_ci`).

The FIRST deck to target these — 568×2832 at 0.2 `d_e`, 35760 steps, 1.6e8 macroparticles —
ran clean (`exit=0`, no non-finite) but at `kappa = 2.5` and so **missed**: front
60.4 vs 149.5 `d_i/T_ci`, `M_A` 9.6 vs 23.8, contrast decaying 6.86 → 2.82, and ambient
`<u²>` up 103×. Its output is archived at `diags_v1_kappa2.5/`. The current deck corrects
kappa to its measured 1.01 and pays for sub-relativistic electrons in grid:
944×4720 at 0.12 `d_e`, 99326 steps, 4.5e8 macroparticles, ~2.2 h on 4 GPU nodes.

### Analysis: library-first

Pure, testable functions live in **`src/`** and are re-exported from `src/__init__.py`
(`moment`, `temperature_profile`, `species_energy_profiles`, FLASH energy-partition
helpers, etc.). The thin, plotting/IO-heavy **`scripts/`** orchestrate them. Each script:
is `--config` driven (with a `$MAGSHOCKZ_SIM_DIR` override), uses
`analysis_utils.MagShockZRun` for unit/field context, `analysis_utils.detect_layout` /
`RunLayout` for dimension-agnostic (1D/2D) axis handling, reads with `osh5io`, plots with
`osh5vis` (metadata-sourced labels/units), and saves under `results/<run_name>/`.

The same rule governs the WarpX heater generator: `init_warpx/` holds CLIs and sbatch
scripts only, and **nothing imports from it**. A script that needed the run-spec loader
used to reach backwards into `init_warpx/gen_heater_deck.py`; it now goes through
`src/heater_spec.py` like any other library module.

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
written **for free** by `scripts/osiris_tune_shock.py` (trajectory mode) — it already loads the t=0
upstream field for `v_A`/`M_A`, so it computes `T_ci = ion_gyroperiod(|rqm_i|, |B'|)` there
and includes it in the `save` write-back — so every `--units ion` run reads one consistent
cached value (`make_movie.py` honours it too).

### Tests

CI installs numpy/scipy/unyt/astropy/plasmapy/yt/PyYAML, so tests may use any of them.
The only thing it cannot install is **OSIRIS's own stack** (`osiris_utils`, `osh5io`,
`osh5def`, `osh5vis`), so `tests/conftest.py` puts `tests/` *ahead of* `src/` on
`sys.path` and lightweight stubs (`tests/osh5def.py`, `tests/analysis_utils.py`) shadow
the real modules that would import it. Keep new testable logic in `src/` and free of
OSIRIS imports; astropy, plasmapy and yt are fine.

plasmapy (~2.5 s) and yt are still imported **lazily** in a few `src/` modules — that is
now purely an import-time cost decision for scripts, not a CI constraint, so don't add a
new lazy import just to keep a module "pure".

One thing yt-in-CI does *not* buy: `src/flash_utils.py` is still un-importable there,
because it calls `yt.enable_plugins()` at module scope and yt **raises**
`FileNotFoundError` when there is no `~/.config/yt/my_plugins.py` (the flash2osiris
plugin, symlinked in on Perlmutter). Testing `flash_utils` in CI needs that call guarded
first — installing yt is necessary but not sufficient.

`tests/test_heater_deck.py` asserts on `key_params` — *what WarpX will do* — rather than on
the deck text wherever the claim is about behaviour, so rewording a comment is not a
failure but moving a constant is. Two tests are worth knowing about: the null control's
rendered deck must differ from production in **only** `heater.*` keys (that is what makes
it a control), and `test_it_still_means_what_the_running_jobs_deck_means` re-renders the
checked-in spec and requires `n_cell` / `prob_lo` / `prob_hi` / `max_step` to be **exactly**
equal to the deck in `input_files/` — a checkpoint is resumable only at the same grid *and*
the same domain, and the sbatch's guard only checks the former.

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

## Code style: the code explains itself, comments explain the physics

**Write self-documenting code, not commented code.** A comment that restates what the
next line does is noise that rots; the same information belongs in the name, the
signature, or the structure, where it cannot go stale. Default to *fewer* comments and
*more* legible code.

- **Units live in the quantity, not in the variable name.** Do **not** write `ne_cm3`,
  `b_gauss`, `v_kms`. Call it `ne`, `b`, `v` — and carry a real unit-bearing object
  (yt/`unyt` array on the FLASH side, plasmapy/astropy `Quantity` for derived plasma
  parameters) so the units are checked by the library instead of asserted by a suffix a
  reader has to trust. When a specific unit matters, **convert explicitly at the point of
  use**: `ne.to("cm**-3")`, `b.to("G")`, `v.to("km/s")`. That call is simultaneously the
  conversion, the assertion, and the documentation — it raises on a wrong-dimension
  input, which a name never does.
  - Prefer **plasmapy's `formulary`** over hand-rolled plasma formulas (`gyrofrequency`,
    `inertial_length`, `Alfven_speed`, `Debye_length`, `plasma_frequency`, `Mag_Reynolds`,
    …) and its `particles`/`constants` over literal constants. It carries the units and
    the definitions, so there is nothing left to get wrong or to comment. Both `analysis`
    and `osiris2` already have `plasmapy` — use it rather than adding a private
    `1836.0`-style constant. (This does not license astropy on the FLASH path: keep FLASH
    data in yt/`unyt` per the Conventions below, and convert at the boundary if a
    plasmapy call needs a `Quantity`.)
  - Bare floats are the exception, and they need the unit spelled out **in the signature
    or docstring**, not smuggled into arithmetic: at the config/YAML boundary (where keys
    like `x_shock_cm`, `v_shock_est_cms`, `t_shock_0_s` legitimately name their unit
    because YAML has no unit type — keep those key names), in the OSIRIS normalized layer
    (primed quantities are dimensionless by construction — see the units table above), and
    in numerical kernels where attaching units per-element would be a real cost. Strip
    units at that boundary deliberately (`.to("cm").value`) rather than letting a plain
    array of unknown unit propagate.
- **Names describe the physical thing, not its formatting.** `upstream_density` over `n`;
  `shock_position`/`x_shock` over `xs`. Physics symbols (`rqm`, `M_A`, `beta`, `T_ci`,
  `d_i`, `ne`, `ti`) are good names where they are the field's own vocabulary and already
  documented above — do not inflate them into prose. A bare `i`/`j`/`k` is fine for a loop
  index and `x`/`y` for plain array axes. The test is whether a reader knows *what it is*;
  the library answers *what unit it's in*.
- **Type-hint every function in `src/`** — arguments and return. Prefer precise types
  (`np.ndarray`, `Path`, `Sequence[float]`, `Literal["electron", "ion"]`,
  `dict[str, float] | None`) over bare `Any`/untyped. Scripts should be hinted too where
  it costs nothing; the pure-function layer in `src/` is where it is required.
- **`@dataclass` for anything with more than ~three related fields** and for every
  multi-value return — the pattern `RunSpec` / `FlashSource` / `DisplayUnits` /
  `RunLayout` already establish. A named field beats a tuple index; returning a
  `(a, b, c, d)` tuple that the caller unpacks positionally is a bug waiting to happen.
  Use `frozen=True` for value objects that shouldn't mutate after resolution.
- **Small pure functions with one job, named for what they return.** If you need a
  comment shaped like `# now do X`, that block is a function called `x(...)`. Keep the
  numerical core pure (arrays in, arrays out, no IO, no globals, no plotting) so it is
  unit-testable in the CI-pure `src/` layer — this is the same rule as "library-first"
  above, stated as a style rule.
- **Structure carries meaning.** Guard clauses and early returns instead of nested `if`
  pyramids; extract complex boolean conditions into a named local
  (`is_downstream = x > x_shock`); one blank-line-separated paragraph per logical step
  inside a longer function; unpack config dicts into named locals near the top rather
  than indexing `cfg["a"]["b"]` deep in the arithmetic. Avoid magic numbers — reach for
  plasmapy/`unyt` constants first, and where the literal is genuinely ours, a named
  module-level constant (`SQRT_4PI`) says what it means without a comment.
- **Docstrings, not inline narration.** Every public function in `src/` gets a short
  docstring stating what it returns (plus a `Parameters`/`Returns` block when the
  signature isn't self-evident). Say the units only where the code can't — for bare-float
  arguments and returns, or to name the normalization (`c/ωpe`, `n_0 m_e c²`); for
  unit-bearing quantities the object already answers that. Every script keeps its module
  docstring describing its CLI usage. That is where explanation goes.
- **Comments earn their place** by explaining what the code *cannot* say: the physics or
  convention behind a formula (`B_Gauss = sqrt(4π)·B_code` for `unitsystem="none"`), a
  non-obvious "why this and not the obvious thing", a reference to a paper/equation, a
  cited gotcha (PyYAML 1.1 exponents), or a `TODO`/known limitation. Prefer one comment
  above a block over a comment per line, and never leave a comment that merely paraphrases
  the code, narrates a change ("changed from X to Y"), or documents history — git does that.
- **Match the surrounding file.** These rules describe where the codebase is heading; when
  editing an existing module, follow its established naming and density rather than
  restyling it mid-task.

## Conventions

- FLASH analysis: yt-native + `unyt` units, and not via the OSIRIS code path. No astropy
  *as the working unit system* — FLASH arrays stay `unyt`. Calling plasmapy `formulary` for
  a derived plasma parameter is fine (it is the preferred alternative to hand-rolling the
  formula, see Code style above); convert at that call boundary and bring the result back
  to `unyt` rather than letting astropy `Quantity`s spread through FLASH code.
- Load FLASH dumps via `load_for_osiris`/`flash_utils` (it registers the OSIRIS-derived
  fields). It uses a plain `yt.load` — the `unitsystem="none"` sqrt(4π) on B is correct
  and must NOT be overridden (see the FLASH magnetic field section above).
- 2D-capable analysis treats `x2` as the shock-normal axis; use `detect_layout` /
  `transverse_profile` rather than hardcoding dimensionality.
- Plan before implementing analysis changes; prefer adding a pure function to `src/`
  (with a test) over embedding logic in a script.
