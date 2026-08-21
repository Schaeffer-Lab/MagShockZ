# FLASH analysis

How a FLASH-side config says which data it analyses, where its output lands, and the
`experiment:` block that registers a measured streak image against the simulation.

The scripts themselves are listed in [user_guide.md](user_guide.md); this file is about
the machinery they share.

## Which data: resolved two ways

A FLASH-side config (`config/flash_*.yaml`) says which FLASH data it analyses in one of
two ways; `magshockz/common/flash_source.py::resolve(cfg, config_path)` normalises both to one
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

### A fan of lines of sight through one dataset

A config may state **several** rays instead of one, as a mapping keyed by label:

```yaml
lines_of_sight:
  los00: {start_point: [0.0,   0.07,  0.0], end_point: [0.0,  1.40,  0.0]}
  los30: {start_point: [0.035, 0.061, 0.0], end_point: [0.70, 1.212, 0.0]}
```

`resolve(cfg, path, los=)` returns one of them and `resolve_all` returns them all;
`--los <label>` selects on the command line, and with none given the first is used and
the available labels printed. `config/flash_3d_corrected_fan.yaml` is the worked example
— FLASH responds strongly to laser-channel preheating down the `-y` normal, so the
on-axis physics is not representative and characterising the shock needs several rays.

A **mapping, not a list**, because the per-ray shock parameters the tuners write back
(`flash:`, `flash_dump_params:`, read through `los_params`) are keyed by the same labels,
and `yaml_edit` addresses config entries by dotted key path — it has no way to name a list
element. Each ray also carries its `label` into the output directory, so one ray's
`flash_overview_*.npz` is never read back for another.

Supported by `flash_overview`, `flash_rh_prediction`, `flash_tune_shock` and
`flash_pressure_partition`.

## Where results go: one dataset tree, one sub-directory per variant config

`magshockz/common/yaml_edit.py::out_dir(base_dir, override, cfg=, config_path=)` is the single
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

## Experimental streak images: the `experiment:` config block

`scripts/flash_experiment_compare.py` compares a **streaked-shadowgraphy** image
(`experiment/<shot>/shadowgraphy_streaked/`) to the FLASH nₑ streak. A streak has the same
layout as the FLASH one — time [ns] horizontal, one spatial axis vertical.

**The image's own axes are the frame.** The experimental streak is never stretched,
resampled or mapped into simulation units; every figure is drawn in its ns and mm. The
simulation — ~15 ns of a ~69 ns record, 6.3 mm of a 10 mm slit — is *translated* onto
those axes and the image is *cropped* to the window in view. `magshockz/analysis/flash/experiment_image.py`
owns both halves and is numpy-only (matplotlib's `imread` is imported lazily), so it is
unit-tested in CI like the rest of the package:

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

## Laser ray trace and deposition: `scripts/flash_laser_audit.py`

Not shock analysis — this one asks whether FLASH's laser is depositing energy where it
should on the way *in*. Four checks, run together or individually with `--checks`:

- **`energy`** reads `<basenm>_LaserEnergyProfile.dat`. FLASH writes it every step from
  `ed_printEnergyStamp` with no runtime switch, and it holds the laser energy pumped
  into the domain and the energy rays carried back out. If the file is missing from a
  run directory it was lost in a copy, not never written — for the 3D MagShockZ runs on
  scratch it is missing, and reconstructing the out-of-domain fraction then takes a
  checkpoint and arithmetic.
- **`deposition`** sums ρ·`depo`·dV, split by material mass fraction and binned by
  distance from the beam axis and along it. **Checkpoints only**: `depo` and `lase` are
  in `unk`, but the production `plot_var` list omits them, so no plot file of the 3D runs
  carries a deposition field. `depo` is `TYPE: PER_MASS`, i.e. the specific energy
  deposited during the step the file was written in, so the sum is an energy and
  compares directly with P·dt. That comparison is only valid with
  `ed_depoReuseMaxSteps = -1`; with reuse on, `depo` is a previous step's value and only
  FLASH's own counters get rescaled.
- **`tau`** integrates FLASH's *own* κ_IB along a fan of rays across the beam footprint
  and reports the Gaussian-weighted Σw·(1 − e^−τ) up to the first cell with
  `targ > 1/2`. `flash_utils.flash_ib_opacity` / `flash_coulomb_factor` mirror
  `ed_inverseBremsstrahlungRate.F90` and `ed_CoulombFactor.F90` line for line rather
  than using plasmapy's Spitzer frequency and NRL's lnΛ — deliberately, and against the
  usual preference for `formulary`: the point is to isolate *numerics* from formula
  convention, and the two conventions differ by enough to swamp what is being measured.
- **`mesh`** reports cell size, `lrefine` and nₑ/n_c along the beam axis, with the beam
  radius in cells, over the ambient the beam crosses (`cham > 1/2`, before the target) —
  the plume in between is neither ambient nor target and its keV temperatures would
  otherwise be reported as if they described the background.

Beam geometry, pulse and wavelength come from the run's `flash.par` via
`flash_utils.parse_flash_par` / `laser_beams`, so the deck stays the source of truth
and nothing about the beam is restated in a config. `--run-dir` is a FLASH run
directory; no analysis config or line of sight is involved.

The FLASH-side counterpart — the 2D Simulation unit, the `IO_writeIntegralQuantities`
override that reports the per-material deposition live, and the convergence-scan deck
generator — lives in `flash/`; see `flash/README.md` for what the audit concluded.
