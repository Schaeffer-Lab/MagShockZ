# MagShockZ reorganization plan

Three simulation codes (FLASH, OSIRIS, WarpX) now share a repository whose directory
names record its history rather than its structure. This plan makes the library
importable, gives each stage and code one obvious home, deletes what is dead, and moves
the physics prose out of docstrings and `CLAUDE.md` into documentation written for a
human reader.

Every decision below was chosen deliberately; the rationale is recorded so it is not
relitigated. One item (`analysis_utils`) is still open and marked as such.

## Target layout

```
magshockz/                    installable package (replaces src/)
  common/                     used by two or more codes
  init/
    osiris/                   deck validation; generation itself is external (flash2osiris)
    warpx/                    units, config, deck, calibration
  analysis/
    flash/                    experiment images, energy partition, shock finding
    osiris/                   moments, shock state, streaks, reflected ions
    warpx/                    resistivity, metrics, plotting, FLASH bridge
scripts/                      flat, code-prefixed CLI wrappers
runs/                         run specs (inputs that define a run)
config/                       analysis configs (how to analyse a run)
tests/                        flat
docs/                         user guide, physics notes, per-pipeline notes
notebooks/                    renamed from analysis_notebooks/
init_warpx/                   sbatch + build scripts (frozen this round; see Phase 4)
input_files/                  generated run trees (gitignored, never moved)
```

Stage is the top-level split, code the second. Imports read
`from magshockz.analysis.flash import piston_profile`.

## Module map

`common/` follows the literal rule: **any module imported by two or more codes**. This
deliberately places OSIRIS physics such as `rankine_hugoniot` in `common/`, because
`flash_osiris_compare.py` genuinely depends on it from the FLASH side.

| destination | modules |
|---|---|
| `common/` | `run_spec` `yaml_edit` `plot_style` `flash_utils` `flash_source` `moments` `dimensionless_params` `energy_partition` `perpendicular_shock` `rankine_hugoniot` `temperature_anisotropy` `piston_profile` `heater_piston_scaling` |
| `init/warpx/` | `units` `config` `deck` `calibration` (from `src/warpx/`) |
| `init/osiris/` | validation logic lifted out of `scripts/validate_init.py` |
| `analysis/flash/` | `experiment_image` `flash_energy_partition` `shock` |
| `analysis/osiris/` | `shock_state` `streak` `reflected_ions` `cross_shock_potential` `field_particle_correlation` `energy_flux` `synthetic_diagnostics` |
| `analysis/warpx/` | `spitzer_resistivity`, plus new `flash` `metrics` `plotting` (Phase 5) |

### `analysis_utils` — decided: moves to `common/` whole

`analysis_utils` goes to `common/` unsplit, per the literal rule. It imports
`osiris_utils`, `osh5def` and `osh5io` at module scope, so `common/` is **not** CI-pure.
That is accepted: the OSIRIS stack is present in the `analysis` and `osiris2` envs, so
`pytest` stays green on Perlmutter, where it is actually run.

The consequence to handle is mechanical, not philosophical. The current
`tests/analysis_utils.py` stub works by *shadowing* a flat module via `sys.path` ordering,
and that stops working once the module is `magshockz.common.analysis_utils` — a package
submodule cannot be shadowed by a same-named file on `sys.path`. Since six library modules
import it (`energy_partition`, `flash_source`, `plot_style`, `shock_state`, `streak`,
`temperature_anisotropy`), a hard import failure takes ~700 lines of tests with it.

Replacement, if GitHub Actions is to stay green: stub the **third-party** modules rather
than our own, by injecting `osiris_utils` and `osh5io` into `sys.modules` in
`tests/conftest.py`. `tests/osh5def.py` already works this way and survives the move
untouched, because `osh5def` is a genuine top-level import. This is about eight lines and
removes the `sys.path` reordering entirely.

If CI is instead allowed to go red, delete `tests/analysis_utils.py` and the path-ordering
block, and drop the workflow — a permanently failing check is worse than no check.

## Phases

Ordered so that each phase leaves the tree working and testable.

### Phase 0 — safety net

- Land the uncommitted WarpX work as its own commit on the current layout, so the
  restructure diff stays separable from it.
- Record the baseline: `pytest` green, and the figure outputs of two or three scripts
  kept for byte-comparison after the move.
- Remove stray artifacts: `slurm-55359733.out`, `.coverage`, `docs/.~lock.*.ods#`.
  Add `.coverage`, `.pytest_cache/`, `._dbindex_/` to `.gitignore`.

### Phase 1 — make it importable (the actual "hook" fix)

This is the highest-value phase and is independent of every move below.

- `pyproject.toml`'s `build-backend` is `setuptools.backends.legacy:build`, which is not a
  real backend — this is why the package has never installed. Fix to
  `setuptools.build_meta`, add `[tool.setuptools] packages = ["magshockz", ...]`.
- `pip install -e .` into both `analysis` and `osiris2`.
- Result: `from magshockz.analysis.flash import piston_profile` works from a notebook, a
  REPL, an sbatch script or another repo — with no path manipulation.

### Phase 2 — delete dead weight

Verified dead, nothing imports them:

- `init_nopython/` — the three `sys.path.insert` lines pointing at it are vestigial; no
  script imports `fitting_functions`, and `Ray`/`pwlf` appear nowhere else. Delete the
  directory and those three lines.
- `init_common/` — referenced only by its own hand-rolled test.
- `simulations/` — untouched since 2025-08.
- `init_python/` — the Python moved to the external `flash2osiris`; the remaining `.sh`
  wrappers move to `runs/` beside the specs they invoke.

Git retains all of it; nothing here is recoverable only from the working tree.

### Phase 3 — the move

- `git mv` `src/` → `magshockz/` and place each module per the map above. Using `git mv`
  keeps history attached, so `git log --follow` still works.
- Add `__init__.py` at every level; keep them thin. The current `src/__init__.py`
  re-exports 60-odd names into a flat namespace — that flattening is what made the
  package opaque, so it does not carry over. Each subpackage exports its own names.
- Rewrite imports to absolute package paths and delete all 24 `sys.path.insert` blocks.
- `pytest` must be green before proceeding.

### Phase 4 — scripts and shims

Flat, code-prefixed. This also resolves the four `scripts/`↔`src/` name collisions
(`energy_flux`, `synthetic_diagnostics`, `spitzer_resistivity`, `dimensionless_params`),
which today only work because the `sys.path` insert happens to win.

| current | new |
|---|---|
| `overview.py` | `osiris_overview.py` |
| `tune_shock.py` | `osiris_tune_shock.py` |
| `make_movie.py` | `osiris_make_movie.py` |
| `pressure_partition.py` | `osiris_pressure_partition.py` |
| `temperature_ratios.py` | `osiris_temperature_ratios.py` |
| `heating_decomposition.py` | `osiris_heating_decomposition.py` |
| `convergence_scan.py` | `osiris_convergence_scan.py` |
| `energy_flux.py` | `osiris_energy_flux.py` |
| `synthetic_diagnostics.py` | `osiris_synthetic_diagnostics.py` |
| `dimensionless_params.py` | `osiris_dimensionless_params.py` |
| `validate_init.py` | `osiris_validate_init.py` |
| `tune_flash_shock.py` | `flash_tune_shock.py` |
| `run_flash_pressure_partition.py` | `flash_pressure_partition.py` |
| `make_warpx_deck.py` | `warpx_make_deck.py` |
| `spitzer_resistivity.py` | `warpx_spitzer_resistivity.py` |
| `plots_for_paper/*/…_plots_for_paper.py` | `paper_shadowgraphy.py`, `paper_xrs3.py` |

Unchanged: the seven already-prefixed `flash_*.py`, `warpx_heater_compare.py`,
`warpx_flash_evolution.py`, `osiris_rh_prediction.py`, `flash_osiris_compare.py`
(cross-code by nature).

**Shims protecting the queued job.** Job `warpx_heater_2d` (16 nodes) runs
`init_warpx/run_heater_2d.sbatch` by absolute path with `WorkDir` at the repo root, and
reads its deck and `run_env.sh` from `input_files/warpx/…` at runtime. Therefore:

- `init_warpx/` and `input_files/` do **not** move this round.
- `scripts/make_warpx_deck.py` stays as a one-line forwarding shim to
  `warpx_make_deck.py`, because the sbatch's help text and the `--verify` loop name it.
- Old script names remain as forwarding shims until the job completes, then are deleted
  in a single follow-up commit.

`plots_for_paper/` images and the TIF move to `docs/figures/` or beside the paper
scripts; `analysis_notebooks/` → `notebooks/`, dropping `__pycache__/`.

### Phase 5 — finish the WarpX subpackage

`src/warpx/__init__.py` already advertises `flash`, `metrics` and `plotting` submodules
that do not exist, and `calibration.py` has no importer. Complete it, making WarpX the
worked example of the new layout:

- `analysis/warpx/flash.py` — the FLASH side (`flash.par` IC, measured piston), lifted out
  of `warpx_flash_evolution.py`.
- `analysis/warpx/metrics.py` — front tracking and the dimensionless scorecard, lifted out
  of `warpx_heater_compare.py`.
- `analysis/warpx/plotting.py` — the comparison figures.
- `calibration.py` — wire up or delete; decide when its caller is written.

Those two scripts are 680 and 462 lines and hold real numerics; this is where the
library-first rule pays off, and each extracted function gets a test.

### Phase 6 — prose

28% of `magshockz/`+`scripts/` (5,336 of 18,808 lines) is docstring or comment. Policy
chosen: **cut narration, keep physics.**

Remove: prose that restates the code, narrates history ("changed from X to Y"), or
re-derives what a name already says. Module docstrings drop to purpose plus CLI usage.

Keep inline: every physics caveat — the `sqrt(4π)` convention, the measured `kappa`, the
absolute-vs-relative front threshold, the `Zbar` reasoning, the `strings | grep -c`
SIGPIPE guard. These are the comments `CLAUDE.md` itself says earn their place.

Move to `docs/`, not delete: the extended narrative currently living in 34–70 line module
docstrings. `warpx_flash_evolution.py`'s 52-line docstring becomes a few lines plus a
pointer to `docs/warpx_heater.md`.

Where a comment exists because a name is poor or a function too long, fix the name or
split the function instead — that is the codebase's own stated rule.

### Phase 7 — documentation

`CLAUDE.md` is 810 lines and is currently the only real documentation, written for an AI
reader. Split it:

- `docs/user_guide.md` (currently empty) — how to run everything: environments, the three
  pipelines end to end, every command with its flags.
- `docs/physics_notes.md` — the conventions and hard-won corrections: OSIRIS normalized
  units, the FLASH `sqrt(4π)`, the PyYAML 1.1 float trap, the reduced-mass caveats.
- `docs/warpx_heater.md` — the heater pipeline and the constraints that shaped the deck.
- `docs/osiris_pipeline.md` — run specs, `RunSpec` resolution, collisions.
- `CLAUDE.md` → roughly 150 lines: layout, conventions, code style, and pointers into
  `docs/`.
- `README.md` — orientation and a real quick-start now that `pip install -e .` works.

Configs keep the `runs/` vs `config/` split, which encodes a real distinction. Each file
gains a top-level `schema:` key for self-description, and each directory gets one loader
that dispatches on it — extending the pattern the WarpX specs already use.

## What this fixes

| complaint | fix |
|---|---|
| "opaque" | Directory names describe stage and code, not history. Seven top-level dirs become four load-bearing ones. |
| "no clear hooks without Claude Code" | `pip install -e .` makes `import magshockz` work anywhere; 24 `sys.path` hacks disappear; `docs/user_guide.md` documents every command. |
| "over-reliance on long comments" | 28% prose cut toward ~10%, with physics preserved inline and narrative harvested into `docs/`. |
| "unnecessary files / merges" | Four dead directories deleted; four name collisions resolved; two 600-line scripts have their numerics extracted into a tested library. |

## Sequencing note

Phases 0–2 are safe to run now and are independently valuable. Phase 1 alone delivers most
of the "hooks" benefit. Phases 3–5 are the structural move. Phase 4's shims exist solely
to protect the queued 16-node job and should be deleted once it completes.
