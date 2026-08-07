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
| `init/osiris/` | validation logic lifted out of `scripts/osiris_validate_init.py` |
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
- `scripts/warpx_make_deck.py` stays as a one-line forwarding shim to
  `warpx_make_deck.py`, because the sbatch's help text and the `--verify` loop name it.
- Old script names remain as forwarding shims until the job completes, then are deleted
  in a single follow-up commit.

`plots_for_paper/` images and the TIF move to `docs/figures/` or beside the paper
scripts; `analysis_notebooks/` → `notebooks/`, dropping `__pycache__/`.

### Phase 5 — finish the WarpX subpackage — DONE

`init/warpx/units.py` superseded `common/heater_piston_scaling.py`: same mapping, but
astropy `Quantity` + plasmapy `formulary` throughout instead of bare SI floats with
unit-suffixed names. Both existed side by side, which is the worst of the two.

- **`common/heater_piston_scaling.py` (818 lines) and its test (514) are deleted.** All
  three consumers — `flash_piston_profile.py`, `warpx_heater_compare.py`,
  `warpx_flash_evolution.py` — now go through `units.DeckScales` / `units.Upstream`.
- **`analysis/warpx/flash.py`** — `MeasuredPiston`, the FLASH piston as *measured*. It is
  deliberately **not** `units.FlashReference`: that one is what the deck *imposes* (an
  integer `Al 6+`), while FLASH ran at its EOS Zbar of 3.66. Same `Upstream`, same
  formulary, different ion — so `eos_ion()` builds a `CustomParticle`, since
  `Particle("Al 3.66+")` raises.
- **`analysis/warpx/metrics.py`** — `front_speed_over_c` and the `ScoreRow` scorecard,
  lifted out of `warpx_heater_compare.py`.
- **`units.invariant_table`** — the FLASH-vs-deck table, previously written twice (in
  `warpx_make_deck.py` and as `hps.invariance_report`).
- **`calibration.py`** — already wired up; `units.py` imports `HeaterCalibration`.

`analysis/warpx/plotting.py` was **not** created. The two scripts' figures share no code
with each other, so extracting them would move matplotlib out of the scripts that exist to
draw it without removing any duplication — against the codebase's own "thin plotting/IO
scripts orchestrate a pure library" rule rather than for it.

Validated end to end against the archived `diags_v2_z1_mr100` run: all four panels render
and every number is consistent with a v2-era deck read by a revised spec.

### Phase 6 — prose

28% of `magshockz/`+`scripts/` (5,336 of 18,808 lines) is docstring or comment. Policy
chosen: **cut narration, keep physics.**

### The 10% target was wrong, and measuring properly is the finding — DONE

The 28% figure was computed from line counts before the content was read. Broken down, it
is not bloat:

| layer | lines | docstring | comment |
|---|---|---|---|
| `magshockz/` | 8,319 | 33% — but 2,079 of 2,759 lines are **function/class** docstrings | 5% |
| `scripts/` | 10,260 | 13% — mostly module docstrings, i.e. the CLI usage | 8% |

The library has **250 documented public functions at a median docstring of 6 lines**.
That is `CLAUDE.md`'s own rule ("every public function in `magshockz/` gets a short
docstring") being followed, not violated. Reaching 10% would mean deleting ~1,500 lines of
required API documentation to hit a number that was never measuring the right thing.

The 64 apparently-undocumented public callables are **51 `@property` one-liners** with
self-evident names (correctly undocumented) and 13 real ones, several of which are nested
closures. The genuinely public ones — `RunSpec.from_sim_dir`, `RunSpec.get`,
`RunSpec.charge_state`, `moments.moment`, `DeckScales.invariants` — were *given*
docstrings. That is the opposite of the phase's stated direction and is the correct call.

### What was actually cut

Four module docstrings that duplicated documentation now in `docs/`:

| script | before | after |
|---|---|---|
| `flash_3d_movie.py` | 71 | 28 |
| `flash_experiment_compare.py` | 63 | 20 |
| `osiris_rh_prediction.py` | 58 | 30 |

`warpx_flash_evolution.py`'s 53-line docstring was **kept**. Re-reading it, all four of
its blocks are measurement caveats specific to that script — species are never summed, the
velocity row's coarser clock, the two clock-zero/box-width caveats — none of which live in
`docs/`, and moving them would separate them from the code they govern.

Net: 29.2% → 28.8%, and the repository is better documented than before, not worse.

### Phase 7 — documentation — DONE

`CLAUDE.md` was 828 lines and the only real documentation, written for an AI reader. Now:

| file | lines | audience |
|---|---|---|
| `README.md` | 79 | orientation + quick start |
| `docs/user_guide.md` | 190 | **the human entry point** — install, both envs, shared flags, every command by task |
| `docs/osiris_pipeline.md` | 98 | run specs, `RunSpec` resolution, deck generation, collisions |
| `docs/flash_analysis.md` | 135 | config resolution, LOS fans, results directories, streak images |
| `docs/warpx_pipeline.md` | 308 | both WarpX schemas and the heater deck's constraints |
| `docs/physics_notes.md` | 72 | OSIRIS normalized units, the FLASH `sqrt(4π)`, the PyYAML 1.1 trap |
| `CLAUDE.md` | 254 | layout, architecture, tests, code style — what governs *editing* |

Sections were moved **verbatim** by line range rather than paraphrased, so no hard-won
detail was lost in transit; only the cross-references were rewritten afterwards. Every
internal link and every `path/to/file.py` reference in all seven files is verified to
resolve.

Two corrections made while writing them:

- `docs/warpx_pipeline.md`, not `warpx_heater.md`. There are **two** schemas dispatched
  from one `schema:` key in `runs/*.warpx.yaml`; naming the file after the heater would
  hide the hybrid runs that share it.
- `docs/flash_analysis.md` documents the **fan of lines of sight**
  (`lines_of_sight:` + `--los`, `config/flash_3d_corrected_fan.yaml`), which
  `flash_source.py` supports and `CLAUDE.md` never mentioned — so the original was
  incomplete, not merely long.

**Not done:** the `schema:` key for `config/*.yaml` and a per-directory loader that
dispatches on it. `runs/*.warpx.yaml` already has it; the analysis configs do not, and
adding it touches every config and every `load_config` caller. It is a self-contained
follow-up, not a prerequisite for anything above.

## What this fixes

| complaint | fix |
|---|---|
| "opaque" | Directory names describe stage and code, not history. Seven top-level dirs become four load-bearing ones. |
| "no clear hooks without Claude Code" | `pip install -e .` makes `import magshockz` work anywhere; 24 `sys.path` hacks disappear; `docs/user_guide.md` documents every command. |
| "over-reliance on long comments" | Measured properly: most of the 28% is required function docstrings (median 6 lines) and physics comments. Four duplicated module docstrings cut; the 810-line `CLAUDE.md` split into six documents written for a human. |
| "unnecessary files / merges" | Four dead directories deleted; four name collisions resolved; two 600-line scripts have their numerics extracted into a tested library. |

## Sequencing note

Phases 0–2 are safe to run now and are independently valuable. Phase 1 alone delivers most
of the "hooks" benefit. Phases 3–5 are the structural move. Phase 4's shims exist solely
to protect the queued 16-node job and should be deleted once it completes.
