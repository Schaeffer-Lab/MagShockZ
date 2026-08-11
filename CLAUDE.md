# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MagShockZ analyzes magnetized collisionless shock simulations for the Magnetized
Collisionless Shocks on Z (MagShockZ) experiment. Its core job is converting **FLASH**
MHD simulation output into initialized **OSIRIS** PIC input decks, then analyzing the
resulting OSIRIS runs (and the source FLASH data). A third code, **WarpX**, runs both
Ohm's-law hybrid and full-PIC piston simulations off the same FLASH data. Most work runs
on NERSC Perlmutter.

## Documentation

`docs/` is written for a human reader and is the reference for anything below. **Read the
relevant one before changing that pipeline** — each records constraints that bite silently.

| document | covers |
|---|---|
| [docs/user_guide.md](docs/user_guide.md) | every command, both environments, the shared flags |
| [docs/osiris_pipeline.md](docs/osiris_pipeline.md) | run specs, `RunSpec` resolution, deck generation, collisions |
| [docs/flash_analysis.md](docs/flash_analysis.md) | FLASH config resolution, results directories, experimental streak images |
| [docs/warpx_pipeline.md](docs/warpx_pipeline.md) | both WarpX schemas; the heater deck's hard-won constraints |
| [docs/physics_notes.md](docs/physics_notes.md) | OSIRIS normalized units, the FLASH `sqrt(4π)`, the PyYAML 1.1 float trap |

The rest of this file is what governs *editing* the repository: its layout, its
architecture, and its code style.

## Environments

Two conda environments, used for disjoint stages — do not mix them. Full detail in
[docs/user_guide.md](docs/user_guide.md).

- **`osiris2`** — FLASH→OSIRIS initialization and deck generation. The converter itself is
  the standalone **`flash2osiris`** package (pip-installed here; repo at
  `/pscratch/sd/d/dschnei/flash2osiris`). Has `yt` + `unyt` and `jinja2`.
- **`analysis`** — OSIRIS and WarpX analysis (`scripts/`). Has `osh5` / pyVisOS and
  `osiris_utils`, plus `yt` + `unyt`, so the FLASH-side scripts run here too.

The package (`magshockz/`, see `pyproject.toml`) is installed editable into both, and
depends on numpy/scipy/unyt + astropy/plasmapy/yt — all of which CI installs, so it may
use them freely. The one stack CI does **not** have is OSIRIS's (`osiris_utils`, `osh5*`),
which is not pip-installable; those imports are stubbed in `tests/` (see Tests below).

## Layout

```
magshockz/            the installable package -- all pure, testable logic
  common/             used by two or more codes
  init/{osiris,warpx} deck generation and validation
  analysis/{flash,osiris,warpx}
scripts/              thin CLI wrappers, flat, prefixed by code (flash_/osiris_/warpx_/paper_)
runs/                 run specs: the inputs that DEFINE a run (*.run.yaml, *.warpx.yaml)
config/               analysis configs: how to ANALYSE a run
init_warpx/           sbatch + build scripts only; nothing imports from it
tests/                flat
docs/                 the human-facing documentation above
input_files/          generated run trees (gitignored, regenerable)
```

Stage is the top-level split inside the package, code the second:
`from magshockz.analysis.flash import experiment_image`.

## Architecture

### Single source of truth: the run spec

Each run's parameters live **once**, in a `run.yaml` in the run's own directory.
`runs/*.run.yaml` are the version-controlled inputs; the flash2osiris generator freezes a
resolved copy to `<run_dir>/run.yaml`. Analysis reads parameters back through
`magshockz/common/run_spec.py::RunSpec` instead of re-copying them into analysis configs.

`RunSpec.from_sim_dir()` resolves in priority order (first hit wins): `run.yaml` →
`run_manifest.yaml` (parse its `cli_command`) → legacy `runme*.sh` (parse python flags).
In a `run.yaml`, the `geometry` / `solver` / `diagnostics` groups exist purely for
readability and are flattened to top-level keys (matching the original CLI flags);
`species_names` / `charge_states` stay nested as metadata. `RunSpec` is deliberately
dependency-light (stdlib + PyYAML + astropy) so it is unit-testable without the OSIRIS stack.
The same file records how the FLASH-side configs resolve their data, and how a variant
config claims its own results directory — see [docs/flash_analysis.md](docs/flash_analysis.md).

### Analysis: library-first

Pure, testable functions live in the **`magshockz/`** package, imported by their real
module path (`from magshockz.common import energy_partition`) — the package `__init__`s
are deliberately thin, because the old flat re-export of ~60 names into one namespace hid
where anything lived and dragged the OSIRIS stack in on any import. The thin,
plotting/IO-heavy **`scripts/`** orchestrate them. Each script:
is `--config` driven (with a `$MAGSHOCKZ_SIM_DIR` override), uses
`analysis_utils.MagShockZRun` for unit/field context, `analysis_utils.detect_layout` /
`RunLayout` for dimension-agnostic (1D/2D) axis handling, reads with `osh5io`, plots with
`osh5vis` (metadata-sourced labels/units), and saves under `results/<run_name>/`.

The same rule governs the WarpX heater generator: `init_warpx/` holds CLIs and sbatch
scripts only, and **nothing imports from it**. A script that needed the run-spec loader
used to reach backwards into `scripts/warpx_make_deck.py`; it now goes through
`magshockz/init/warpx/config.py` like any other library module.

`MagShockZRun` wraps an OSIRIS deck (via `osiris_utils.Simulation`) for field access and
astropy-unit conversions (cyclotron/ion frequencies, gyrotime). FLASH-side analysis uses
yt + unyt and does **not** go through `MagShockZRun`.

The package is installed editable (`pip install -e .`) into both envs, so `import
magshockz` works from a script, a notebook, a REPL or an sbatch file with no `sys.path`
manipulation anywhere.

Every plotting script takes a shared `--publication` (alias `--pub`) flag: off by default
(matplotlib's own sizes, so saved figures are unchanged), on it bumps all text to large
paper/slide sizes. The look lives once in `magshockz/common/plot_style.py`; a script calls
`plot_style.add_publication_arg(parser)` before `parse_args` and `plot_style.apply(args.publication)`
after. It is rcParams-only (set before any figure is drawn, so it also restyles `make_movie`'s
forked render workers) and imports matplotlib lazily, so it stays out of the CI-pure layer.

`magshockz/common/plot_style.py` also owns the **display-unit** mapping (the second shared flag,
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

CI installs the package (`pip install -e .`) plus pytest, so tests may use numpy, scipy,
unyt, astropy, plasmapy, yt and PyYAML freely. The only thing it cannot install is
**OSIRIS's own stack** (`osiris_utils`, `osh5io`, `osh5def`, `osh5vis`), which is not
pip-installable.

`tests/conftest.py` therefore tries to import that stack and, only when it is absent,
injects minimal stand-ins into `sys.modules` (plus `tests/osh5def.py` on `sys.path`). It
stubs the **third-party** modules, not ours — once `analysis_utils` became
`magshockz.common.analysis_utils`, a same-named file on `sys.path` could no longer shadow
it, because a package submodule is resolved through its parent. On Perlmutter, where the
real stack is present, nothing is stubbed and the suite runs against it.

plasmapy (~2.5 s) and yt are still imported **lazily** in a few `magshockz/` modules —
that is purely an import-time cost decision for scripts, not a CI constraint, so don't add
a new lazy import just to keep a module "pure".

One thing yt-in-CI does *not* buy: `magshockz/common/flash_utils.py` is still un-importable there,
because it calls `yt.enable_plugins()` at module scope and yt **raises**
`FileNotFoundError` when there is no `~/.config/yt/my_plugins.py` (the flash2osiris
plugin, symlinked in on Perlmutter). Testing `flash_utils` in CI needs that call guarded
first — installing yt is necessary but not sufficient.

`tests/test_warpx_deck.py` asserts on `key_params` — *what WarpX will do* — rather than on
the deck text wherever the claim is about behaviour, so rewording a comment is not a
failure but moving a constant is. Two tests are worth knowing about: the null control's
rendered deck must differ from production in **only** `heater.*` keys (that is what makes
it a control), and `test_it_still_means_what_the_running_jobs_deck_means` re-renders the
checked-in spec and requires `n_cell` / `prob_lo` / `prob_hi` / `max_step` to be **exactly**
equal to the deck in `input_files/` — a checkpoint is resumable only at the same grid *and*
the same domain, and the sbatch's guard only checks the former.

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
    (primed quantities are dimensionless by construction — see
    [docs/physics_notes.md](docs/physics_notes.md)), and
    in numerical kernels where attaching units per-element would be a real cost. Strip
    units at that boundary deliberately (`.to("cm").value`) rather than letting a plain
    array of unknown unit propagate.
- **Names describe the physical thing, not its formatting.** `upstream_density` over `n`;
  `shock_position`/`x_shock` over `xs`. Physics symbols (`rqm`, `M_A`, `beta`, `T_ci`,
  `d_i`, `ne`, `ti`) are good names where they are the field's own vocabulary and already
  documented in [docs/physics_notes.md](docs/physics_notes.md) — do not inflate them into
  prose. A bare `i`/`j`/`k` is fine for a loop
  index and `x`/`y` for plain array axes. The test is whether a reader knows *what it is*;
  the library answers *what unit it's in*.
- **Type-hint every function in `magshockz/`** — arguments and return. Prefer precise types
  (`np.ndarray`, `Path`, `Sequence[float]`, `Literal["electron", "ion"]`,
  `dict[str, float] | None`) over bare `Any`/untyped. Scripts should be hinted too where
  it costs nothing; the pure-function layer in `magshockz/` is where it is required.
- **`@dataclass` for anything with more than ~three related fields** and for every
  multi-value return — the pattern `RunSpec` / `FlashSource` / `DisplayUnits` /
  `RunLayout` already establish. A named field beats a tuple index; returning a
  `(a, b, c, d)` tuple that the caller unpacks positionally is a bug waiting to happen.
  Use `frozen=True` for value objects that shouldn't mutate after resolution.
- **Small pure functions with one job, named for what they return.** If you need a
  comment shaped like `# now do X`, that block is a function called `x(...)`. Keep the
  numerical core pure (arrays in, arrays out, no IO, no globals, no plotting) so it is
  unit-testable in the CI-pure `magshockz/` layer — this is the same rule as "library-first"
  above, stated as a style rule.
- **Structure carries meaning.** Guard clauses and early returns instead of nested `if`
  pyramids; extract complex boolean conditions into a named local
  (`is_downstream = x > x_shock`); one blank-line-separated paragraph per logical step
  inside a longer function; unpack config dicts into named locals near the top rather
  than indexing `cfg["a"]["b"]` deep in the arithmetic. Avoid magic numbers — reach for
  plasmapy/`unyt` constants first, and where the literal is genuinely ours, a named
  module-level constant (`SQRT_4PI`) says what it means without a comment.
- **Docstrings, not inline narration.** Every public function in `magshockz/` gets a short
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
  and must NOT be overridden (see [docs/physics_notes.md](docs/physics_notes.md)).
- 2D-capable analysis treats `x2` as the shock-normal axis; use `detect_layout` /
  `transverse_profile` rather than hardcoding dimensionality.
- Plan before implementing analysis changes; prefer adding a pure function to `magshockz/`
  (with a test) over embedding logic in a script.
