# OSIRIS pipeline

Generating an OSIRIS deck from a FLASH dump, and the run spec that is the single source of
truth for both the generation and the analysis that follows.

## Setup (one time)

```bash
# Install the generator into the generation env and link its yt plugin:
/global/homes/d/dschnei/.conda/envs/osiris2/bin/pip install -e /pscratch/sd/d/dschnei/flash2osiris
ln -sfn /pscratch/sd/d/dschnei/flash2osiris/flash_osiris/yt_plugin.py ~/.config/yt/my_plugins.py
```

## Generate a deck

```bash
conda activate osiris2
python -m flash_osiris.generator --config runs/perlmutter_1d.run.yaml   # run from the repo root
```

Wrappers for the standard runs:

```bash
bash runs/runme_perlmutter_1d.sh        # 1D, runs/perlmutter_1d.run.yaml
bash runs/runme_perlmutter_2d.sh        # 2D, runs/perlmutter_2d.run.yaml
bash runs/runme_perlmutter_1d_rqm1.sh   # 1D, rqm_factor=1 convergence run
bash runs/run_dx_scan.sh [nominal_dx]   # dx convergence scan
```

The deck, py-init script, interp slices and a frozen `run.yaml` land in
`input_files/<name>.<dim>d/`. The generator's CLI is **terminal-only with no hidden
defaults**: argparse enforces required-ness once, every parameter is explicit in the run
spec, and CLI flags override individual spec keys.

## Single source of truth: the run spec

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

## The generator: flash2osiris (external)

Deck generation lives in the standalone **`flash2osiris`** package
(`/pscratch/sd/d/dschnei/flash2osiris`, pip-installed into `osiris2`); MagShockZ keeps
only the run specs (`runs/`) and thin drivers (`init_python/`). `flash_osiris.generator`
(class `FLASH_OSIRIS_Base`, 1D/2D subclasses) reads FLASH via `yt`, derives OSIRIS
normalizations and per-population `rqm` (edens-weighted 1836/ye), and renders the deck +
py-init templates; `dt` is the CFL condition (`dx*0.95/sqrt(dims)`). Ion populations are
separated by **FLASH material** (target/chamber) via the yt plugin
(`~/.config/yt/my_plugins.py` → `flash_osiris/yt_plugin.py`), not by ion mass; the run
spec's `species_names: {cham: al, targ: si}` renames them. Units stay yt-native + `unyt`.

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
