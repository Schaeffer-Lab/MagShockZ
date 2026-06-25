# MagShockZ

Analysis code for the Magnetized Collisionless Shocks on Z (MagShockZ)
experiment. Its core job is converting **FLASH** MHD simulation output into
initialized **OSIRIS** PIC input decks, then analyzing the resulting OSIRIS runs
(and the source FLASH data). Most work runs on NERSC Perlmutter.

> **Authoritative reference:** [`CLAUDE.md`](CLAUDE.md) documents the full
> architecture, the two conda environments, and every command. This README is a
> short orientation — see `CLAUDE.md` for details.

## Layout

- `runs/*.run.yaml` — version-controlled run specs (the single source of truth
  for each run's parameters).
- `init_python/` — thin drivers/wrappers around the external **`flash2osiris`**
  package (the FLASH→OSIRIS converter; pip-installed, repo at
  `/pscratch/sd/d/dschnei/flash2osiris`).
- `src/` — the pure, unit-testable analysis library (numpy/scipy/unyt only).
- `scripts/` — `--config`-driven analysis orchestration (plotting/IO).
- `analysis_notebooks/` — exploratory notebooks.
- `config/*.yaml` — analysis configs (inspection results; parameters are read
  back through `src/run_spec.py::RunSpec`, not re-copied here).
- `tests/` — pytest suite for the `src/` library.

## Environments

Two conda environments, used for disjoint stages — do not mix them:

- **`osiris2`** — FLASH→OSIRIS initialization and deck generation
  (`flash2osiris` + `init_python/`).
- **`analysis`** — OSIRIS/FLASH analysis (`scripts/`); has pyVisOS
  (`osh5io`/`osh5def`/`osh5vis`), `osiris_utils`, and `yt`/`unyt`.

## Quick start

```bash
# Generate an OSIRIS deck from a run spec
conda activate osiris2
python -m flash_osiris.generator --config runs/perlmutter_1d.run.yaml

# Run an analysis script
conda activate analysis
python scripts/overview.py --config config/perlmutter_1.3.1d.yaml

# Tests (CI): pip install numpy scipy unyt pyyaml pytest pytest-cov
pytest
```

See [`CLAUDE.md`](CLAUDE.md) for the full command list, the run-spec / config
single-source-of-truth design, and the OSIRIS normalized-unit conventions.

## Contributing

Please contact me through GitHub (ID: dschneidinger) with questions, or if you
would like to use these tools to convert other FLASH simulation data to OSIRIS.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for more details.
