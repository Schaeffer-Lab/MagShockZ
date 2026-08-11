# MagShockZ

Analysis code for the Magnetized Collisionless Shocks on Z (MagShockZ) experiment. It
converts **FLASH** MHD output into initialized **OSIRIS** PIC input decks, drives
**WarpX** hybrid and full-PIC piston runs off the same data, and analyses all three. Most
work runs on NERSC Perlmutter.

## Quick start

```bash
conda activate analysis
pip install -e .                    # editable; also do this in the osiris2 env

python scripts/osiris_overview.py --config config/perlmutter_1.3.1d.yaml
```

`import magshockz` then works from a script, a notebook, a REPL or an sbatch file with no
path manipulation:

```python
from magshockz.common import run_spec, piston_profile
from magshockz.init.warpx import units
```

## Documentation

**[docs/user_guide.md](docs/user_guide.md) is the place to start** — every command, both
environments, the shared flags.

| document | covers |
|---|---|
| [docs/user_guide.md](docs/user_guide.md) | every command, both environments, the shared flags |
| [docs/osiris_pipeline.md](docs/osiris_pipeline.md) | run specs, `RunSpec` resolution, deck generation, collisions |
| [docs/flash_analysis.md](docs/flash_analysis.md) | FLASH config resolution, results directories, experimental streak images |
| [docs/warpx_pipeline.md](docs/warpx_pipeline.md) | both WarpX schemas; the heater deck's constraints |
| [docs/physics_notes.md](docs/physics_notes.md) | OSIRIS normalized units, the FLASH `sqrt(4π)`, the PyYAML 1.1 float trap |

[`CLAUDE.md`](CLAUDE.md) is the guidance file for Claude Code: layout, architecture and
code style. It is not a substitute for the above.

## Layout

```
magshockz/            the installable package -- all pure, testable logic
  common/             used by two or more codes
  init/{osiris,warpx} deck generation and validation
  analysis/{flash,osiris,warpx}
scripts/              thin CLI wrappers, flat, prefixed by code (flash_/osiris_/warpx_/paper_)
runs/                 run specs: the inputs that DEFINE a run (*.run.yaml, *.warpx.yaml)
config/               analysis configs: how to ANALYSE a run
init_warpx/           WarpX sbatch + build scripts
notebooks/            exploratory notebooks
tests/                pytest suite
docs/                 the documentation above
input_files/          generated run trees (gitignored, regenerable)
```

Each run's parameters live **once**, in a `run.yaml` in the run's own directory; analysis
reads them back through `magshockz/common/run_spec.py::RunSpec` rather than re-copying
them into analysis configs.

## Environments

Two conda environments, used for disjoint stages — do not mix them:

- **`osiris2`** — FLASH→OSIRIS deck generation. The converter is the standalone
  [`flash2osiris`](https://github.com/Schaeffer-Lab/flash2osiris) package.
- **`analysis`** — OSIRIS/WarpX/FLASH analysis; has pyVisOS (`osh5io`/`osh5def`/`osh5vis`),
  `osiris_utils`, `yt` and `unyt`.

## Tests

```bash
pytest                                        # full suite
pytest --cov=magshockz --cov-report=term-missing
```

CI installs the package and runs the same suite. The OSIRIS stack is not pip-installable,
so `tests/conftest.py` stubs it when absent and runs against the real one on Perlmutter.
