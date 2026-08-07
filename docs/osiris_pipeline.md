# init_python/ — FLASH→OSIRIS generation drivers

The FLASH→OSIRIS converter (the generator, the yt plugin, and the deck/py-init
templates) **no longer lives here**. It was extracted into the standalone, reusable
package **[`flash2osiris`](https://github.com/Schaeffer-Lab/flash2osiris)**
(`/pscratch/sd/d/dschnei/flash2osiris`), which separates OSIRIS ion populations by FLASH
material (target vs. chamber) and works for arbitrary plasmas.

This directory now holds only the **MagShockZ-specific run drivers**. The run parameters
themselves are the single source of truth in `../runs/*.run.yaml`.

## Setup (one time)

```bash
# Install the package into the generation env and link its yt plugin:
/global/homes/d/dschnei/.conda/envs/osiris2/bin/pip install -e /pscratch/sd/d/dschnei/flash2osiris
ln -sfn /pscratch/sd/d/dschnei/flash2osiris/flash_osiris/yt_plugin.py ~/.config/yt/my_plugins.py
```

## Generate a deck

```bash
bash init_python/runme_perlmutter_1d.sh        # 1D, runs/perlmutter_1d.run.yaml
bash init_python/runme_perlmutter_2d.sh        # 2D, runs/perlmutter_2d.run.yaml
bash init_python/runme_perlmutter_1d_rqm1.sh   # 1D, rqm_factor=1 convergence run
bash init_python/run_dx_scan.sh [nominal_dx]   # dx convergence scan (config + per-point overrides)
```

Each wrapper runs `python -m flash_osiris.generator --config runs/<spec>.run.yaml` from
the repo root; the deck, py-init script, interp slices and a frozen `run.yaml` are
written under `input_files/<name>.<dim>d/`. Analysis reads that frozen `run.yaml` via
`src/run_spec.py::RunSpec`.

> Ion populations come from the FLASH materials. `species_names: {cham: al, targ: si}`
> in the run specs keeps the OSIRIS species named `al`/`si` (cham = Al chamber,
> targ = Si target) while separating by material rather than ion mass.
