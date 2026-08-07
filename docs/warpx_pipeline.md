# WarpX pipelines

Two different things share `runs/*.warpx.yaml`, dispatched on a top-level `schema:` key:
the **flash2warpx hybrid** runs, which slice a FLASH dump and step WarpX's Ohm's-law
solver over it, and the **heater-driven piston** runs (`schema: heater_pic_2d`), which are
full electromagnetic PIC and read no FLASH data at all.

## Hybrid generation: the `flash2warpx` package (external)

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
`magshockz/analysis/warpx/spitzer_resistivity.py`) reads these trees to pick `run.plasma_resistivity`.

## Heater-driven piston runs: `schema: heater_pic_2d` (no FLASH extraction)

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
  *dimensionless* terms only. `magshockz/init/warpx/units.py` owns that mapping and is
  explicit about both halves: preserved are `M_A`, `M_ms`, `beta_e/beta_i`,
  `n_piston/n_amb`, `r_spot/d_i`, `omega_ci·t`; deliberately broken are `m_i/(Z m_e)`,
  `Z`, `omega_pe/omega_ci`, and absolute ns/µm. `units.invariant_table(scales)` renders
  both halves as one FLASH-vs-deck table — the generator prints it and
  `warpx_heater_compare.py` saves it; read it. `DeckScales.to_time` / `to_length` bridge
  back through the matched *ion* scales (`d_i`, `T_ci`), never through `1/omega_pe`.

**Where the code lives.** `init_warpx/` holds only sbatch and build scripts; the CLI is
`scripts/warpx_make_deck.py`, and nothing imports from either. The library modules:

| module | role |
|---|---|
| `magshockz/init/warpx/units.py` | FLASH reference → deck constants; the invariants |
| `magshockz/init/warpx/deck.py` | render the deck; parse/resolve/`key_params`/`verify` it back |
| `magshockz/init/warpx/config.py` | `load` (raises) / `scales` / `validate` (warns) / `freeze` |
| `magshockz/init/warpx/calibration.py` | the measured heater-setpoint → piston-speed map |
| `magshockz/analysis/warpx/metrics.py` | front tracking + the invariant scorecard |
| `magshockz/analysis/warpx/flash.py` | the FLASH piston as MEASURED (fractional EOS Zbar) |

The spec block names are `meta:` / `flash:` / `reference:` / `plasma:` / `geometry:` /
`numerics:` / `operators:` / `calibration:` / `diagnostics:` / `runtime:` / `smoke:`.
`flash:` and `reference:`/`geometry:` stay distinct on purpose: this pipeline has **two
tiers of primaries** — what FLASH *measured* and what the reduced-mass bridge *chose* —
and merging them would erase the most important distinction in the file.

`units.FlashReference` and `analysis.warpx.MeasuredPiston` are the same distinction one
level down, and they are NOT interchangeable. `FlashReference` is what the deck
*imposes* — an integer charge state (`Al 6+`) named in the spec. `MeasuredPiston` is what
FLASH *is* — the EOS ionization state it actually ran at (Zbar 3.66 for this dataset),
which is what `flash_piston_profile.py` must report. Same formulary, same `Upstream`,
different ion; the deck's ambient `n_e` is 1.64x FLASH's as a result, by choice.

**The spec → deck mapping is one-way, and the deck round-trips back to prove it.**
`parse_inputs` → `resolve_constants` → `key_params` → `verify` resolves a deck's
`my_constants` in a restricted `eval` and diffs the resulting numbers against a freshly
rendered one, so the comparison is independent of formatting, comments, and whether a
length was written as `20.*de` or `2.*di`. Two uses: the generator self-verifies whatever
it writes, and `--verify` closes the loop against the `warpx_used_inputs` WarpX echoes
after the run. `rtol` is `1e-6`, not exact, because the WarpX build ships CODATA-2022
constants against `units.py`'s CODATA-2018 — a ~1e-9 shift in every resolved
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
python scripts/warpx_make_deck.py --config runs/magshockz_2d_heater.warpx.yaml --smoke
# 3. run it
sbatch init_warpx/run_heater_2d.sbatch
# 4. close the loop against what WarpX actually ran
python scripts/warpx_make_deck.py --config runs/magshockz_2d_heater.warpx.yaml --verify
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
