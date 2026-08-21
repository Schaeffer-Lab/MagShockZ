# FLASH side: the laser ray-trace / deposition audit

Why this exists: the FLASH runs show strong heating and ionization of the ambient Al
foam as the beam crosses it on its way to the target, and we needed to know whether
that is real inverse-bremsstrahlung absorption or an artifact of the ray trace.

## What the audit found (2026-08-21, `FLASH_MagShockZ3D-corrected` + the
`INCORRECT_FIELD_FLASH_3D_noshield` checkpoint, whose deck differs only in
`sim_MagField` and whose ray trace therefore is the same run)

It is real. FLASH's own inverse-bremsstrahlung closure, integrated over the beam
footprint through the same checkpoint being audited, predicts 4.1 % of the incident
beam absorbed before the target; summing ρ·`depo`·dV over the ambient in that
checkpoint measures 3.5 %. They agree to ~20 %, i.e. within the sampling error of the
comparison. The optical depth in the ambient is τ ≈ 0.21 at t = 0 (Tₑ = 9.8 eV,
Z̄ = 3.7), so ~19 % is absorbed on the first pass; the channel then strips to Z̄ = 13
at ~1 keV and κ ∝ T^−3/2 drops τ to 0.009 by 1.75 ns. 4 % of 1 TW into the ~1e-6 g of
ambient inside the channel is ~580 eV, against the ~1 keV observed.

Two things are *not* settled by that agreement, and are what the scan targets:

- **13 % of the beam leaves the domain.** FLASH already records this, every step, in
  `<basenm>_LaserEnergyProfile.dat` (written unconditionally by `ed_printEnergyStamp`).
  That file was never copied into the MagShockZ run directories on scratch; it is
  present for the 2D OmegaShock run, where it reads 9.2 %.
- **44 % of the ambient deposition lands outside the 800 µm beam footprint**, out to
  r = 1.2 cm, in speckled discrete ray tracks — a reflected/refracted halo off the
  ablation plume. Ray count and `ed_gradOrder` move this, not the absorbed total.
  Meanwhile `refine_var` is `dens` + `magz` only, so nothing refines the beam channel:
  on the beam axis in the ambient the mesh runs `lrefine` 2–6, and the 500 µm Gaussian
  radius is as little as 2.8 cells wide.

## Contents

- `MagShockZ2D/` — the 2D (x-y) Simulation unit, ported from the CFS tree's
  `OmegaShock` setup with the MagShockZ materials, layering and field. Copy it into
  `source/Simulation/SimulationMain/MagShockZ2D` of a FLASH tree
  (`/global/cfs/cdirs/m5032/OmegaShock/FLASH4.8` is the one on Perlmutter) and set up:

      ./setup -auto MagShockZ2D -2d -nxb=16 -nyb=16 +cartesian +hdf5typeio \
              species=cham,targ,vac +mtmmmt +laser +usm3t +mgd mgd_meshgroups=1 \
              -maxblocks=500 -site perlmutter.nersc.gov

  It includes an `IO_writeIntegralQuantities.F90` override that appends six columns to
  the `.dat` file: laser energy deposited per step, and cumulatively, in each of
  `targ` / `cham` / `vac`. Comparing the cumulative sum with `Energy in` from
  `LaserEnergyProfile.dat` closes the budget live, with no post-processing.

- `make_laser_scan.py` — reduces a MagShockZ **3D** `flash.par` to 2D (the 3D deck
  stays the source of truth for every physical number) and emits one deck per scan
  point plus a Slurm script. Legs: `ed_numberOfRays_1` × 1/4, ×4, ×16;
  `ed_gradOrder` 1/2/3; and beam-channel refinement (`refine_var_3 = "tele"`,
  optionally `+ "depo"`, and an `lrefine_min` floor). Every deck turns on
  `ed_useLaserIO` and `ed_saveOutOfDomainRays` and adds `depo`/`lase` to `plot_var`,
  which the production deck omits — which is why no 3D plot file has a deposition
  field at all.

      python flash/make_laser_scan.py \
          --par-3d /pscratch/sd/d/dschnei/FLASH_MagShockZ3D-corrected/flash.par \
          --output-dir runs/flash_laser_scan

## Analysing the output

`scripts/flash_laser_audit.py` runs all four checks on any FLASH run directory:

    python scripts/flash_laser_audit.py --run-dir <run> [--checks energy deposition tau mesh]

The `deposition` check needs a checkpoint (or a plot file from one of the scan decks,
which do carry `depo`); `tau` and `mesh` work on any plot file.
