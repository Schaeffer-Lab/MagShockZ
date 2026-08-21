MagShockZ analyzes magnetized collisionless shock simulations for the Magnetized
Collisionless Shocks on Z (MagShockZ) experiment. Most work runs on NERSC Perlmutter.

## Environments

- **`analysis`** — OSIRIS and WarpX analysis (`scripts/`). Has `osh5` / pyVisOS and
  `osiris_utils`, plus `yt` + `unyt`, so the FLASH-side scripts run here too.

Single source of truth: the run spec

Code style: the code explains itself, comments explain the physics

**Units live in the quantity, not in the variable name.** 
Prefer **plasmapy's `formulary`** over hand-rolled plasma formulas
**Type-hint every function in `magshockz/`**

**Always ask the user before making any design or physics decision.**
