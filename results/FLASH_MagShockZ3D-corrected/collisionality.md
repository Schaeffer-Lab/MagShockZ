# Collisionality of the MagShockZ upstream — FLASH-derived

Companion to [upstream_parameters.md](upstream_parameters.md). Every input here is taken
from that document's upstream table (the corrected sqrt(4pi) `FLASH_MagShockZ3D-corrected`
dataset); nothing is re-measured from the dumps. The purpose is a single citable number for
the claim that the shock is **collisionless**.

## Method

The ion-ion slowing-down mean free path for a test ion at speed `v` through a like-ion
background (Gaussian units):

```
lambda_ii = m_i^2 v^4 / (4 pi Z^4 e^4 n_i lnLambda)
```

with `m_i = 27 u` (Al), `Z = Zbar` as measured by FLASH, `n_i` the FLASH ion density, and
`lnLambda` from plasmapy's `Coulomb_logarithm` evaluated at the same velocity.

The `v^4` scaling is the whole story, so **which velocity** matters more than anything else:

- **`lambda_th`** — evaluated at the ion thermal speed `v_ti = sqrt(2 T_i / m_i)`. This is the
  NRL thermal mfp and measures how collisional the *fluid* is.
- **`lambda_in`** — evaluated at the shock-frame inflow speed `v_inflow` from the shock table.
  This is the relevant one for a shock: an upstream ion entering the front must be stopped
  over the transition, so `v_inflow` is the test-particle velocity. It is also the convention
  in the HEDP collisionless-shock literature.

`lambda_ei` uses the same form with `m_e`, `v_te` and `Z^2`.

The comparison scale is the ion inertial length `d_i` (already tabulated), since a
perpendicular magnetized shock transitions over a few `d_i`. A second comparison against
`L = 1 mm`, the shocked-layer / diagnostic scale, is given for context.

## Results

Lengths in um, speeds in km/s.

| quantity | los00 5 ns | los00 9 ns | los00 12 ns | los45 5 ns | los45 9 ns | los45 12 ns |
|---|---|---|---|---|---|---|
| v_ti | 23.5 | 24.4 | 28.4 | 11.4 | 10.2 | 10.0 |
| v_inflow | 948 | 794 | 680 | 492 | 292 | 217 |
| lnLambda (thermal) | 3.49 | 3.88 | 4.37 | 0.80 | 0.57 | 0.53 |
| lnLambda (at v_inflow) | 10.19 | 10.15 | 10.03 | 7.64 | 6.58 | 5.99 |
| lambda_ii thermal | 0.074 | 0.136 | 0.254 | 0.054 | 0.070 | 0.075 |
| **lambda_ii at v_inflow** | 6.73e+04 | 5.80e+04 | 3.66e+04 | 2.49e+04 | 7.06e+03 | 2.81e+03 |
| lambda_ei thermal | 122 | 71.4 | 46.9 | 0.56 | 0.56 | 0.57 |
| d_i | 257.5 | 339.8 | 357.9 | 229.1 | 273.2 | 286.7 |
| rho_i (thermal) | 180.6 | 216.4 | 221.1 | 35.0 | 31.4 | 30.6 |
| **lambda_ii / d_i** | **262** | **171** | **102** | **109** | **26** | **9.8** |
| lambda_ii / 1 mm | 67 | 58 | 37 | 25 | 7.1 | 2.8 |
| lambda_th / d_i | 2.9e-04 | 4.0e-04 | 7.1e-04 | 2.4e-04 | 2.6e-04 | 2.6e-04 |

## The number to cite

At **9 ns**, the column the idealized-run inputs are already drawn from:

> **lambda_ii ~ 7 mm against a shock transition scale d_i ~ 0.27 mm, i.e. lambda_ii / d_i ~ 26**
> in the unperturbed upstream (`los45`), and ~170 in the laser channel (`los00`).

Across the whole dataset: **lambda_ii / d_i ~ 10-260**.

The sharpest single statement, because it uses both mfp rows:

> The thermal ion-ion mean free path is **sub-micron** (< 0.3 um), so the upstream behaves as a
> fluid — which is what justifies treating it with MHD. But the mean free path scales as v^4,
> and at the shock's own relative velocity it reaches **millimetres**, 10-260x the ion inertial
> length. Coulomb collisions cannot mediate the transition: the shock must be collisionless.

## Caveats

- **`lambda_in` is soft, because lambda ~ v^4.** Substituting the lab-frame downstream flow speed
  `v_shock (1 - 1/r)` for the shock-frame inflow changes `los00` by 5x (lambda/d_i falls to ~35)
  and `los45` barely at all (factor 0.6-1.2). Always state which velocity was used. The
  shock-frame inflow is used throughout here.

  | | los00 5 | los00 9 | los00 12 | los45 5 | los45 9 | los45 12 |
  |---|---|---|---|---|---|---|
  | v_dn (lab) [km/s] | 651 | 534 | 458 | 430 | 298 | 225 |
  | (v_dn / v_inflow)^4 | 0.22 | 0.20 | 0.21 | 0.59 | 1.08 | 1.16 |

- **`los45` at 12 ns is the weakest column** (lambda/d_i = 9.8, lambda/L = 2.8). The shock
  decelerates from 564 to 288 km/s across the window and collisionality catches up as v^4.
  Quote 9 ns for consistency with the rest of the analysis, or quote the range — do not quote
  the 5 ns value alone.
- **Electrons are not collisionless.** `lambda_ei` is 0.56 um in the ambient (47-122 um in the
  channel). The claim is about ions, which is what sets the shock structure, but the poster
  should not imply a fully collisionless plasma.
- **The thermal mfp on `los45` is indicative only.** The classical Coulomb logarithm comes out
  at 0.53-0.80 there, i.e. below the weak-coupling validity floor, so plasmapy raises a
  `CouplingWarning` and the tabulated `lambda_th` is computed with lnLambda clamped at 1. It is
  safe to describe as "sub-micron" and unsafe to quote to two digits.
- **Z^4 dominates the thermal mfp.** Al at Zbar = 13 gives Z^4 = 28561, which is why the thermal
  ion-ion mfp is so short. The channel and ambient rays differ by Zbar = 13 vs 5 as much as by
  density or temperature.
- These are **upstream** values. Downstream densities are 3-6x higher and temperatures higher
  still; the mfp there has not been tabulated.

## Status in the codebase

None of this is computed by the package. The nearest existing machinery is
`magshockz/analysis/warpx/spitzer_resistivity.py` (electron-ion Spitzer resistivity, for the
WarpX hybrid resistivity) and `magshockz/common/dimensionless_params.py::magnetic_reynolds`
(Rm built on it). The numbers above come from a standalone script and are **not regenerated**
when the FLASH analysis is re-run.

If they are to be maintained, the natural shape is a pure `ion_ion_mfp` / collisionality helper
in `magshockz/common/` with a unit test, plus a `lambda_ii / d_i` row emitted into
`upstream_parameters.md` alongside `d_i` and `1/w_ci` — that is a design change and has not
been made.
