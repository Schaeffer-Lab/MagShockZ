# FLASH upstream parameters — MagShockZ (corrected sqrt(4pi) dataset)

Two lines of sight through `FLASH_MagShockZ3D-corrected`, chosen to isolate the
two upstream conditions the experiment will see:

- **`los00`** — the **laser-channel** case. Parallel to the axis at x = +500 um,
  so it samples the channel while staying close to normal to the shock front.
- **`los45`** — the **unperturbed-ambient** case, 45 deg off axis.

Upstream is a 600 um window starting 200 um ahead of the front. `rho/rho_0` is
measured against the t=0 chamber fill (3.73e-05 g/cm^3): los00 is heavily
processed by the channel, which is the point of the comparison, not a defect.
Every ray lies in z=0 with **B along z**, so all are perpendicular shocks
(theta_Bn = 90 deg) and directly comparable to perpendicular MHD theory.

## Upstream state

| quantity | los00 5 ns | los00 9 ns | los00 12 ns | los45 5 ns | los45 9 ns | los45 12 ns |
|---|---|---|---|---|---|---|
| rho / rho_0 | 0.149 | 0.086 | 0.081 | 0.911 | 0.883 | 0.865 |
| n_e [cm^-3] | 1.611e+18 | 9.255e+17 | 8.529e+17 | 4.480e+18 | 3.697e+18 | 3.489e+18 |
| n_ion [cm^-3] | 1.239e+17 | 7.123e+16 | 6.711e+16 | 7.581e+17 | 7.347e+17 | 7.203e+17 |
| Zbar | 13.00 | 12.99 | 12.71 | 5.91 | 5.03 | 4.84 |
| T_e [eV] | 373.5 | 209.2 | 156.9 | 18.0 | 14.6 | 13.9 |
| T_i [eV] | 77.3 | 83.6 | 112.6 | 18.1 | 14.6 | 13.9 |
| T_i / T_e | 0.207 | 0.400 | 0.717 | 1.005 | 0.997 | 1.000 |
| B [T] | 2.80 | 2.43 | 2.76 | 7.00 | 7.01 | 7.01 |
| B_perp [T] | 2.80 | 2.43 | 2.76 | 7.00 | 7.01 | 7.01 |
| theta_Bn [deg] | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 |
| beta_e | 30.950 | 13.243 | 7.081 | 0.663 | 0.443 | 0.396 |
| beta_i | 0.493 | 0.407 | 0.400 | 0.113 | 0.088 | 0.082 |
| beta (total) | 31.443 | 13.651 | 7.481 | 0.776 | 0.530 | 0.478 |
| v_A [km/s] | 33.5 | 38.3 | 44.9 | 33.9 | 34.5 | 34.8 |
| c_s [km/s] | 171.5 | 129.2 | 112.1 | 27.2 | 22.9 | 22.0 |
| v_ms [km/s] | 174.7 | 134.8 | 120.7 | 43.5 | 41.4 | 41.2 |
| d_i [um] | 257.5 | 339.8 | 357.9 | 229.1 | 273.2 | 286.7 |
| T_ci [ns] | 48.3 | 55.7 | 50.1 | 42.5 | 49.8 | 51.7 |

## Shock

| quantity | los00 5 ns | los00 9 ns | los00 12 ns | los45 5 ns | los45 9 ns | los45 12 ns |
|---|---|---|---|---|---|---|
| v_shock [km/s] | 948 | 791 | 678 | 564 | 363 | 288 |
| v_inflow [km/s] | 948 | 794 | 680 | 492 | 292 | 217 |
| M_s | 5.53 | 6.15 | 6.07 | 18.07 | 12.73 | 9.86 |
| M_A | 28.30 | 20.73 | 15.15 | 14.54 | 8.47 | 6.22 |
| M_ms | 5.43 | 5.89 | 5.63 | 11.33 | 7.05 | 5.26 |
| r measured (jump band) | 3.19 | 3.08 | 3.08 | 4.22 | 5.58 | 4.57 |
| r predicted (RH) | 3.62 | 3.67 | 3.64 | 3.88 | 3.71 | 3.51 |
| momentum flux dn/up | 0.98 | 0.95 | 0.98 | 0.97 | 0.95 | 0.93 |

M_ms > 1 on every column, so a shock exists in all cases; the ion-reflection
threshold is M_ms ~ 2.76, which every column also exceeds (super-critical).
`momentum flux dn/up` is the independent Rankine-Hugoniot check and should be 1.

## Inputs for the idealized reflecting-wall runs

Ready to paste into `idealizd-shocks-osiris/runs/*.yaml` under `physics:`
(`input_units: dimensionful`). Ambient is Al foam, so `ion_A: 27`; `ion_Z` is
the measured **Zbar**, not the atomic number. `v_flow` is the shock-frame
inflow. Values below are the **9 ns** column of each ray.

### `los00` — laser-channel upstream

```yaml
physics:
  n_e: 9.255e+17 cm**-3
  T_e: 209.2 eV
  T_i: 83.6 eV
  B: 2.43 T
  v_flow: 794 km/s
  ion_Z: 13.0
  ion_A: 27
  theta_deg: 90.0
```

Implies beta_e = 13.243, T_i/T_e = 0.400, M_A = 20.73.

### `los45` — unperturbed upstream

```yaml
physics:
  n_e: 3.697e+18 cm**-3
  T_e: 14.6 eV
  T_i: 14.6 eV
  B: 7.01 T
  v_flow: 292 km/s
  ion_Z: 5.0
  ion_A: 27
  theta_deg: 90.0
```

Implies beta_e = 0.443, T_i/T_e = 0.997, M_A = 8.47.


---

# Reasoning behind these numbers

## Why `los00` is not on axis

The on-axis ray originally ran radially from the laser spot at x = 0. Its momentum-flux
continuity — the independent Rankine-Hugoniot check, which must be 1 across a steady
front — degraded badly with time: 0.96 at 5 ns, 1.13 at 9 ns, **1.60 at 12 ns**.

The cause is geometric. By 9-12 ns the on-axis structure is a mushroom cap whose nose
has drifted to positive x, so a ray at x = 0 crosses the front obliquely, and the jump
conditions assume the ray lies along the shock normal. Scanning rays parallel to the
axis at 12 ns:

| x offset [um] | front [um] | v_shock [km/s] | dn/up |
|---|---|---|---|
| -1000 | 6816 | 339 | 4.33 |
| -500 | 8635 | 117 | 3.08 |
| -250 | 9187 | 594 | 1.86 |
| 0 (original) | 9506 | 629 | 1.60 |
| +250 | 9726 | 652 | 1.24 |
| **+500 (chosen)** | 9947 | 682 | 1.08 |
| +750 | 10077 | 704 | 1.01 |

The front position plateaus at +750 to +1000 um, which locates the nose. The same scan
in **z** is flat (front 9454-9519 um over z = -500..+750), so the displacement is in x
only. +500 um was chosen as the compromise: far enough that the front is near normal,
near enough that the ray still samples the laser channel, which is what makes this the
"channel" case at all.

After the shift, dn/up is **0.98 / 0.95 / 0.98** at 5 / 9 / 12 ns. The discrepancy is
resolved, and the shock speed is 682 km/s here against 629 km/s on axis — the nose
genuinely outruns the flank, so v_shock is a property of the ray, not of the blast.

**Caveat worth stating out loud:** at 12 ns this structure is a mushroom cap with
roll-up vortices, not a textbook planar shock. It now satisfies momentum-flux
continuity, which is real evidence it behaves as a steady front locally, but it should
be described as a plume front rather than implied to be a clean shock.

## Why the compression is measured over two different bands

The RH conditions are a *local* statement at the discontinuity, so they must be tested
against a thin band at the front. Over the full shocked layer they fail by
construction, because the layer's inner edge holds material shocked nanoseconds earlier
under a faster shock and a denser upstream. Measured on `los45` at 12 ns:

| downstream band width [um] | dn/up |
|---|---|
| 50 | 1.00 |
| 100 | 0.97 |
| 200 | 0.91 |
| 400 | 0.82 |
| 940 (to the piston contact) | 0.53 |

So two bands are used and reported separately: a **100 um jump band** at the front for
the RH check (compression, field jump, continuity), and the **full layer** from the
front back to the piston contact for downstream heating, Zbar and the electron/ion
partition — the latter being the plasma an experiment would actually diagnose.

## Why the measured compression exceeds 4

`r measured` reaches 4.2-5.6 on `los45`, above the gamma = 5/3 ceiling of 4 that an
ideal single-fluid shock cannot exceed. The shock is **ionizing**: Zbar roughly doubles
across the front, and energy that the adiabatic baseline would put into temperature
goes into ionization instead, which permits higher compression and lower downstream
temperature. An order-of-magnitude accounting with literature Al ionization potentials
puts the ionization cost at ~3400 eV/ion against ~3700 eV/ion of thermal energy gained,
i.e. **roughly half the post-shock energy budget**.

This is also why measured downstream temperatures fall well below the adiabatic RH
prediction. Note the prediction plotted in the figures is the **unmodified**
single-fluid `T2/T1 = (p2/p1)/(rho2/rho1)`. That form assumes a fixed mean molecular
weight, which an ionizing shock violates; the composition-corrected value
`x (1+Zbar_1)/(1+Zbar_2)` is reported alongside it as a diagnostic, roughly a factor of
two smaller, but is **not** applied — it is only half a correction, since `r` and
`p_ratio` still come from an energy equation with no ionization sink.

## What separates the two scenarios

The comparison is not mainly about density or temperature — it is about
**magnetization**:

- **beta_e differs by 30x** (13.2 in the channel against 0.44 in the ambient). The
  laser channel has made the field dynamically almost irrelevant on axis, while the
  off-axis case is properly magnetized at beta_e < 1.
- **T_i/T_e is 0.40 in the channel against 1.00 in the ambient**, so the two idealized
  runs differ in `temp_ratio` as well as `beta_e`.
- **M_ms is nearly the same** (5.9 against 7.1) despite M_A differing by 2.4x, because
  the channel's high sound speed compensates for its weak field. The two shocks are of
  comparable *strength* but very different *magnetization* — which is the cleanest way
  to frame the comparison.
- **d_i and T_ci come out similar** (340 vs 273 um, 56 vs 50 ns), so one box size and
  duration serves both idealized runs.

## Caveats on the tabulated values

- The idealized-run inputs are taken from the **9 ns** column. Both rays evolve
  strongly across the window — `los00`'s beta_e runs 31 -> 13.2 -> 7.5 and `los45`'s
  M_A runs 14.5 -> 8.5 -> 6.2 — so the choice of time materially changes the setup.
- `ion_Z` is the **measured Zbar** (13.0 and 5.0), not Al's atomic number of 13.
  `ion_A: 27` is right for both, since the ambient is Al foam in each case.
- `los00`'s upstream is flagged `PROCESSED` at every time (rho/rho_0 = 0.08-0.15). That
  is the intended physics of the channel case, not a measurement failure, but its Mach
  numbers are not comparable to the off-axis ray's without saying so.
- The 15 and 30 degree rays were analysed and then set aside. `los15` is unusable: a
  laser-channel column of ambient material crosses its upstream, and a rarefaction
  separates its piston from its shell. `los30` never reached momentum-flux
  conservation (dn/up ~ 1.9-2.4) for reasons not yet identified.
