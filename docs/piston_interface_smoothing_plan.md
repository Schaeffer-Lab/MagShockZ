# Physically-motivated smoothing of the piston/field-pileup interface

**Status: plan only — no code has been changed.**

## 1. The problem

The FLASH runs that seed our kinetic (OSIRIS / WarpX-hybrid) simulations were run as
**ideal MHD**: no explicit resistivity, and (by design of the unsplit-staggered-mesh
solver plus FLASH's interface-capturing) very little numerical mixing. In ideal MHD the
boundary between the laser-produced piston (Si) and the swept-up, compressed magnetic
field is a **contact discontinuity**, and nothing in the equations gives it a finite
width — its thickness in the dump is set by the grid, not by physics. Consequences we
observe at extraction time:

- a razor-sharp delineation between piston ions and the magnetic pile-up in front of
  them (zero interpenetration);
- a grid-scale current sheet (`J = ∇×B/μ0` spikes at the interface);
- piston profiles that fall off exponentially *until* a hard cut, rather than decaying
  smoothly into the upstream.

Handing this to a kinetic code is bad in a specific, mechanical way: WarpX's hybrid
Ohm's-law solve computes `E` from `J = ∇×B/μ0`, so a grid-scale `B` jump produces a
grid-scale `E` spike and launches spurious whistler/fast-wave transients at t=0; in
OSIRIS the equivalent problem is an initialized current/pressure imbalance the PIC noise
then rings against. The current fix — overwrite the pile-up with upstream values and
extrapolate the piston exponentially — removes the artifact but with **arbitrary**
choices (where to cut, what decay length). This document reviews what physics actually
sets that interface's structure and proposes a replacement in which every length scale
is computed, not chosen.

## 2. What physically sets the interface width

A laser-produced plasma expanding into a magnetized background is the classic
**diamagnetic-cavity / magnetized-piston** problem: the expanding plasma expels the
field, drives an azimuthal diamagnetic current at its surface, and piles the field up
in a compressed shell ahead of it (see the recent non-ideal-MHD study of exactly this
configuration: [Collimation of diamagnetic laser-driven plasma outflows](https://arxiv.org/pdf/2604.02704),
and the classic self-similar treatments going back to Raizer 1963
([modern analytic descendant](https://www.researchgate.net/publication/51951218_Self-similar_analytical_model_of_plasma_expansion_in_a_magnetic_field))).
Three distinct mechanisms give the cavity edge / pile-up a finite width, and they live
on very different scales:

### (a) Collisional (Spitzer) resistive diffusion — the fluid answer

With finite resistivity `η`, field slips through the plasma with magnetic diffusivity
`D_m = η/μ0`, and an initially sharp interface of age `t` acquires the diffusive width

```
δ_Sp(x) = sqrt( D_m(x) · t_age ),     D_m = η_Spitzer(T_e, Z̄) / μ0
```

where `t_age` is the time since the laser drive created the interface (i.e. the dump
time, minus any early time before the piston existed). This is the *only* smoothing an
MHD code could ever have produced, and it is fully computable from the dump: we already
have the machinery in `magshockz/analysis/warpx/spitzer_resistivity.py` / `scripts/warpx_spitzer_resistivity.py`
(Spitzer `η(T_e, Z̄)` maps over an extracted FLASH slice, built for choosing WarpX's
`plasma_resistivity`).

Scale feel (formulas, to be evaluated on our dumps in step 0): for
`T_e ~ 100 eV`, `Z̄ ~ 10`, `η ~ 5×10⁻⁶ Ω·m → D_m ~ 4 m²/s`, so over `t_age = 10 ns`,
`δ_Sp ~ 200 µm`; in the keV laser channel it drops to tens of µm. Whether this is
large or small compared to the kinetic scales below is *the* decision input — it
determines whether resistive physics alone fixes the problem (§4, decision point).

The corresponding dimensionless check is the **magnetic Reynolds number**
`Rm = v L / D_m` at the interface. If `Rm ≫ 1` there (hot piston surface), the honest
conclusion is that *the interface is physically sharp at MHD scales* and the width is
set by the kinetic mechanisms below — in which case rerunning FLASH with resistivity
would barely help, and the smoothing legitimately belongs in the extraction step.

### (b) Collisionless ion interpenetration (Larmor coupling) — the piston-ion answer

At kinetic level the piston ions are not fluid: they penetrate into (and through) the
compressed-field region ballistically over a **directed ion gyroradius** before being
magnetically deflected and coupled to the ambient plasma:

```
ρ_d = m_i v_exp / (Z_i e B_compressed)
```

This "Larmor coupling" picture is well established experimentally and numerically:
[Bondarenko et al., PoP 2017 (LAPD)](https://pubs.aip.org/aip/pop/article/24/8/082110/212269/Laboratory-study-of-collisionless-coupling-between),
[Winske & Gary 2007 hybrid simulations of debris–ambient coupling](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007JA012276),
[Le et al. 2021, JGR — scaling laws for debris cloud size and coupling](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JA029125)
([arXiv](https://arxiv.org/pdf/2109.00583)), and most recently
[Collisionless Larmor coupling and blob formation in a laser-plasma expanding into a magnetized ambient plasma (2026)](https://arxiv.org/pdf/2602.03494).
A companion semi-analytic model with explicit expressions for cavity size, compression
ratio, and boundary speeds exists in
[Collisionless coupling of a high-β expansion to an ambient magnetized plasma II](https://pubs.aip.org/aip/pop/article/25/4/042110/900664/Collisionless-coupling-of-a-high-expansion-to-an).

**This is the physical motivation the current "exponential extrapolation" is missing:**
the piston-ion density *should* decay into the pile-up/upstream with scale length
`~ ρ_d` (the directed gyroradius evaluated with the local expansion speed and the
compressed field), because that is how far debris ions physically interpenetrate before
coupling. The decay length stops being a free parameter.

### (c) Anomalous (lower-hybrid-drift) transport — the field-boundary answer

The steep density gradient at a diamagnetic cavity edge drives the **lower-hybrid drift
instability (LHDI)** whenever the gradient scale falls below roughly the hybrid
gyroradius `sqrt(ρ_e ρ_i)`; the resulting turbulence produces anomalous resistivity and
cross-field transport that saturate the boundary width at kinetic scales — this is the
same physics as at Earth's magnetopause and was directly measured in a laser-driven
magnetopause experiment
([Nature Comms: direct observations of anomalous resistivity and diffusion in collisionless plasma](https://pmc.ncbi.nlm.nih.gov/articles/PMC9135766/);
LHDI background: [LHDI at reconnecting current sheets](https://pubs.aip.org/aip/pop/article/32/6/062114/3350781/Anomalous-resistivity-and-electron-heating-by),
[drift turbulence at the magnetopause](https://pubs.aip.org/aip/pop/article/25/6/062103/320101/Drift-turbulence-particle-transport-and-anomalous)).
The AMPTE releases and LAPD cavities consistently show boundary widths of order
`sqrt(ρ_e ρ_i)` up to `ρ_i`.

**Key consequence:** the kinetic code will *regenerate* boundary structure on these
scales by itself (our PIC runs include LHDI physics; the hybrid run includes the
ion-scale part). The initialization does not need to guess the final boundary — it
needs to (i) not be sharper than the kinetic code can represent, (ii) conserve flux,
and (iii) not launch artificial blast waves. That reframes the task from "invent the
right profile" to "hand the kinetic code a resolvable, force-balanced state and let it
find its own boundary within a few gyroperiods."

### The scale hierarchy to compute (step 0 deliverable)

For each dump/extraction we should tabulate, at the interface:

| scale | meaning |
|---|---|
| `Δx_FLASH` | grid width of the discontinuity as dumped (expect: the measured width) |
| `δ_Sp = sqrt(η t_age/μ0)` | resistive width FLASH *would* have had (from `spitzer_resistivity`) |
| `ρ_d = m_i v_exp/(Z e B_comp)` | piston-ion interpenetration depth |
| `sqrt(ρ_e ρ_i)` … `ρ_i` (ambient) | LHDI-saturated boundary width |
| `d_i = c/ω_pi` | hybrid-code resolvable scale (cf. `flash_warpx.resolution`) |
| `Δx_WarpX / Δx_OSIRIS` | what the kinetic grid can actually hold |

The physically-correct initialization width is `max(δ_Sp, kinetic scale)`, and it must
also be `≳` a few kinetic-grid cells.

## 3. What the literature does about it (and what we can borrow)

1. **Run the MHD with real resistivity.** FLASH has a `MagneticResistivity` unit with a
   **SpitzerHighZ** implementation built for HEDP (Braginskii-based, parallel and
   perpendicular coefficients) — see the
   [FLASH user guide §23.2](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node146.html)
   and the Flash Center's own laser-experiment work which runs with explicit Spitzer
   resistivity ([Fatenejad et al. 2012](https://www.sciencedirect.com/science/article/abs/pii/S157418181200095X),
   [Tzeferacos et al. 2015](https://www.sciencedirect.com/science/article/abs/pii/S1574181814000779),
   and the [TDYNO turbulent-dynamo campaigns](https://arxiv.org/pdf/2007.12837), which
   state that diffusion coefficients are computed at runtime from collision models).
   So "FLASH without resistivity" is a choice we made, not a limitation of the code.

2. **Drive the kinetic code with a piston and skip the interface entirely.** Much of the
   laser-shock PIC literature does not initialize the MHD state at all: it initializes
   only the upstream and drives with a piston population or moving-wall boundary
   ([PIC simulations of collisionless perpendicular shocks driven at a laser-plasma device](https://pubs.aip.org/aip/adv/article/13/6/065302/2893762/Particle-in-cell-simulations-of-collisionless),
   [hybrid modeling of a magnetized super-critical laboratory shock](https://arxiv.org/pdf/2104.12170)).
   Robust, but discards the FLASH-derived piston structure that is the entire point of
   our pipeline; kept as a fallback/cross-check.

3. **Ion-scale magnetosphere / cavity studies** initialize a smooth driver and let the
   kinetic boundary form self-consistently within ~ an ion gyroperiod
   ([Laser-driven ion-scale magnetospheres I](https://arxiv.org/pdf/2201.02176),
   [II — PIC](https://arxiv.org/pdf/2201.02416),
   [detached bow shock off a magnetized obstacle](https://arxiv.org/pdf/2201.03520)).
   This supports the §2c claim that the init only needs to be *resolvable and balanced*,
   not exact.

## 4. Proposed plan

### Step 0 — Diagnose (no new physics, ~a day)

- Measure the interface width in the FLASH dumps as a function of AMR level and dump
  index (line-outs through the LOS; `tune_flash_shock.py --mode regions` already gives
  the geometry). Confirm it is grid-limited (width ∝ Δx).
- Build the scale table of §2 for the production dump(s): evaluate `η_Spitzer` maps with
  the existing `scripts/warpx_spitzer_resistivity.py`, compute `δ_Sp`, `ρ_d`, `sqrt(ρ_e ρ_i)`,
  `ρ_i`, `d_i`, and the local `Rm`.
- **Decision point:** if `δ_Sp ≳ ρ_i` at the interface, resistive physics dominates and
  Option B (FLASH rerun) is worth its cost; if `δ_Sp ≪` kinetic scales (`Rm ≫ 1`),
  the interface is physically sharp at fluid level and Option A below is not merely a
  workaround — it is the *correct* place to impose kinetic-scale structure.

### Option A (recommended): physics-based reprocessing at extraction time

Replace the ad-hoc overwrite/extrapolation with a post-extract step (a pure function in
`src/`, unit-tested per house rules; wired as an optional block in the `*.warpx.yaml`
config and applied to the extracted SI `.npy` tree so FLASH dumps stay untouched):

1. **Field: resistive diffusion, not overwrite.** Evolve the extracted `B` with the
   diffusion the run should have had:
   `∂B/∂t = ∇·(D_m ∇B)` with spatially varying Spitzer `D_m(T_e, Z̄)` for a duration
   `t_age` (the dump time). One physically-fixed parameter, zero free ones. In 2D/3D
   diffuse the **vector potential** (or apply a divergence-clean after) so `∇·B = 0`
   survives spatially-varying `D_m`. If step 0 shows `δ_Sp` under-resolves the kinetic
   grid, apply an **anomalous floor**: extend the diffusion with a uniform
   `D_anom` chosen so the final width equals the LHDI scale `sqrt(ρ_e ρ_i)`–`ρ_i`
   (report both `δ_Sp` and the floor in the provenance metadata — this is exactly the
   anomalous transport the real boundary undergoes, cf. §2c).
2. **Piston ions: gyroradius-scale interpenetration, not arbitrary exponential.** Decay
   the piston (Si/targ) species density beyond its fluid edge with scale length
   `ρ_d = m_i v_exp/(Z e B)` evaluated locally (using the *diffused* `B` and the piston
   edge velocity from the dump), carrying the piston bulk velocity with it. Same
   functional form as today's extrapolation — but the length is now computed from
   Larmor-coupling physics (§2b) instead of chosen.
3. **Re-balance.** After modifying `B` and densities, restore smooth **total pressure**
   `P_e + P_i + B²/2μ0` across the modified region (adjust the electron/ion pressure or
   the ambient density within the overlap zone) so no artificial blast wave is launched
   at t=0. Check the implied `J = ∇×B/μ0` is resolved over several cells.
4. **Provenance.** Freeze every computed scale (`t_age`, `δ_Sp` map summary, `ρ_d`,
   floor used, cells-per-width) into the extraction tree's `meta.yaml` alongside the
   existing metadata, so any run can be audited later.

Why this ordering of effort: it keeps the single-source-of-truth pipeline intact, costs
no FLASH rerun, has exactly one semi-free parameter (the anomalous floor, and even that
is bounded by literature to a factor ~`sqrt(ρ_i/ρ_e)`), and is verifiable cheaply in 1D.

### Option B (medium-term, physical gold standard at fluid level): rerun FLASH with `MagneticResistivity/SpitzerHighZ`

- Enable the unit ([FLASH UG §23.2](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node146.html));
  the USM solver adds resistive fluxes; coefficients come from the Spitzer/Braginskii
  model at runtime.
- Costs/caveats: full 3D rerun; the explicit resistive step imposes
  `Δt ≲ Δx²/(2·D_m)`, which can be punishing in the cold dense target where `D_m` is
  largest — check the timestep on a 2D test first. And per the step-0 decision point,
  if `Rm ≫ 1` at the interface this rerun reproduces our sharp interface anyway.
- Best use even if Option A is adopted: a **coarse 2D resistive rerun as validation**
  that the Option A diffusion step reproduces what FLASH-with-resistivity would give.

### Option C (cross-check): analytic magnetized-piston reconstruction

Fit the semi-analytic cavity/compression model of
[the collisionless-coupling papers](https://pubs.aip.org/aip/pop/article/25/4/042110/900664/Collisionless-coupling-of-a-high-expansion-to-an)
(cavity radius, compression ratio, shell speed from drive energy `E_d`, `B_0`,
ambient density) to the FLASH state away from the interface, and use it to sanity-check
the reprocessed profiles (bubble radius `R_B = (3μ0 E_d / 2πB_0²)^{1/3}`, equal-mass
radius, expected compression). Not an initialization path by itself.

### Option D (fallback): upstream-only kinetic run with a piston driver

Standard in the field (§3.2); sacrifices the FLASH piston fidelity. Keep in reserve if
reprocessed-init runs still show initialization-dominated transients.

## 5. Validation plan (applies to whichever option is implemented)

1. **1D A/B test:** extract the same FLASH slice with (i) current ad-hoc extrapolation,
   (ii) Option A reprocessing; run short WarpX-hybrid pairs. Metrics: t=0→few-gyroperiod
   field-energy transient, spurious wave launch from the interface, interface width vs
   time (does the kinetic run relax to the same `~ρ_i` boundary from both inits?),
   shock formation time and downstream RH jump.
2. **Insensitivity check (the real success criterion):** vary the anomalous floor by
   ×2 up/down; the post-formation shock (speed, compression, heating partition) should
   be insensitive. If it is not, the boundary physics matters at shock scale and
   Option B/D must be escalated.
3. **Resistive-FLASH cross-check:** coarse 2D SpitzerHighZ rerun vs Option A diffusion
   of the ideal run at the same time — interface widths should agree where `δ_Sp` is
   resolved.
4. **∇·B and force-balance audits** on the reprocessed trees (extend
   `flash_warpx.resolution`-style checks: max `|∇·B|`, cells-per-`J`-layer, total
   pressure smoothness).

## 6. Key references

- [Collisionless Larmor coupling and blob formation in a laser-plasma expanding into a magnetized ambient plasma (2026)](https://arxiv.org/pdf/2602.03494)
- [Bondarenko et al., Laboratory study of collisionless coupling between explosive debris plasma and magnetized ambient plasma, PoP (LAPD)](https://pubs.aip.org/aip/pop/article/24/8/082110/212269/Laboratory-study-of-collisionless-coupling-between)
- [Collisionless coupling of a high-β expansion to an ambient magnetized plasma II: semi-analytic cavity/compression model, PoP](https://pubs.aip.org/aip/pop/article/25/4/042110/900664/Collisionless-coupling-of-a-high-expansion-to-an)
- [Winske & Gary, Hybrid simulations of debris–ambient ion interactions in astrophysical explosions, JGR 2007](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007JA012276)
- [Le et al., Astrophysical explosions revisited: collisionless coupling of debris to magnetized plasma, JGR 2021](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JA029125) ([arXiv](https://arxiv.org/pdf/2109.00583))
- [Collimation of diamagnetic laser-driven plasma outflows by an ambient magnetic-pressure gradient (non-ideal MHD of the cavity+shell)](https://arxiv.org/pdf/2604.02704)
- [Self-similar analytical model of plasma expansion in a magnetic field](https://www.researchgate.net/publication/51951218_Self-similar_analytical_model_of_plasma_expansion_in_a_magnetic_field) (Raizer-lineage)
- [Direct observations of anomalous resistivity and diffusion in collisionless plasma (laser-driven magnetopause, Nature Comms)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9135766/)
- [Anomalous resistivity and electron heating by lower hybrid drift waves inside reconnecting current sheets, PoP](https://pubs.aip.org/aip/pop/article/32/6/062114/3350781/Anomalous-resistivity-and-electron-heating-by)
- [Drift turbulence, particle transport, and anomalous dissipation at the reconnecting magnetopause, PoP](https://pubs.aip.org/aip/pop/article/25/6/062103/320101/Drift-turbulence-particle-transport-and-anomalous)
- [FLASH user guide §23.2 Magnetic Resistivity (SpitzerHighZ)](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node146.html)
- [Fatenejad et al., FLASH MHD simulations of shock-generated magnetic field experiments (HEDP 2012)](https://www.sciencedirect.com/science/article/abs/pii/S157418181200095X); [Tzeferacos et al. (HEDP 2015)](https://www.sciencedirect.com/science/article/abs/pii/S1574181814000779); [time-resolved turbulent dynamo (TDYNO, resistive-FLASH usage)](https://arxiv.org/pdf/2007.12837)
- [Schaeffer-style upstream+piston PIC of laser-driven perpendicular shocks, AIP Advances 2023](https://pubs.aip.org/aip/adv/article/13/6/065302/2893762/Particle-in-cell-simulations-of-collisionless)
- [Hybrid modeling of a magnetized super-critical laboratory collisionless shock](https://arxiv.org/pdf/2104.12170)
- [Laser-driven ion-scale magnetospheres I (experiment)](https://arxiv.org/pdf/2201.02176), [II (PIC)](https://arxiv.org/pdf/2201.02416); [detached bow shock off a magnetized obstacle](https://arxiv.org/pdf/2201.03520)

Classic background (paywalled/older; cited for lineage): Raizer, *Deceleration and
energy conversions of a plasma expanding in a vacuum with a magnetic field* (1963);
Wright (1971) magnetized-piston theory; Dimonte & Wiley, PRL **67**, 1755 (1991)
(sub-Alfvénic cavity scaling); the AMPTE barium-release cavity literature.
