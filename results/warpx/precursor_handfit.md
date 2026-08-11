Fast-ion precursor, hand-marked — surface-target theta scan (Z = 14)
====================================================================

NOT a calibration input. The calibration: block in runs/magshockz_2d_heater.warpx.yaml
measures the BULK piston (banded piston electron density crossing 1.0x the initial
ambient), because that is what FLASH's M_A was built from and what carries the momentum
driving the shock. This file records a different, genuinely kinetic feature that FLASH
cannot produce at all: a suprathermal population streaming ahead of the dense plume.

Marked by eye on every dump of each run (22 marks each), on the banded piston density
line-out, at the outermost visible extent of piston material.

  theta   precursor     bulk piston      shock       precursor/bulk
  0.035     0.03017        0.02296     0.03406           1.31
  0.055     0.03559        0.02701     0.03766           1.32
  0.085     0.04542        0.03213     0.05221           1.41

  All speeds in units of c, all fitted over the SAME 0.03-0.06 T_ci window, so they are
  comparable to each other and to calibration:. The shock is the ambient compression peak
  ahead of the contact. Fitting the precursor marks over all 22 dumps instead gives
  0.03132/0.03751/0.04522 -- higher, because the earlier dumps include the slab still
  accelerating out of its initial condition.

Two things worth keeping from this:

1. The precursor sits between the piston and the shock, and tracks the SHOCK's scaling
   rather than the piston's: across the three runs it stays at 1.31-1.41x the bulk piston
   while the shock/piston ratio runs 1.39-1.62. It is consistent with a shock-reflected
   or shock-accelerated population rather than a ballistic ablation tail, but the
   phase-space dumps (diags/phase*, 2% of particles, (z, u_z)) are what would settle it
   and have not been examined.

2. The marked edge is NOT a reproducible rule, which is why it cannot be a calibration
   criterion even in principle. Inverting the marks to the quantity a rule would hold
   fixed gives nothing constant: the piston density at the mark varies 2.5x across the
   three runs and drifts ~6x within each one (theta = 0.035: enclosed areal quantile
   0.0030 early to 0.0005 late). Every mark lands at n_piston ~ 1e-3 to 1e-2 n_0, i.e.
   at the macroparticle noise floor — a detectability threshold set by ppc and by the
   plotted dynamic range, not a physical front. Re-run at different ppc and it moves.

A shock has separated from the piston by the end of these runs (3.4 d_i apart at
theta = 0.085, compression ratio ~2.1), so the three features — contact, precursor,
shock — are all distinct and separately measurable at 0.06 T_ci.

Marks are in the session scratchpad as picks.json; the underlying banded profiles are
scan_surface_*_profiles.npz (z_di, t_gyro, n_over_n0).
