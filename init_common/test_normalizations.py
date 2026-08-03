"""Verification for init_common.normalizations.

Run in the osiris2 conda env (which has yt) with::

    python init_common/test_normalizations.py

The first check (conversion-constant consistency) is pure numpy and runs in any
env; the yt-based checks are skipped automatically if yt is unavailable.
"""

import numpy as np


def check_conversion_constants():
    """The two OSIRIS B-field conversions must agree.

    B[Gauss] = 5.681e-8 * B_osiris * omega_pe[rad/s]
             = 3.204e-3 * B_osiris * sqrt(n0[cm^-3])

    so 5.681e-8 * omega_pe must equal 3.204e-3 * sqrt(n0).  Compute omega_pe with
    CODATA constants (no yt needed) and confirm the identity holds for a range of
    densities.  This validates reference_check_lines() independently of yt.
    """
    # CODATA 2018, SI
    e = 1.602176634e-19      # C
    eps0 = 8.8541878128e-12  # F/m
    me = 9.1093837015e-31    # kg

    for n0_cc in [1e17, 5e18, 1e20]:
        n0_si = n0_cc * 1e6  # cm^-3 -> m^-3
        omega_pe = np.sqrt(n0_si * e**2 / (eps0 * me))  # rad/s
        via_omega = 5.681e-8 * omega_pe
        via_density = 3.204e-3 * np.sqrt(n0_cc)
        rel = abs(via_omega - via_density) / via_density
        assert rel < 2e-3, f"conversion mismatch at n0={n0_cc:.1e}: {rel:.2e}"
        print(f"  n0={n0_cc:.1e} cm^-3  omega_pe={omega_pe:.3e} rad/s  rel.diff={rel:.2e}  OK")


def check_compute_norms():
    """yt-based: compute_norms round-trips a known B field back to Gauss."""
    try:
        import yt  # noqa: F401
    except ImportError:
        print("  yt not available - skipping compute_norms check")
        return

    from normalizations import compute_norms, reference_check_lines

    norms = compute_norms(reference_density_cc=5e18, rqm_factor=100)
    print(f"  omega_pe = {norms.omega_pe:.3e}")
    print(f"  B_norm   = {norms.B:.3e}")
    print(f"  E_norm   = {norms.E:.3e}")
    print(f"  v_norm   = {norms.v:.3e}")

    # v_norm must be c / sqrt(rqm_factor)
    import yt as _yt
    expected_v = (_yt.units.speed_of_light / np.sqrt(100)).to('cm/s')
    assert np.isclose(float(norms.v.value), float(expected_v.value), rtol=1e-10)

    # round-trip: 1e5 Gauss -> OSIRIS -> Gauss
    for line in reference_check_lines(norms, B_gauss=1e5):
        print("   ", line)
    B_test = (1e5 * _yt.units.Gauss / norms.B).to(_yt.units.dimensionless)
    recovered = float((5.681e-8 * B_test * norms.omega_pe).value)
    assert np.isclose(recovered, 1e5, rtol=1e-2), recovered
    print("  compute_norms round-trip OK")


def check_run_params():
    """Pure-numpy: the shared helpers reproduce the old inline formulas."""
    from run_params import cfl_dt, ndump, estimate_particle_load

    dx = 0.3
    # cfl_dt vs old inline forms
    assert np.isclose(cfl_dt(dx, 1), dx * 0.95)
    assert np.isclose(cfl_dt(dx, 2), dx * 0.95 / np.sqrt(2.0))

    # ndump vs both old forms
    tmax, dt, ntot = 5000.0, cfl_dt(dx, 2), 512
    assert ndump(tmax, dt, ntot) == int(tmax / (ntot * dt))
    assert ndump(tmax, dt, ntot) == int(tmax / float(dt) / ntot)

    # particle load: 1D against FLASH_OSIRIS_define's old inline expression
    xext, ppc = 4000.0, 5
    n_cells_1d = xext / dx
    load1 = estimate_particle_load(n_cells_1d, ppc)
    old_np_1d = (xext / dx) * 3 * ppc
    old_bytes_1d = old_np_1d * 2 * 70
    assert np.isclose(load1.n_particles, old_np_1d)
    assert load1.n_gpus == np.ceil(old_bytes_1d / (40e9 * 0.8))
    assert load1.n_nodes == np.ceil(old_bytes_1d / (40e9 * 0.8) / 4)

    # particle load: 2D against the old inline expression
    yext = 3650.0
    n_cells_2d = xext * yext / dx**2
    load2 = estimate_particle_load(n_cells_2d, ppc**2)
    old_np_2d = xext * yext / dx**2 * 3 * ppc**2
    assert np.isclose(load2.n_particles, old_np_2d)

    print(f"  cfl_dt/ndump/particle-load reproduce old formulas; "
          f"2D load = {load2.n_particles:.3e} particles, "
          f"{load2.n_gpus:.0f} GPUs, {load2.n_nodes:.0f} nodes  OK")


def check_tiles():
    """Pure-numpy: tile_numbers matches the README shared-memory formula."""
    from run_params import max_tile_cells, tile_numbers

    # README formula, reproduced independently (the +1 guard-cell term).
    def readme_cap(dims, interp_order, shmem=163 * 1024, precision=8, frac=0.8):
        side = (shmem * frac / (2 * 3 * precision)) ** (1.0 / dims)
        return int(side - (2 * interp_order + 1))

    for dims, interp, order in [(1, "cubic", 3), (2, "cubic", 3),
                                (2, "linear", 1), (2, "quadratic", 2)]:
        cap = max_tile_cells(dims, interp)
        assert cap == readme_cap(dims, order), (dims, interp, cap, readme_cap(dims, order))

    # 2D cubic on a realistic grid: powers of two, every tile within the cap,
    # and the *smallest* such power (largest tiles).
    cap2d = max_tile_cells(2, "cubic")
    cells = [13333, 12166]
    tiles = tile_numbers(cells, 2, "cubic")
    for n, t in zip(cells, tiles):
        assert (t & (t - 1)) == 0, f"{t} not a power of two"
        assert n / t <= cap2d, f"tile {n/t:.1f} exceeds cap {cap2d}"
        if t > 1:
            assert n / (t // 2) > cap2d, "not the smallest valid power of two"
    print(f"  2D cubic cap={cap2d} cells/dim; cells {cells} -> tiles {tiles}  OK")

    # 1D: large cap means a short lineout fits in a single tile (old define
    # branch raised NameError here; old simplified used an unrelated heuristic).
    assert tile_numbers([2648], 1, "cubic") == [1]
    print(f"  1D cubic cap={max_tile_cells(1, 'cubic')} cells; 2648-cell lineout -> [1]  OK")


if __name__ == "__main__":
    print("conversion-constant consistency:")
    check_conversion_constants()
    print("run-parameter helpers:")
    check_run_params()
    print("tile numbers (README formula):")
    check_tiles()
    print("compute_norms (needs yt):")
    check_compute_norms()
    print("done")
