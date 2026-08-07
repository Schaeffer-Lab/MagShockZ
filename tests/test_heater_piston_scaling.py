"""Tests for heater_piston_scaling: the FLASH -> reduced-mass WarpX deck mapping.

Numpy-only, so these run in CI without WarpX itself.  The contract under test
is the one the deck depends on: the dimensionless invariants listed in
``PistonTargets.invariants()`` come out of ``derive()`` unchanged, and everything else
is either derived consistently from them or flagged in ``warnings``.
"""

import numpy as np
import pytest

import heater_piston_scaling as hps


def make_targets(**overrides) -> hps.PistonTargets:
    """A representative FLASH target: MagShockZ-like ambient a few ns in (chamber
    plasma already rarefied), 7 T out-of-plane, ~900 km/s front."""
    fields = dict(
        n_amb_per_m3=3.0e23,
        b_amb_tesla=7.0,
        te_amb_ev=120.0,
        ti_amb_ev=80.0,
        n_piston_drive_per_m3=6.0e24,
        v_front_ms=9.0e5,
        l_piston_m=250.0e-6,
        r_spot_m=500.0e-6,
        t_window_s=9.0e-9,
    )
    fields.update(overrides)
    return hps.PistonTargets(**fields)


class TestElementary:
    def test_plasma_frequency_known_value(self):
        # omega_pe = 5.641e4 * sqrt(n_e[cm^-3]) rad/s is the standard formula.
        density_cm3 = 1.0e12
        expected = 5.641e4 * np.sqrt(density_cm3)
        assert hps.plasma_frequency_rad_s(density_cm3 * 1e6) == pytest.approx(
            expected, rel=2e-4)

    def test_skin_depth_is_c_over_omega_pe(self):
        density = 1.0e18
        assert hps.electron_skin_depth_m(density) == pytest.approx(
            hps.C_LIGHT_MS / hps.plasma_frequency_rad_s(density), rel=1e-12)

    def test_nonpositive_density_is_nan(self):
        assert np.isnan(hps.plasma_frequency_rad_s(0.0))
        assert np.isnan(hps.electron_skin_depth_m(-1.0))
        assert np.isnan(hps.ion_inertial_length_m(0.0, hps.M_P_KG))
        assert np.isnan(hps.alfven_speed_ms(1.0, 0.0, hps.M_P_KG))
        assert np.isnan(hps.ion_gyrofrequency_rad_s(1.0, 0.0))

    def test_alfven_speed_known_value(self):
        expected = 1.0 / np.sqrt(hps.MU_0 * 1.0e19 * hps.M_P_KG)
        assert hps.alfven_speed_ms(1.0, 1.0e19, hps.M_P_KG) == pytest.approx(
            expected, rel=1e-12)

    def test_theta_roundtrip(self):
        for temperature_ev, mass_kg in ((1.0, hps.M_E_KG), (5.0e3, hps.M_E_KG),
                                        (250.0, 100 * hps.M_E_KG)):
            theta = hps.theta_from_ev(temperature_ev, mass_kg)
            assert hps.ev_from_theta(theta, mass_kg) == pytest.approx(
                temperature_ev, rel=1e-12)

    def test_theta_is_one_at_the_electron_rest_mass(self):
        assert hps.theta_from_ev(510.999e3, hps.M_E_KG) == pytest.approx(1.0, rel=1e-4)

    def test_cfl_matches_the_two_validated_decks(self):
        density = 1.0e18
        d_e_m = hps.electron_skin_depth_m(density)
        omega_pe = hps.plasma_frequency_rad_s(density)
        # run_shock_1d: 1D, dx = 0.5 d_e, cfl 0.75 -> dt*omega_pe = 0.375
        dt_1d = hps.cfl_timestep_s(0.5 * d_e_m, 0.75, n_dims=1)
        assert dt_1d * omega_pe == pytest.approx(0.375, rel=1e-6)
        # run_flatfoil_compare: 2D, dx = 1 d_e, cfl 0.75 -> dt*omega_pe = 0.5303
        dt_2d = hps.cfl_timestep_s(1.0 * d_e_m, 0.75, n_dims=2)
        assert dt_2d * omega_pe == pytest.approx(0.5303, rel=1e-3)

    def test_plasma_beta_definition(self):
        density, temperature_ev, b_tesla = 1.0e20, 100.0, 0.5
        expected = (density * temperature_ev * hps.Q_E
                    / (b_tesla**2 / (2.0 * hps.MU_0)))
        assert hps.plasma_beta(density, temperature_ev, b_tesla) == pytest.approx(
            expected, rel=1e-12)
        assert hps.plasma_beta(density, temperature_ev, 0.0) == np.inf

    def test_ion_acoustic_speed_clamps_negative_temperatures(self):
        c_s = hps.ion_acoustic_speed_ms(-5.0, 100.0, hps.M_P_KG)
        assert np.isfinite(c_s) and c_s > 0.0


class TestPistonTargets:
    def test_derived_quantities_are_self_consistent(self):
        targets = make_targets()
        assert targets.n_i_amb_per_m3 == pytest.approx(
            targets.n_amb_per_m3 / targets.z_amb)
        assert targets.mach_alfven == pytest.approx(
            targets.v_front_ms / targets.v_alfven_ms)
        assert targets.v_fast_ms == pytest.approx(
            np.hypot(targets.v_alfven_ms, targets.c_s_ms))
        assert targets.mach_magnetosonic == pytest.approx(
            targets.v_front_ms / targets.v_fast_ms)
        assert targets.contrast == pytest.approx(
            targets.n_piston_drive_per_m3 / targets.n_amb_per_m3)
        assert targets.gyroperiod_s == pytest.approx(
            2 * np.pi / targets.omega_ci_rad_s)
        assert targets.t_window_gyro == pytest.approx(
            targets.t_window_s / targets.gyroperiod_s)

    def test_mach_numbers_land_in_the_expected_regime(self):
        # CLAUDE.md records M_A ~ 6-8.5 and beta ~ 2-6 for these runs; the
        # representative target must sit in a super-critical, moderate-beta regime or
        # the whole exercise is mis-parameterised.
        targets = make_targets()
        assert 2.0 < targets.mach_alfven < 30.0
        assert targets.mach_magnetosonic > 2.76      # ion-reflection threshold
        assert 0.05 < targets.beta_e < 50.0

    def test_magnetosonic_mach_never_exceeds_alfvenic(self):
        targets = make_targets()
        assert targets.mach_magnetosonic <= targets.mach_alfven

    def test_length_ratios_use_the_ambient_d_i(self):
        targets = make_targets()
        assert targets.l_piston_di == pytest.approx(
            targets.l_piston_m / targets.d_i_m)
        assert targets.r_spot_di == pytest.approx(targets.r_spot_m / targets.d_i_m)

    def test_is_frozen(self):
        targets = make_targets()
        with pytest.raises(Exception):
            targets.n_amb_per_m3 = 1.0

    def test_invariant_keys_match_the_deck_side(self):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18)
        assert set(targets.invariants()) == set(scaling.invariants())


class TestDeriveInvariants:
    @pytest.mark.parametrize("key", ["M_A", "M_ms", "beta_e", "beta_i",
                                     "contrast", "r_spot/d_i", "t_run/T_ci"])
    def test_invariant_is_preserved(self, key):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0)
        assert scaling.invariants()[key] == pytest.approx(
            targets.invariants()[key], rel=1e-9)

    def test_measured_piston_scale_length_is_not_among_the_invariants(self):
        # The slab half-width is the piston *reservoir* (FLASH's ablator), set by
        # slab_halfwidth_di, not by the measured density e-folding length. Claiming to
        # match l_piston_m would be false; invariance_report shows the gap instead.
        targets = make_targets()
        assert "L_piston/d_i" not in targets.invariants()
        scaling = hps.derive(targets, n0_per_m3=1.0e18, slab_halfwidth_di=2.0)
        assert scaling.slab_halfwidth_di == pytest.approx(2.0, rel=1e-9)
        assert "L_piston/d_i" in hps.invariance_report(scaling)

    @pytest.mark.parametrize("n0_per_m3", [1.0e16, 1.0e18, 1.0e20, 1.0e26])
    def test_invariants_are_independent_of_n0(self, n0_per_m3):
        targets = make_targets()
        reference = hps.derive(targets, n0_per_m3=1.0e18).invariants()
        got = hps.derive(targets, n0_per_m3=n0_per_m3).invariants()
        for key, value in reference.items():
            assert got[key] == pytest.approx(value, rel=1e-9), key

    @pytest.mark.parametrize("mass_ratio", [25.0, 100.0, 400.0])
    def test_invariants_are_independent_of_mass_ratio(self, mass_ratio):
        targets = make_targets()
        reference = hps.derive(targets, n0_per_m3=1.0e18,
                               mass_ratio=100.0).invariants()
        got = hps.derive(targets, n0_per_m3=1.0e18,
                         mass_ratio=mass_ratio).invariants()
        for key, value in reference.items():
            assert got[key] == pytest.approx(value, rel=1e-9), key

    def test_n0_only_rescales_the_reference_scales(self):
        low = hps.derive(make_targets(), n0_per_m3=1.0e18)
        high = hps.derive(make_targets(), n0_per_m3=1.0e20)
        assert high.d_e_m == pytest.approx(low.d_e_m / 10.0, rel=1e-9)
        assert high.omega_pe_rad_s == pytest.approx(
            low.omega_pe_rad_s * 10.0, rel=1e-9)
        assert (high.n_cells_x, high.n_cells_z) == (low.n_cells_x, low.n_cells_z)
        assert high.dt_omega_pe == pytest.approx(low.dt_omega_pe, rel=1e-9)
        assert high.max_step == low.max_step


class TestDeriveDrive:
    def test_theta_e_and_piston_speed_are_inverses(self):
        targets = make_targets()
        from_speed = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0,
                                v_piston_c=0.05, kappa=2.5)
        assert from_speed.v_piston_c == pytest.approx(0.05, rel=1e-12)
        from_theta = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0,
                                theta_e_heater=from_speed.theta_e_heater, kappa=2.5)
        assert from_theta.v_piston_c == pytest.approx(from_speed.v_piston_c, rel=1e-12)
        assert from_theta.theta_e_heater == pytest.approx(
            from_speed.theta_e_heater, rel=1e-12)

    def test_psc_flatfoil_values_reproduce_the_documented_sound_speed(self):
        # SHOCK_PLAN.md: theta_e = 0.04 at Mi/me = 100 gives c_s ~ 0.02 c.
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, mass_ratio=100.0,
                             theta_e_heater=0.04, kappa=2.5)
        assert scaling.c_s_piston_ms / hps.C_LIGHT_MS == pytest.approx(0.02, rel=1e-6)
        assert scaling.v_piston_c == pytest.approx(0.05, rel=1e-6)

    def test_theta_e_scales_with_mass_ratio_at_fixed_piston_speed(self):
        # c_s = sqrt(theta_e/M) c, so holding v_piston fixed needs theta_e ~ M.
        light = hps.derive(make_targets(), n0_per_m3=1.0e18, mass_ratio=100.0,
                           v_piston_c=0.05)
        heavy = hps.derive(make_targets(), n0_per_m3=1.0e18, mass_ratio=400.0,
                           v_piston_c=0.05)
        assert heavy.theta_e_heater == pytest.approx(
            4.0 * light.theta_e_heater, rel=1e-9)

    def test_tuning_theta_e_up_moves_b0_to_hold_mach_alfven(self):
        # The behaviour the tuning loop relies on: theta_e is the knob, M_A is the
        # invariant, so B0 follows.
        cool = hps.derive(make_targets(), n0_per_m3=1.0e18, theta_e_heater=0.02)
        hot = hps.derive(make_targets(), n0_per_m3=1.0e18, theta_e_heater=0.08)
        assert hot.v_piston_ms == pytest.approx(2.0 * cool.v_piston_ms, rel=1e-9)
        assert hot.b0_tesla == pytest.approx(2.0 * cool.b0_tesla, rel=1e-9)
        assert hot.mach_alfven == pytest.approx(cool.mach_alfven, rel=1e-9)

    def test_relativistic_piston_is_rejected_or_warned(self):
        targets = make_targets()
        with pytest.raises(ValueError, match="unphysical"):
            hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0,
                       theta_e_heater=1.0e4)
        scaling = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0,
                             v_piston_c=0.3)
        assert any("relativistic" in w for w in scaling.warnings)

    @pytest.mark.parametrize("kwargs", [
        {"n0_per_m3": 0.0},
        {"n0_per_m3": 1.0e18, "mass_ratio": 0.0},
        {"n0_per_m3": 1.0e18, "kappa": 0.0},
        {"n0_per_m3": 1.0e18, "v_piston_c": 1.5},
        {"n0_per_m3": 1.0e18, "theta_e_heater": 0.0},
    ])
    def test_bad_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            hps.derive(make_targets(), **kwargs)

    def test_zero_field_target_raises(self):
        targets = make_targets(b_amb_tesla=0.0)      # v_A = 0 -> M_A = inf
        with pytest.raises(ValueError, match="Alfven Mach"):
            hps.derive(targets, n0_per_m3=1.0e18)


class TestDeriveAmbient:
    def test_temperatures_reproduce_the_target_betas(self):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18)
        assert hps.plasma_beta(scaling.n0_per_m3, scaling.te_amb_ev,
                               scaling.b0_tesla) == pytest.approx(
                                   targets.beta_e, rel=1e-9)
        assert hps.plasma_beta(scaling.n0_per_m3, scaling.ti_amb_ev,
                               scaling.b0_tesla) == pytest.approx(
                                   targets.beta_i, rel=1e-9)

    def test_u_std_uses_the_species_mass(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, mass_ratio=100.0)
        assert scaling.theta_e_amb == pytest.approx(
            hps.theta_from_ev(scaling.te_amb_ev, hps.M_E_KG), rel=1e-12)
        assert scaling.theta_i_amb == pytest.approx(
            hps.theta_from_ev(scaling.ti_amb_ev, scaling.m_i_kg), rel=1e-12)
        # Same physical temperature -> ion theta smaller by the mass ratio.
        electron_theta_at_ion_mass = hps.theta_from_ev(scaling.te_amb_ev,
                                                      scaling.m_i_kg)
        assert electron_theta_at_ion_mass == pytest.approx(
            scaling.theta_e_amb / 100.0, rel=1e-12)

    def test_cold_piston_ions_are_colder_by_the_mass_ratio(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, mass_ratio=100.0,
                             theta_e_cold=1.0e-3)
        assert scaling.theta_e_cold == pytest.approx(1.0e-3)
        assert scaling.theta_i_cold == pytest.approx(1.0e-5)

    def test_target_density_follows_the_contrast(self):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18)
        assert scaling.n_target_per_m3 == pytest.approx(
            targets.contrast * 1.0e18, rel=1e-12)

    def test_inverted_contrast_warns(self):
        targets = make_targets(n_piston_drive_per_m3=1.0e23, n_amb_per_m3=3.0e23)
        scaling = hps.derive(targets, n0_per_m3=1.0e18)
        assert any("contrast" in w for w in scaling.warnings)

    def test_b0_reproduces_the_sim_alfven_speed(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, mass_ratio=100.0)
        assert hps.alfven_speed_ms(scaling.b0_tesla, scaling.n0_per_m3,
                                   scaling.m_i_kg) == pytest.approx(
                                       scaling.v_alfven_ms, rel=1e-12)

    def test_gyroperiod_follows_b0(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        omega_ci = hps.ion_gyrofrequency_rad_s(scaling.b0_tesla, scaling.m_i_kg)
        assert scaling.gyroperiod_s == pytest.approx(2 * np.pi / omega_ci, rel=1e-12)


class TestDeriveGeometryAndTime:
    def test_domain_z_is_sized_so_the_front_stays_inside(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        travel_de = scaling.v_piston_ms * scaling.t_run_s / scaling.d_e_m
        assert scaling.domain_z_halfwidth_de > travel_de + scaling.slab_halfwidth_de
        assert not any("wrap" in w for w in scaling.warnings)

    def test_too_small_domain_z_warns_about_wrapping(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18,
                             domain_z_halfwidth_de=10.0)
        assert any("wrap" in w for w in scaling.warnings)

    def test_cell_counts_match_the_domain(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, cell_size_de=0.5,
                             domain_x_halfwidth_de=128.0,
                             domain_z_halfwidth_de=768.0, blocking=8)
        assert (scaling.n_cells_x, scaling.n_cells_z) == (512, 3072)
        assert scaling.n_cells_x % 8 == 0 and scaling.n_cells_z % 8 == 0

    def test_cell_counts_round_up_to_the_blocking_factor(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, cell_size_de=1.0,
                             domain_x_halfwidth_de=17.0,
                             domain_z_halfwidth_de=17.0, blocking=8)
        assert (scaling.n_cells_x, scaling.n_cells_z) == (40, 40)   # ceil(34/8)*8

    def test_slab_and_spot_are_set_in_ion_units(self):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0,
                             slab_halfwidth_di=2.0)
        assert scaling.slab_halfwidth_de == pytest.approx(20.0)   # 2 d_i = 20 d_e
        assert scaling.spot_radius_de == pytest.approx(
            targets.r_spot_di * 10.0, rel=1e-9)

    def test_narrow_x_domain_warns_about_overlapping_spot_images(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18,
                             domain_x_halfwidth_de=1.0)
        assert any("images" in w for w in scaling.warnings)

    def test_default_x_domain_clears_the_spot_images(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        assert scaling.domain_x_halfwidth_de >= 4.0 * scaling.spot_radius_de

    def test_max_step_just_covers_the_requested_window(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        assert scaling.max_step * scaling.dt_s >= scaling.t_run_s
        assert (scaling.max_step - 1) * scaling.dt_s < scaling.t_run_s

    def test_run_window_override_shortens_the_run(self):
        targets = make_targets()
        default = hps.derive(targets, n0_per_m3=1.0e18)
        assert default.t_run_gyro == pytest.approx(targets.t_window_gyro, rel=1e-12)
        half = hps.derive(targets, n0_per_m3=1.0e18,
                          run_window_gyro=0.5 * targets.t_window_gyro)
        assert half.max_step < default.max_step
        assert half.t_run_s == pytest.approx(0.5 * default.t_run_s, rel=1e-9)

    def test_dt_is_the_2d_cfl_step(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, cell_size_de=0.5,
                             cfl=0.75)
        assert scaling.dt_omega_pe == pytest.approx(
            0.75 * 0.5 / np.sqrt(2.0), rel=1e-9)


class TestDebyeResolution:
    def test_debye_length_is_sqrt_theta(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        assert scaling.debye_length_de == pytest.approx(
            np.sqrt(scaling.theta_e_amb), rel=1e-12)
        assert scaling.debye_per_cell == pytest.approx(
            scaling.debye_length_de / scaling.cell_size_de, rel=1e-12)

    def test_cold_ambient_at_coarse_dx_warns(self):
        # The unperturbed-background case: low beta_e -> cold ambient -> small lambda_De.
        cold = make_targets(te_amb_ev=9.83, ti_amb_ev=9.83, b_amb_tesla=7.0,
                            n_amb_per_m3=3.04e24)
        scaling = hps.derive(cold, n0_per_m3=1.0e18, cell_size_de=0.5)
        assert scaling.debye_per_cell < hps.DEBYE_PER_CELL_MIN
        assert any("lambda_De/dx" in w for w in scaling.warnings)

    def test_refining_dx_clears_the_warning(self):
        cold = make_targets(te_amb_ev=9.83, ti_amb_ev=9.83, b_amb_tesla=7.0,
                            n_amb_per_m3=3.04e24)
        coarse = hps.derive(cold, n0_per_m3=1.0e18, cell_size_de=0.5)
        fine = hps.derive(cold, n0_per_m3=1.0e18, cell_size_de=0.2)
        # Halving dx doubles the margin; the suggested value in the warning must work.
        assert fine.debye_per_cell == pytest.approx(2.5 * coarse.debye_per_cell, rel=1e-9)
        assert fine.debye_per_cell >= hps.DEBYE_PER_CELL_MIN
        assert not any("lambda_De/dx" in w for w in fine.warnings)

    def test_warm_ambient_does_not_warn(self):
        warm = make_targets()          # beta_e ~ 2 -> warm enough at dx = 0.5
        scaling = hps.derive(warm, n0_per_m3=1.0e18, cell_size_de=0.5)
        assert scaling.debye_per_cell >= hps.DEBYE_PER_CELL_MIN
        assert not any("lambda_De/dx" in w for w in scaling.warnings)

    def test_margin_is_independent_of_n0(self):
        # lambda_De/dx is set by theta_e_amb and dx in d_e, both n0-independent.
        cold = make_targets(te_amb_ev=9.83, ti_amb_ev=9.83, b_amb_tesla=7.0,
                            n_amb_per_m3=3.04e24)
        a = hps.derive(cold, n0_per_m3=1.0e18, cell_size_de=0.2)
        b = hps.derive(cold, n0_per_m3=1.0e22, cell_size_de=0.2)
        assert a.debye_per_cell == pytest.approx(b.debye_per_cell, rel=1e-9)


class TestFlow:
    def test_zero_flow_is_the_default_and_silent(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        assert scaling.v_flow_ms == 0.0
        assert scaling.u_flow == 0.0
        assert scaling.e_motional_v_per_m == 0.0
        assert not any("Galilean" in w for w in scaling.warnings)

    def test_flow_sets_gamma_beta_and_the_motional_field(self):
        v_flow_ms = 1.0e6
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, v_flow_ms=v_flow_ms)
        beta = v_flow_ms / hps.C_LIGHT_MS
        assert scaling.u_flow == pytest.approx(beta / np.sqrt(1 - beta**2), rel=1e-12)
        assert scaling.e_motional_v_per_m == pytest.approx(
            -v_flow_ms * scaling.b0_tesla, rel=1e-12)
        assert abs(scaling.u_flow) < 1.0

    def test_flow_warns_that_the_injector_has_no_drift(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, v_flow_ms=1.0e6)
        assert any("TargetInjector" in w for w in scaling.warnings)

    def test_superluminal_flow_raises(self):
        with pytest.raises(ValueError, match="sub-luminal"):
            hps.derive(make_targets(), n0_per_m3=1.0e18, v_flow_ms=4.0e8)


class TestUnitBridges:
    def test_time_bridge_maps_one_gyroperiod_to_one_gyroperiod(self):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18)
        one_gyro_omega_pe = scaling.gyroperiod_s * scaling.omega_pe_rad_s
        assert scaling.to_ns(one_gyro_omega_pe) == pytest.approx(
            targets.gyroperiod_s * 1e9, rel=1e-9)
        assert scaling.to_ns(0.0) == pytest.approx(0.0)

    def test_length_bridge_maps_one_d_i_to_one_d_i(self):
        targets = make_targets()
        scaling = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0)
        one_d_i_in_de = scaling.d_i_m / scaling.d_e_m
        assert scaling.to_um(one_d_i_in_de) == pytest.approx(
            targets.d_i_m * 1e6, rel=1e-9)

    def test_bridges_are_linear_and_vectorised(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        lengths_de = np.array([0.0, 1.0, 2.0, 4.0])
        out = scaling.to_um(lengths_de)
        assert out.shape == lengths_de.shape
        assert out[3] == pytest.approx(4.0 * out[1], rel=1e-9)

    def test_bridges_are_independent_of_the_sim_reference_state(self):
        # The whole point: the same physical FLASH time/length comes back regardless
        # of n0 or the mass ratio the deck happens to use.
        targets = make_targets()
        a = hps.derive(targets, n0_per_m3=1.0e18, mass_ratio=100.0)
        b = hps.derive(targets, n0_per_m3=1.0e20, mass_ratio=400.0)
        assert a.to_ns(a.gyroperiod_s * a.omega_pe_rad_s) == pytest.approx(
            b.to_ns(b.gyroperiod_s * b.omega_pe_rad_s), rel=1e-9)
        assert a.to_um(a.d_i_m / a.d_e_m) == pytest.approx(
            b.to_um(b.d_i_m / b.d_e_m), rel=1e-9)

    def test_bridges_need_the_targets(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        scaling.targets = None
        with pytest.raises(ValueError, match="targets"):
            scaling.to_ns(1.0)
        with pytest.raises(ValueError, match="targets"):
            scaling.to_um(1.0)


class TestCostAndReport:
    def test_cost_report_is_self_consistent(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18, cell_size_de=0.5,
                             domain_x_halfwidth_de=128.0,
                             domain_z_halfwidth_de=768.0)
        cost = scaling.cost_report(ppc_each_dim=(5, 5), n_species=4)
        assert cost["cells"] == pytest.approx(512.0 * 3072.0)
        assert cost["ppc_per_species"] == 25.0
        assert cost["macroparticles"] == pytest.approx(512.0 * 3072.0 * 25.0 * 4)
        assert cost["steps"] == float(scaling.max_step)
        assert cost["node_hours"] > 0.0

    def test_cost_scales_linearly_in_particles(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        single = scaling.cost_report(ppc_each_dim=(5, 5))
        double = scaling.cost_report(ppc_each_dim=(10, 5))
        assert double["node_hours"] == pytest.approx(
            2.0 * single["node_hours"], rel=1e-9)

    def test_invariance_report_lists_both_halves(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        report = hps.invariance_report(scaling)
        assert "Matched dimensionless invariants" in report
        assert "Deliberately NOT matched" in report
        for key in ("M_A", "beta_e", "contrast", "m_i/m_e", "Z_ambient", "T_ci [s]"):
            assert key in report
        # A clean mapping flags nothing as OFF.
        assert "<-- OFF" not in report

    def test_invariance_report_flags_a_hand_broken_invariant(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        scaling.n_target_per_m3 *= 3.0
        assert "<-- OFF" in hps.invariance_report(scaling)

    def test_invariance_report_needs_targets(self):
        scaling = hps.derive(make_targets(), n0_per_m3=1.0e18)
        scaling.targets = None
        with pytest.raises(ValueError, match="targets"):
            hps.invariance_report(scaling)
