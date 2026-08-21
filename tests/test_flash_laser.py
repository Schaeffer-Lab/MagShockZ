"""Tests for the FLASH laser deck parser and FLASH's inverse-bremsstrahlung closure."""

import numpy as np
import pytest

from magshockz.common import flash_utils as fu


DECK = """
# a comment
useEnergyDeposition = .true.
ed_gradOrder        = 2
ed_numberOfBeams = 1
ed_numberOfPulses = 1
ed_numberOfSections_1 = 5
ed_time_1_1  = 0.0
ed_time_1_2 = 0.05e-9
ed_time_1_3  = 0.1e-09
ed_time_1_4  = 1.85e-09
ed_time_1_5  = 1.90e-09
ed_power_1_1 = 0.0
ed_power_1_2 = 0.0
ed_power_1_3 = 1.e+12
ed_power_1_4 = 1.e+12
ed_power_1_5 = 0.0
ed_lensX_1                    =  0.0e-04
ed_lensY_1                    =  42000.0e-04
ed_lensZ_1                    =  0.0e-04
ed_targetX_1                  =  0.0e-04
ed_targetY_1                  =  0.0e-04
ed_targetZ_1                  =  0.0e-04
ed_targetSemiAxisMajor_1      =  800.0e-04
ed_pulseNumber_1              =  1
ed_wavelength_1               =  0.532
ed_gaussianExponent_1         =  2.0
ed_gaussianRadiusMajor_1      =  500.e-04
ed_numberOfRays_1             =  4096
ed_gridType_1                 = "square2D"
"""


@pytest.fixture
def par(tmp_path):
    p = tmp_path / "flash.par"
    p.write_text(DECK)
    return fu.parse_flash_par(str(p))


def test_parse_types_and_case(par):
    assert par["useenergydeposition"] is True
    assert par["ed_gradorder"] == 2.0
    assert par["ed_gridtype_1"] == "square2D"


def test_beam_geometry(par):
    beam = fu.laser_beams(par)[0]
    assert beam.lens == (0.0, 4.2, 0.0)
    assert beam.target == (0.0, 0.0, 0.0)
    # Pointing from the lens toward the target is -y.
    assert np.allclose(beam.axis, [0.0, -1.0, 0.0])
    assert beam.n_rays == 4096


def test_pulse_is_converted_to_cgs(par):
    beam = fu.laser_beams(par)[0]
    # The deck states Watts; everything downstream is CGS.
    assert beam.power_erg_s(1.0e-9) == pytest.approx(1.0e19)
    assert beam.power_erg_s(0.0) == 0.0
    assert beam.power_erg_s(3.0e-9) == 0.0
    # Flat 1 TW from 0.1 to 1.85 ns plus the two ramps.
    assert beam.energy_erg() == pytest.approx(1.8e10, rel=1e-3)


def test_critical_density(par):
    beam = fu.laser_beams(par)[0]
    assert beam.critical_density_cm3 == pytest.approx(3.939e21, rel=1e-3)


def test_coulomb_factor_matches_flash_expression():
    # ed_CoulombFactor.F90: lnLambda = log( (1.5/(Z e^3)) sqrt((kT)^3/(pi Ne)) ),
    # which for these units reduces to ln(1.549e10 * T_eV^1.5 / (Z sqrt(Ne))).
    t_eV, zbar, n_ele = 1040.0, 13.0, 1.076e19
    expected = np.log(1.549e10 * t_eV**1.5 / (zbar * np.sqrt(n_ele)))
    got = fu.flash_coulomb_factor(zbar, t_eV * 11604.518, n_ele)
    assert got == pytest.approx(expected, rel=2e-3)


def test_coulomb_factor_is_floored_at_one():
    # ed_CoulombFactor floors lnLambda at 1.0; a cold dense plasma would go below.
    assert fu.flash_coulomb_factor(13.0, 1.0, 1.0e24) == pytest.approx(1.0)


def test_ib_rate_reduces_to_the_standard_prefactor():
    # nu_ib = nu_ei * (Ne/Nc) with nu_ei = 2.905e-6 Z Ne lnLambda T_eV^-1.5.
    t_eV, zbar, n_ele, n_crit = 1040.0, 13.0, 1.076e19, 3.939e21
    ln_lambda = fu.flash_coulomb_factor(zbar, t_eV * 11604.518, n_ele)
    nu_ei = 2.905e-6 * zbar * n_ele * ln_lambda * t_eV**-1.5
    got = fu.flash_ib_rate(zbar, t_eV * 11604.518, n_ele, n_crit)
    assert got == pytest.approx(nu_ei * n_ele / n_crit, rel=2e-3)


def test_ib_opacity_is_the_rate_over_the_group_speed():
    t_K, zbar, n_ele, n_crit = 1040.0 * 11604.518, 13.0, 1.076e19, 3.939e21
    nu = fu.flash_ib_rate(zbar, t_K, n_ele, n_crit)
    kappa, _ = fu.flash_ib_opacity(zbar, t_K, n_ele, n_crit)
    v_group = 2.99792458e10 * np.sqrt(1.0 - n_ele / n_crit)
    assert kappa == pytest.approx(nu / v_group, rel=1e-6)


def test_ib_opacity_falls_as_the_ambient_is_heated():
    # kappa ~ T^-3/2 is what makes the laser channel bootstrap to transparency:
    # 10 eV Al foam has tau ~ 0.2 over 1.5 cm, 1 keV Al foam has tau ~ 0.01.
    n_crit = 3.939e21
    cold, _ = fu.flash_ib_opacity(3.7, 9.8 * 11604.518, 3.03e18, n_crit)
    hot, _ = fu.flash_ib_opacity(13.0, 1040.0 * 11604.518, 1.076e19, n_crit)
    assert cold * 1.5 == pytest.approx(0.2, rel=0.3)
    assert hot * 1.5 == pytest.approx(0.015, rel=0.3)
