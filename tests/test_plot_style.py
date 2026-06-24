"""Tests for the display-unit mapping in src/plot_style.py (DisplayUnits).

Only the pure dataclass core (numpy-only) is exercised here — value rescaling and
label/title strings.  The field-resolving ``build_units()`` path is IO (reads OSIRIS
dumps via analysis_utils) and is not part of the CI-pure layer, matching the repo
convention of testing the pure logic and leaving the orchestration to the scripts.
"""

import argparse

import numpy as np
import pytest

import plot_style
from plot_style import DisplayUnits, electron_units


def test_electron_units_is_identity():
    e = electron_units()
    assert e.units == "electron"
    # x/t rescaling is a no-op
    assert e.x(123.0) == pytest.approx(123.0)
    assert e.t(456.0) == pytest.approx(456.0)
    np.testing.assert_allclose(e.x(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
    # native OSIRIS unit labels
    assert e.length_label == r"c/\omega_{pe}"
    assert e.time_label == r"\omega_{pe}^{-1}"
    assert e.xlabel() == r"$x\ [c/\omega_{pe}]$"
    assert e.tlabel() == r"$t\ [\omega_{pe}^{-1}]$"


def test_ion_units_rescale_values():
    d = DisplayUnits("ion", length_factor=10.0, time_factor=500.0,
                     length_label="d_i", time_label="T_{ci}")
    # a length of 1000 c/ωpe is 100 d_i; a time of 1500 1/ωpe is 3 T_ci
    assert d.x(1000.0) == pytest.approx(100.0)
    assert d.t(1500.0) == pytest.approx(3.0)
    np.testing.assert_allclose(d.x(np.array([10.0, 20.0])), [1.0, 2.0])
    np.testing.assert_allclose(d.t(np.array([500.0, 1000.0])), [1.0, 2.0])


def test_ion_units_labels_and_title():
    d = DisplayUnits("ion", 10.0, 500.0, "d_i", "T_{ci}")
    assert d.xlabel() == r"$x\ [d_i]$"
    assert d.xlabel("x_\\mathrm{shock}") == r"$x_\mathrm{shock}\ [d_i]$"
    assert d.tlabel() == r"$t\ [T_{ci}]$"
    # time_title formats the rescaled time to two decimals with the display unit
    assert d.time_title(1605.0) == r"$t = 3.21\ T_{ci}$"


def test_add_units_arg_default_and_choice():
    parser = argparse.ArgumentParser()
    plot_style.add_units_arg(parser)
    assert parser.parse_args([]).units == "auto"
    assert parser.parse_args(["--units", "ion"]).units == "ion"
    assert parser.parse_args(["--units", "electron"]).units == "electron"
    with pytest.raises(SystemExit):
        parser.parse_args(["--units", "proton"])


def test_resolve_units_follows_publication():
    # auto follows the publication state recorded by apply(); explicit choices pass through.
    plot_style.apply(False)
    assert plot_style.resolve_units("auto") == "electron"
    assert plot_style.resolve_units("ion") == "ion"
    plot_style.apply(True)
    try:
        assert plot_style.resolve_units("auto") == "ion"
        assert plot_style.resolve_units("electron") == "electron"
    finally:
        plot_style.apply(False)  # reset global so other tests see screen defaults
