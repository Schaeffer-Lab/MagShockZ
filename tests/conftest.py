"""Shared fixtures and lightweight H5Data stub for tests."""

import sys
import os

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(__file__)


def _stub_osiris_stack() -> None:
    """Stand in for the OSIRIS libraries, which are not pip-installable.

    magshockz.common.analysis_utils imports osiris_utils, osh5io and osh5def at
    module scope, and six library modules import analysis_utils, so without these
    the bulk of the suite cannot even be collected.  Only import-time symbols are
    provided; any test needing real behaviour requires the real stack.
    """
    import types

    sys.path.insert(0, _TESTS_DIR)  # tests/osh5def.py

    diagnostic = types.ModuleType("osiris_utils.data.diagnostic")
    for registry in ("OSIRIS_FLD", "OSIRIS_PHA", "OSIRIS_SPECIE_REPORTS", "OSIRIS_SPECIE_REP_UDIST"):
        setattr(diagnostic, registry, {})
    data = types.ModuleType("osiris_utils.data")
    data.diagnostic = diagnostic
    osiris_utils = types.ModuleType("osiris_utils")
    osiris_utils.data = data

    sys.modules.update({
        "osiris_utils": osiris_utils,
        "osiris_utils.data": data,
        "osiris_utils.data.diagnostic": diagnostic,
        "osh5io": types.ModuleType("osh5io"),
    })


# Prefer the real stack where it exists (Perlmutter's `analysis` env), so the tests
# exercise what the scripts actually run against; stub only where it does not (CI).
try:
    import osiris_utils  # noqa: F401
    import osh5io  # noqa: F401
    import osh5def  # noqa: F401
except ImportError:
    _stub_osiris_stack()


class FakeAxis:
    """Minimal axis descriptor matching the osh5def.DataAxis interface."""

    def __init__(self, name: str, arr: np.ndarray):
        self.name = name
        self.min = float(arr[0])
        self.max = float(arr[-1])
        self.size = len(arr)


class FakeH5Data(np.ndarray):
    """numpy ndarray subclass that mimics the osh5def.H5Data used by moments.py."""

    def __new__(cls, array, axes):
        obj = np.asarray(array).view(cls)
        obj.axes = axes
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.axes = getattr(obj, "axes", [])

    def has_axis(self, name: str) -> bool:
        return any(ax.name == name for ax in self.axes)


def make_phase_space(p_arr: np.ndarray, x_arr: np.ndarray, values: np.ndarray) -> FakeH5Data:
    """Build a 2-D (p, x) FakeH5Data from pre-computed values."""
    axes = [FakeAxis("p1", p_arr), FakeAxis("x1", x_arr)]
    return FakeH5Data(values, axes)


# ---------------------------------------------------------------------------
# Common grids
# ---------------------------------------------------------------------------

@pytest.fixture
def p_grid():
    return np.linspace(-5.0, 5.0, 401)


@pytest.fixture
def x_grid():
    return np.linspace(0.0, 100.0, 50)
