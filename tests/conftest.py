"""Shared fixtures and lightweight H5Data stub for tests."""

import sys
import os

import numpy as np
import pytest

# Put tests/ before src/ so stub modules (osh5def, analysis_utils) shadow the
# real ones that pull in heavy optional dependencies (astropy, osiris, etc.).
_TESTS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
for _p in (_SRC_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Ensure tests/ is first so stubs win.
sys.path.remove(_TESTS_DIR)
sys.path.insert(0, _TESTS_DIR)


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
