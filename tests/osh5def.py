"""Minimal osh5def stub for test environments that lack the full OSIRIS library.

Only the symbols used by src/moments.py are defined here.  The real H5Data
is a numpy subclass; our FakeH5Data in conftest.py replaces it for all tests.
"""

import numpy as np


class H5Data(np.ndarray):
    """Bare-minimum H5Data so ``isinstance(x, osh5def.H5Data)`` passes if needed."""

    def __new__(cls, array, axes=None):
        obj = np.asarray(array).view(cls)
        obj.axes = axes or []
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.axes = getattr(obj, "axes", [])

    def has_axis(self, name: str) -> bool:
        return any(ax.name == name for ax in self.axes)
