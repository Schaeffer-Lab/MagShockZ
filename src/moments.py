"""Velocity-space moments of an OSIRIS phase-space distribution.

``moment(data, order, axis)`` integrates f over one momentum axis:
    order 0 -> number density        n      = ∫ f dp
    order 1 -> mean (bulk) velocity  <p>    = ∫ p f dp / n
    order 2 -> velocity variance     <(p-<p>)^2>  (second central moment)

All integrals use Simpson's rule along the named axis. Orders share the
zeroth/first moment so a 2nd-order call integrates each quantity once.
"""

import numpy as np
import osh5def
import scipy.integrate


def moment(data: osh5def.H5Data, order: int, axis: str, debug: bool = False):
    if not data.has_axis(axis):
        raise ValueError(f"Data does not have axis '{axis}'")

    ax = next(i for i in range(len(data.axes)) if data.axes[i].name == axis)
    p_axis = np.linspace(data.axes[ax].min, data.axes[ax].max, data.axes[ax].size)
    if debug:
        print(f"Found axis '{axis}' at index {ax}, "
              f"range [{data.axes[ax].min}, {data.axes[ax].max}], size {data.axes[ax].size}")

    # Reshape the momentum coordinate to broadcast against `data` along `ax`
    # without allocating a full-size weights array.
    shape = [1] * data.ndim
    shape[ax] = -1
    weights = p_axis.reshape(shape)

    # Zeroth moment (number density) — needed to normalise the higher orders.
    m0 = scipy.integrate.simpson(data, p_axis, axis=ax)
    if order == 0:
        return m0

    # First moment (mean velocity), normalised by the zeroth.
    m1 = np.divide(
        scipy.integrate.simpson(data * weights, p_axis, axis=ax),
        m0, out=np.zeros_like(m0), where=m0 != 0,
    )
    if order == 1:
        return m1

    # Second central moment (variance) about the mean velocity.
    if order == 2:
        w = weights - np.expand_dims(m1, axis=ax)
        return np.divide(
            scipy.integrate.simpson(data * np.square(w), p_axis, axis=ax),
            m0, out=np.zeros_like(m0), where=m0 != 0,
        )

    raise ValueError(f"Unsupported moment order: {order} (expected 0, 1, or 2)")
