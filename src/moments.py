import osh5io
import osh5def
import scipy
import numpy as np

def moment(data: osh5def.H5Data, order: int, axis: str, debug: bool = False):
    if not data.has_axis(axis):
        raise ValueError(f"Data does not have axis '{axis}'")
    for ax in range(len(data.axes)):
        if data.axes[ax].name == axis:
            p_axis = np.linspace(data.axes[ax].min, data.axes[ax].max, data.axes[ax].size)
            if debug:
                print(f"Found axis '{axis}' at index {ax} with range [{data.axes[ax].min}, {data.axes[ax].max}] and size {data.axes[ax].size}")
            break

    # Keep one non-singleton axis so this broadcasts against `data` without allocating a full-size weights array.
    weights = np.copy(p_axis)
    shape = [1] * data.ndim
    shape[ax] = -1
    weights = weights.reshape(shape)

    if order == 0:
        if debug:
            print(f"p_axis: {p_axis}")
            print(f"data shape: {data.shape}, p_axis shape: {p_axis.shape}")
        return scipy.integrate.simpson(data, p_axis, axis = ax)
    elif order == 1:
        zeroth_moment = moment(data, order=0, axis=axis)
        return np.divide(scipy.integrate.simpson(data * weights, p_axis, axis = ax),
                        zeroth_moment, out=np.zeros_like(zeroth_moment), where=zeroth_moment!=0)
    elif order == 2:
        zeroth_moment = moment(data, order=0, axis=axis)
        first_moment = moment(data, order=1, axis=axis)
        first_moment_expanded = np.expand_dims(first_moment, axis=ax)
        w = weights - first_moment_expanded

        if debug:
            print(f"zeroth_moment shape: {zeroth_moment.shape}, first_moment shape: {first_moment.shape}")
            print(f"weights shape: {weights.shape}, data shape: {data.shape}")
            print(f"first_moment_expanded shape: {first_moment_expanded.shape}")
            w = weights - first_moment_expanded
        return np.divide(scipy.integrate.simpson(data * np.square(w), p_axis, axis = ax),
                        zeroth_moment, out=np.zeros_like(zeroth_moment), where=zeroth_moment!=0)

def write_moments_to_h5():
    pass

def test_zeroth_moment(file_path, axis = 'p1'):
    data = osh5io.read_h5(file_path)
    result = moment(data, order = 0, axis = axis, debug = True)
    # assert result.shape == data.shape[:data.axes.index(axis)] + data.shape[data.axes.index(axis)+1:]
    assert np.all(result >= 0), "Zeroth moment should be non-negative"

def test_first_moment(file_path, axis = 'p1'):
    data = osh5io.read_h5(file_path)
    result = moment(data, order = 1, axis = axis, debug = True)

def test_second_moment(file_path, axis = 'p1'):
    data = osh5io.read_h5(file_path)
    result = moment(data, order = 2, axis = axis, debug = True)
    assert np.all(result >= 0), "Second moment (variance) should be non-negative"
    return result

def test_write_moments_to_h5():
    pass

if __name__ == "__main__":
    filename = "/pscratch/sd/d/dschnei/perlmutter_1.1.1d/MS/PHA/p1x1/aluminum/p1x1-aluminum-000000.h5"
    test_zeroth_moment(filename)
    test_first_moment(filename)
    test_second_moment(filename)
    test_write_moments_to_h5(test_zeroth_moment(filename), test_first_moment(filename), test_second_moment(filename))

    filename = "/pscratch/sd/d/dschnei/perlmutter_2.8.2d/MS/PHA/p2x1x2/al/p2x1x2-al-000001.h5"
    test_zeroth_moment(filename, axis = 'p2')
    test_first_moment(filename, axis = 'p2')
    test_second_moment(filename, axis = 'p2')
