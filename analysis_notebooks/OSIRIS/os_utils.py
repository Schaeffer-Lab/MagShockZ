# Gotta take my own medicine. Going to make a goal that no functions are defined in jupyter notebooks.  
# Run file directly to test functions

from pathlib import Path
import numpy as np
import osh5vis
import osh5io
import matplotlib.pyplot as plt
import os
import subprocess
from multiprocessing import Pool, cpu_count
from IPython.display import HTML
import base64

def gather_diags(MS_dir: Path) -> dict:
    '''
    omnomnom give me a yummy MS directory and I will spit out a dictionary of diagnostics
    '''
    if not isinstance(MS_dir, Path):
        MS_dir = Path(MS_dir)
    diagnostics = {} 
    for item in MS_dir.rglob('*.h5'):
        parent_dir = item.parent.relative_to(MS_dir)
        if parent_dir not in diagnostics:
            diagnostics[str(parent_dir)] = item.parent.as_posix() + '/'

    if not diagnostics:
        print("No subdirectories found.")
        return

    return diagnostics

def time_series(diag_data_dir: Path) -> np.ndarray:
    '''
    stack up a time series of H5Data objects using osh5io.read_h5 given a directory of diagnostic data
    final shape is (time, x, y)
    '''
    result = []
    if not isinstance(diag_data_dir, Path):
        diag_data_dir = Path(diag_data_dir)
    data_sorted = sorted(list(diag_data_dir.glob('*.h5')))
    for f in data_sorted:
        result.append(osh5io.read_h5(f.as_posix()))
    return result

def streak_plot(time_series_data: list) -> np.ndarray:
    '''
    Given a time series of some data, make a streak plot by stacking the data in time
    Assumes that we want axes to be y vs t
    '''
    x_bnds = [time_series_data[0].run_attrs['SIMULATION']['XMIN'][-1], time_series_data[0].run_attrs['SIMULATION']['XMAX'][-1]]
    t_bnds = [time_series_data[0].run_attrs["TIME"][0], time_series_data[-1].run_attrs["TIME"][0]]

    plt.imshow(np.array(time_series_data), origin='lower', aspect='auto', extent=[x_bnds[0], x_bnds[-1], t_bnds[0], t_bnds[-1]])
    plt.ylabel(r'Time $\omega_p^{-1}$')
    plt.xlabel(r'x (c/$\omega_{p}$)')
    plt.colorbar(label=time_series_data[0].data_attrs['LONG_NAME'])


def get_temperature(phase_space_time_series_data: list, moment: int) -> np.ndarray:
    '''
    Get pressure from phase space data, right now this only works if integrating over x1. Output is NOT NORMALIZED!!!
    '''
    assert isinstance(moment, int)
    assert isinstance(phase_space_time_series_data[0], osh5io.H5Data)

    # for slice in phase_space_time_series_data:
        # v_n = 

def track_shock_front(data: np.ndarray) -> np.ndarray:
    '''
    Given a time series of some data, track the shock front position over time.
    Assumes data is a 3D numpy array with shape (time, x, y) and that shock is along y-axis
    '''
    shock_positions = []
    for frame in data:
        # Sum along x-axis to get a 1D profile
        profile = np.sum(frame, axis=1)
        # Find the position of the maximum gradient (shock front)
        gradient = np.gradient(profile)
        shock_pos = np.argmax(gradient)
        shock_positions.append(shock_pos)
    return np.array(shock_positions)

def _frame_generator(frame_number: int, data_at_t: osh5io.H5Data, vlimits: list) -> None:
    '''
    Not intended to be used directly, just a helper function for movie()
    Doesn't return anything directly, just saves a png to tmp/ folder
    '''
    plt.clf()
    if data_at_t.ndim == 1:
        osh5vis.osplot(data_at_t, ylim=vlimits)
    elif data_at_t.ndim == 2:
        osh5vis.osplot(data_at_t, vmin=vlimits[0], vmax=vlimits[1])
    plt.savefig(f'tmp/{frame_number:05d}.png')

def _save_movie(path_to_tmp, fps=40, x=1920, y=1080) -> str:
    '''
    Not intended to be used directly, just a helper function for movie()
    '''
    path_to_tmp = Path(path_to_tmp).resolve()
    png_pattern = str(path_to_tmp / '*.png')

    # Use ffmpeg to create the movie in a temporary file
    output_file = str(path_to_tmp / "../temp_movie.mp4")
    subprocess.call([
        "ffmpeg", "-loglevel", "quiet", "-framerate", str(fps), "-pattern_type", "glob", "-i", png_pattern,
        "-c:v", "libx264", "-vf", f"scale={x}:{y},format=yuv420p", output_file, "-y"
    ])

    return output_file

def movie(time_series_data, **kwargs):
    '''
    This is generally designed to be a one-size-fits-all movie maker for OSIRIS data.
    The expected format of time_series_data should be a list of H5Data objects, where the first dimension is time
    Additional keyword arguments can be passed to customize the movie, such as:
    '''
    kwargs_defaults = {
        'fps': 40,
        'x': 1920,
        'y': 1080,
        'vlimits': None,  # If None, will use min/max of middle frame
    }
    for key, value in kwargs_defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    if kwargs['vlimits'] is None:
        vlimits = (time_series_data[len(time_series_data)//2].min(), time_series_data[len(time_series_data)//2].max())
    else:
        vlimits = kwargs['vlimits']
    print(f"Using vlimits: {vlimits}")



    # print("\nWriting files...")
    Path('tmp').mkdir(exist_ok=True)
    if any(Path('tmp').iterdir()):
        print("There exists a non-empty 'tmp' directory. Exiting to prevent accidental deletion of data.")
        raise SystemExit
    # Run that ish in parallel so it goes FAST
    with Pool(cpu_count()) as pool:
        pool.starmap(_frame_generator, [(frame, time_series_data[frame], vlimits) for frame in range(len(time_series_data))])
    # if isinstance(time_series_data, list) and isinstance(time_series_data[0], osh5io.H5Data):
        # pool.starmap(_frame_generator, [(frame, time_series_data[frame], vlimits) for frame in range(len(time_series_data))])

    movie_path = _save_movie(path_to_tmp='tmp')
    # Read the movie file into memory and return bytes
    with open(movie_path, "rb") as f:
        movie_bytes = f.read()

    # Clean up temporary movie file
    os.unlink(movie_path)

    if Path('tmp').exists():
        for filename in os.listdir(Path('tmp')):
            file_path = os.path.join(Path('tmp'), filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.rmdir(Path('tmp'))

    # Encode movie bytes to base64 for HTML5 video embedding
    video_b64 = base64.b64encode(movie_bytes).decode('utf-8')
    video_html = f"""
    <video width="640" height="480" controls>
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    return HTML(video_html)

def test_gather_diags(MS: Path):
    diags = gather_diags(MS)
    assert isinstance(diags, dict)
    assert 'FLD/e1' in diags
    assert 'DENSITY/electrons/charge' in diags
    assert 'DENSITY/aluminum/charge' in diags
    assert 'DENSITY/silicon/charge' in diags
    print("gather_diags test passed.")

def test_time_series(test_diag_dir: Path):
    data = time_series(test_diag_dir)
    assert isinstance(data, list)
    assert len(data) == sum(1 for _ in Path(test_diag_dir).glob('*.h5'))
    print("time_series test passed.")

def test_track_shock_front(data: np.ndarray):
    shock_positions = track_shock_front(data)
    assert isinstance(shock_positions, np.ndarray)
    assert shock_positions.ndim == 1
    assert shock_positions.shape[0] == data.shape[0]
    print("track_shock_front test passed.")

def test_movie(data: np.ndarray):
    movie(data)
    print("movie function executed without error.")

def test_streak_plot(data: list):
    streak_plot(data)
    print("streak_plot function executed without error.")

if __name__ == "__main__":
    MS = '/home/dschneidinger/MagShockZ/magshockz-v2.21.2d/MS/'

    test_gather_diags(MS)
    test_gather_diags(Path(MS))
    diags = gather_diags(MS)
    test_time_series(diags['DENSITY/electrons/charge'])
    test_time_series(diags['FLD/e1'])
    # test_track_shock_front(time_series(diags['DENSITY/electrons/charge']))
    test_movie(time_series(diags['DENSITY/electrons/charge']))
    edens = time_series(diags['DENSITY/electrons/charge'])
    edens_savg = [np.mean(frame, axis=1) for frame in edens]  # Average along x-axis to get (time, density)
    test_streak_plot(edens_savg)