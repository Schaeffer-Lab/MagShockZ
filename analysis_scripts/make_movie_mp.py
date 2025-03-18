from matplotlib.animation import FuncAnimation
from IPython import display
from multiprocessing import Pool, cpu_count
import osh5io
import osh5vis
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate_frame(args):
    frame, path_to_data, vlimits, gyrotime_scale, width, height, dpi, gyrotime = args
    data = osh5io.read_h5(f'{path_to_data}-{frame:06d}.h5')
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    if len(np.shape(data)) == 1:
        osh5vis.osplot(data, ylim=[vlimits[0], vlimits[1]]) if vlimits else osh5vis.osplot(data)
    else:
        osh5vis.osplot(data, vmax=vlimits[1], vmin=vlimits[0]) if vlimits else osh5vis.osplot(data)
    if gyrotime_scale:
        plt.title(f"{data.run_attrs['NAME']} {round(data.run_attrs['TIME'][0]/gyrotime, 2)}" + r" $\omega_{ci}$")
    fig.canvas.draw()
    frame_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    return frame, frame_data

def movie(path_to_data, frames=True, vlimits=None, n_jobs=cpu_count(), gyrotime = 625, dpi=100):
    parent_dir = Path(path_to_data).parent
    frames = len(list(parent_dir.glob('*.h5'))) if frames is True else frames

    # Determine the size of the figure
    fig, ax = plt.subplots(dpi=dpi)
    width, height = fig.canvas.get_width_height()
    plt.close(fig)

    with Pool(n_jobs) as pool:
        results = pool.map(generate_frame, [(frame, path_to_data, vlimits, True, width, height, dpi, gyrotime) for frame in range(frames)])

    frame_data = dict(results)

    fig, ax = plt.subplots(dpi=dpi)
    def animate(frame):
        ax.clear()
        ax.imshow(frame_data[frame], aspect='auto')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return ax

    ani = FuncAnimation(fig, animate, frames=frames, interval=60)
    video = ani.to_html5_video()
    display.display(display.HTML(video))
    plt.close()