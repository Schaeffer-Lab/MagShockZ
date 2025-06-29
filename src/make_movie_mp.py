from matplotlib.animation import FuncAnimation
from IPython import display
from multiprocessing import Pool, cpu_count
import osh5io
import osh5vis
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate_frame(args):

    frame, path_to_data, vlimits, gyrotime_scale, dpi, gyrotime = args
    data = osh5io.read_h5(f'{path_to_data}-{frame:06d}.h5')
    fig, ax = plt.subplots(dpi=dpi)
    # Add this line to adjust subplot parameters
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    if len(np.shape(data)) == 1:
        osh5vis.osplot(data, ylim=[vlimits[0], vlimits[1]]) if vlimits else osh5vis.osplot(data)
    else:
        osh5vis.osplot(data, vmax=vlimits[1], vmin=vlimits[0]) if vlimits else osh5vis.osplot(data)
    
    if gyrotime_scale:
        plt.title(f"{data.run_attrs['NAME']} {np.round(data.run_attrs['TIME'][0]/gyrotime, 2)}" + r" $\omega_{ci}$")
    
    # Add this line to ensure tight layout
    fig.tight_layout()
    
    fig.canvas.draw()
    frame_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
    plt.close(fig)
    return frame, frame_data


def movie(path_to_data, frames=True, vlimits=None, n_jobs=cpu_count(), gyrotime = 625, dpi=100):
    parent_dir = Path(path_to_data).parent
    frames = len(list(parent_dir.glob('*.h5'))) if frames is True else frames


    with Pool(n_jobs) as pool:
        results = pool.map(generate_frame, [(frame, path_to_data, vlimits, True, dpi, gyrotime) for frame in range(frames)])

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

def phase_space_frame(args):
    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    # Create custom colormaps with alpha gradients
    cmap1 = pl.cm.RdPu
    my_cmap1 = ListedColormap(
        np.column_stack([
            cmap1(np.arange(cmap1.N))[:, :3],
            np.linspace(0, 1, cmap1.N)
        ])
    )

    cmap2 = pl.cm.Greens
    my_cmap2 = ListedColormap(
        np.column_stack([
            cmap2(np.arange(cmap2.N))[:, :3],
            np.linspace(0, 1, cmap2.N)
        ])
    )   
    frame, path_to_data, vlimits, gyrotime_scale, dpi, gyrotime = args
    
    fig, ax = plt.subplots(dpi=dpi)
    # Add this line to adjust subplot parameters
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Initialize data_0
    data_0 = None
   
    # Define a list of colormaps
    cmaps = [my_cmap1,my_cmap2]            # Use a different colormap for each index i
    for i in range(len(path_to_data)):
        if len(path_to_data) > 2:
            colorbar= False        
        else:
            colorbar= True
        if i == 0:
            piston = osh5io.read_h5(f'{path_to_data[i]}-{frame:06d}.h5')
            osh5vis.osplot(piston, ylim=[vlimits[0], vlimits[1]], colorbar=colorbar) if vlimits else osh5vis.osplot(piston)
            continue
        elif i > 0:
            if data_0:
                data_0 += osh5io.read_h5(f'{path_to_data[i]}-{frame:06d}.h5')
            else:
                data_0 = osh5io.read_h5(f'{path_to_data[i]}-{frame:06d}.h5')
        osh5vis.osplot(np.log(data_0), vmax=vlimits[1], vmin=vlimits[0], cmap=cmaps[i],colorbar=colorbar) 
        
  

    if gyrotime_scale:
        plt.title(f"{np.round(data_0.run_attrs['TIME'][0]/gyrotime, 2)}" + r" $\omega_{ci}$")
    
    # Add this line to ensure tight layout
    fig.tight_layout()
    
    fig.canvas.draw()
    frame_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
    plt.close(fig)
    return frame, frame_data


def phase_space_movie(path_to_data, frames=True, vlimits=None, n_jobs=cpu_count(), gyrotime = 625, dpi=300):
    parent_dir = Path(path_to_data[0]).parent
    frames = len(list(parent_dir.glob('*.h5'))) if frames is True else frames


    with Pool(n_jobs) as pool:
        results = pool.map(phase_space_frame, [(frame, path_to_data, vlimits, True, dpi, gyrotime) for frame in range(frames)])

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