import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path
import vysxd_analysis
import vysxd_define

def make_joy_plot(path_to_data, y_spacing=1, alpha=0.4, 
                  cmap='viridis', gyrotime=625, figsize=(8, 5), dpi=100, xlabel=r'x $[c/\omega_p]$', ylabel=r'Field Value',
                  n_skip=1, time_labels=True, **kwargs):
    """
    Create a stacked spatial plot with colors changing over time (à la Joy Division album cover).
    
    Parameters:
    path_to_data : pathlib.Path
        Path to the data directory
    y_spacing : float
        Vertical spacing between lines
    alpha : float
        Transparency of fill
    cmap : str
        Colormap name for color progression
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Dots per inch for the figure
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    skip_steps : int
        Number of timesteps to skip between plotted lines (default=1, plots every timestep)
    **kwargs : dict
        Additional keyword arguments passed to plot function
    """
    # Load data
    data = vysxd_analysis.get_osiris_quantity_1d(path_to_data.as_posix() + '/', n_skip=n_skip)
    
    # Create the figure with limited DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create colormap
    colors = plt.get_cmap(cmap)
    for t in range(len(data[3])):
        y_offset = t * y_spacing
        points = np.array([data[4], y_offset+data[0][t]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(data[0][t].min(), data[0][t].max())
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(data[0][t])
        ax.add_collection(lc)

        # Fill below the line
        ax.fill_between(data[4], 
                       y_offset * np.ones_like(data[0][t]), 
                       data[0][t] + y_offset,
                       color=colors(t/len(data[3])),
                       alpha=alpha)

        if time_labels and t % 10 == 0:
            ax.text(data[4][-1], y_offset, f'       t={data[3][t]/gyrotime:.1f}', 
                    verticalalignment='center')
    
    # Customize the plot
    # plt.colorbar(lc, ax=ax, label=f'{ylabel}', location = 'left')
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    # ax.set_ylabel(ylabel)
    ax.autoscale()
    
    return fig, ax

# # Example usage
# path_to_data = Path('/home/dschneidinger/shock_reformation/raw_data/B_0.04_ufl_0.2_vi_0.00001_ve_0.0001.1d/MS/DENSITY/ions/charge')
# make_joy_plot(path_to_data, n_skip=2, time_labels=True, alpha=0.1, y_spacing=1, cmap='plasma', dpi=100)