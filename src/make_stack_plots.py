import numpy as np
import matplotlib.pyplot as plt
import vysxd_analysis
import vysxd_define
from matplotlib.collections import LineCollection
from pathlib import Path
from matplotlib.colors import Normalize


def make_joy_plot(path_to_data, y_spacing=1, alpha=0.4, 
                  cmap='viridis', gyrotime = 625, figsize=(4, 3), xlabel='x [c/wp]', ylabel='Field Value',
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
    # Ensure consistent path format
    data = vysxd_analysis.get_osiris_quantity_1d(path_to_data.as_posix() + '/', n_skip=n_skip)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap
    colors = plt.get_cmap(cmap)
    # Plot each time slice with offset
    for t in range(len(data[3])):
        y_offset = t * y_spacing
        # ax.plot(data[4], data[0][i] + y_offset, color=colors(i/len(data[3])), lw=1)
        # Create a line collection
        # Create points for the line segments
        points = np.array([data[4], data[0][t]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection

        norm = Normalize(data[0][t].min(), data[0][t].max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(data[0][t])

        # Create the plot
        fig, ax = plt.subplots()
        ax.add_collection(lc)

        # Fill below the line
        # ax.fill_between(data[4], 
        #                y_offset * np.ones_like(data[0][i]), 
        #                data[0][i] + y_offset,
        #                color=colors((data[0][i]/data[0].max())),
        #                alpha=alpha)
        
        # Add time labels if provided
        if time_labels:
            ax.text(data[4][-1], y_offset, f't = {data[3][t]/gyrotime:.1f}', 
                    verticalalignment='center')
    
    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # norm = plt.Normalize(0, len(data))
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, ax=ax, label='Time')
    
    return fig, ax

# Ensure the path to ion charge density files is correct
# path_to_data = Path('/home/dschneidinger/shock_reformation/raw_data/B_0.04_ufl_0.2_vi_0.00001_ve_0.0001.1d/MS/DENSITY/ions/charge')
# make_joy_plot(path_to_data,n_skip=10, time_labels=True,alpha=0.1,y_spacing=0.4,cmap='plasma')