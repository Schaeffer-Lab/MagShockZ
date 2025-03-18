import numpy as np
import matplotlib.pyplot as plt

def make_joy_plot(data, spatial_coords, time_values=None, y_spacing=1, alpha=0.4, 
                  cmap='viridis', figsize=(10, 8), xlabel='x [c/wp]', ylabel='Field Value',
                  skip_steps=1, **kwargs):
    """
    Create a stacked spatial plot with colors changing over time (à la Joy Division album cover).
    
    Parameters:
    data : 2D array-like
        Data array where each row represents a spatial series at different times
    spatial_coords : array-like
        x-axis values representing spatial coordinates
    time_values : array-like, optional
        Time values for labeling
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
    
    # Apply timestep skipping
    data = data[::skip_steps]
    if time_values is not None:
        time_values = time_values[::skip_steps]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(data)))
    
    # Plot each time slice with offset
    for i, (line, color) in enumerate(zip(data, colors)):
        y_offset = i * y_spacing
        ax.plot(spatial_coords, line + y_offset, color=color, lw=1)
        
        # Fill below the line
        ax.fill_between(spatial_coords, 
                       y_offset * np.ones_like(line), 
                       line + y_offset,
                       color=color,
                       alpha=alpha)
        
        # Add time labels if provided
        if time_values is not None:
            ax.text(spatial_coords[-1], y_offset, f't = {time_values[i]:.1f}', 
                   verticalalignment='center')
    
    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    norm = plt.Normalize(0, len(data))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, ax=ax, label='Time')
    
    return fig, ax
