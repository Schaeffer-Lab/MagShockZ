import yt
from pathlib import Path
import os
import glob
import subprocess
import numpy as np

def add_temperature_ratio_field(ds):
    """Add a derived field for electron-to-ion temperature ratio Te/Ti"""
    def _te_ti_ratio(field, data):
        # Get electron and ion temperatures
        te = data["tele"]
        ti = data["tion"]
        # Avoid division by zero
        ratio = np.where(ti > 0, te / ti, 0.0)
        return ratio
    
    # Add the derived field if not already present
    if ("gas", "TeTi_ratio") not in ds.derived_field_list:
        ds.add_field(
            ("gas", "TeTi_ratio"),
            function=_te_ti_ratio,
            sampling_type="cell",
            units="dimensionless",
            display_name="$T_e/T_i$"
        )

def find_plot_files(directory):
    """Find all FLASH plot files in the specified directory"""
    plot_pattern = os.path.join(directory, "MagShockZ_hdf5_plt_cnt_*")
    plot_files = sorted(glob.glob(plot_pattern))
    return plot_files

def make_movie(data_dir, field, output_dir=None, slice_axis="z", 
                                  fps=10, vmin=None, vmax=None, cmap = 'jet'):
    """
    Create a movie of the specified field from FLASH simulation data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing FLASH plot files
    output_dir : str, optional
        Directory to save frames and movie (default: current directory)
    slice_axis : str, optional
        Axis perpendicular to slice plane ("x", "y", or "z", default: "z")
    fps : int, optional
        Frames per second for output movie (default: 10)
    vmin, vmax : float, optional
        Min/max values for colorbar range
    """
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    frames_dir = os.path.join(output_dir, f"{field}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Find all plot files
    plot_files = find_plot_files(data_dir)
    
    if not plot_files:
        print(f"No plot files found in {data_dir}")
        return
    
    print(f"Found {len(plot_files)} plot files")
    print(f"Creating temperature ratio frames in {frames_dir}")
    
    # # Process each plot file
    # for i, plot_file in enumerate(plot_files):
    #     print(f"Processing {i+1}/{len(plot_files)}: {os.path.basename(plot_file)}")
        
    #     # Load dataset
    #     ds = yt.load(plot_file)
    #     if field == 'TeTi_ratio':
    #         # # Add temperature ratio field
    #         add_temperature_ratio_field(ds)
        
    #     # Create slice plot
    #     slc = yt.SlicePlot(ds, slice_axis, ("gas", field))
        
    #     # Set colormap and range
    #     slc.set_cmap(("gas", field), cmap)
    #     if vmin is not None and vmax is not None:
    #         slc.set_zlim(("gas", field), vmin, vmax)
        
    #     # Add timestamp annotation
    #     slc.annotate_timestamp(corner="upper_left", draw_inset_box=True)
        
    #     # Save frame
    #     frame_name = f"{field}_{str(i).zfill(4)}.png"
    #     slc.save(os.path.join(frames_dir, frame_name))
    
    # Create movie using ffmpeg - try multiple encoders in order of preference
    print(f"\nCreating movie...")
    
    # List of encoder configurations to try (in order of preference)
    encoder_configs = [
        {
            "name": "mpeg4 (MPEG-4/MP4)",
            "path": os.path.join(output_dir, f"{field}_movie.mp4"),
            "cmd": [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", os.path.join(frames_dir, f"{field}_%04d.png"),
            ]
        },
    ]
    
    movie_created = False
    for config in encoder_configs:
        try:
            cmd = config["cmd"] + [config["path"]]
            print(f"Trying {config['name']}...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Movie created successfully: {config['path']}")
            print(f"  Frames saved in: {frames_dir}")
            movie_created = True
            break
        except subprocess.CalledProcessError as e:
            # Check if it's an encoder issue
            stderr_str = e.stderr if e.stderr else ""
            if "Unknown encoder" in stderr_str:
                continue  # Try next encoder
            else:
                print(f"  Error with {config['name']}: {e}")
                print(f"  stderr: {stderr_str[:500] if stderr_str else 'N/A'}")
                continue
        except FileNotFoundError:
            print(f"  ffmpeg not found. Please install ffmpeg to create the movie.")
            break
    
    if not movie_created:
        print(f"  Could not create movie with available encoders.")
        print(f"  Frames are still available in: {frames_dir}")
        print(f"  You can create a movie manually or use ImageMagick to create an animated GIF:")

def timeseries(field, data_dir="/pscratch/sd/d/dschnei/FLASH_3D_noshield", 
               output_base="/pscratch/sd/d/dschnei/MagShockZ/analysis_notebooks/FLASH"):
    """Legacy function for creating time series of standard fields"""
    os.makedirs(f"{output_base}/{field}_plots", exist_ok=True)
    plot_files = find_plot_files(data_dir)
    
    for i, plot_file in enumerate(plot_files):
        ds = yt.load(plot_file)
        slc = yt.SlicePlot(ds, "z", field)
        slc.set_cmap(field, "Reds")
        slc.annotate_timestamp(corner="upper_left", redshift=True, draw_inset_box=True)
        slc.save(f"{output_base}/{field}_plots/{field}_{str(i).zfill(4)}.png")

if __name__ == "__main__":
    # Specify the directory containing FLASH plot files
    data_directory = "/pscratch/sd/d/dschnei/FLASH_3D_noshield"
    output_directory = "/pscratch/sd/d/dschnei/MagShockZ/analysis_notebooks/FLASH"
    
    # Create temperature ratio movie
    print("Creating Te/Ti ratio movie...")
    make_movie(
        data_dir=data_directory,
        field="TeTi_ratio",
        output_dir=output_directory,
        slice_axis="z",  # Change to "x" or "y" for different slice orientation
        fps=10,  # Frames per second
        vmin=1e-2, vmax=1e2,  # Uncomment to set fixed colorbar range
    )
    
    # Optionally, still create the original time series
    # timeseries("density")
    # timeseries("magx")
    # timeseries("magy")
    # timeseries("magz")