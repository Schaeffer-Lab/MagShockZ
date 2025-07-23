## IMPORTS ##
from pathlib import Path
import yt
import numpy as np
import matplotlib.pyplot as plt
from fitting_functions import Ray 


def get_template_config(template_type, lineout, **kwargs):
    """
    Get configuration dictionary for different template types
    """
    # Base configurations for different simulation types
    templates = {
        "basic": {  # Fallback with reasonable defaults
            'xmax': [-64, 64], # try to get 6 ion inertial lengths in x direction. sqrt(380) ~ 20
            "nx_p": [int(128*8000/lineout.osiris_length[-1]), 8000], # Get about the same resolution in x and y
            "num_par_x": [3, 3],
            "ndump": "512",
            "dt": "0.118",
            "tmax": "40000",
            "node_number": [1, 2],
            "n_threads": 1,
            "tile_number": [32, 64],
            "boundary_x2": ["reflecting", "open"],
            "reports": '"charge"',
            "rep_udist": '',
            "ps_pmin": [-0.2, -0.2, -0.2],
            "ps_pmax": [0.2, 0.2, 0.2],
            "ps_np": [128, 1024, 128],
            "ps_nx": [256, 1024],
            "emf_reports": '"e1", "e2", "e3", "b1", "b2", "b3"',
            "ps_xmin_x1": lineout.osiris_length[0]
        }
    }
    
    # Get base config and override with kwargs
    config = templates.get(template_type, templates["basic"]).copy()
    config.update(kwargs)
    
    # Add physics-based calculations
    config["upstream_gyrotime"] = int(1836 * 27 / 13 / lineout.rqm_factor / (70_000 / lineout.normalizations['magx']))
    config["rqm_al"] = int(1836 * 27 / 13 / lineout.rqm_factor)
    config["rqm_si"] = int(1836 * 28 / 14 / lineout.rqm_factor)
    config["tmax"] = config["upstream_gyrotime"] * 10 # want to run for 10 upstream gyroperiods

    # # num_tiles must be a power of two and greater than n_cells_tot / 1024
    # n_cells_tot = config["nx_p"][0] * config["nx_p"][1]
    # num_tiles = 2 ** np.ceil(np.log2(n_cells_tot / 1024))
    # config["tile_number"] = [np.sqrt(int(num_tiles)), np.sqrt(int(num_tiles))]

    return config


def write_input_file(lineout: Ray, template_type="basic", **kwargs):
    """
    Write OSIRIS input file with different template configurations
    
    Parameters:
    - lineout: Ray object with fitted data
    - template_type: "basic", etc.
    - **kwargs: Override any default values
    """
    # Default configuration
    config = get_template_config(template_type, lineout, **kwargs)
    
    return f'''
!----------global simulation parameters----------
simulation
\u007b
 algorithm = "cuda",
\u007d


!--------the node configuration for this simulation--------
node_conf 
\u007b
 node_number = {config["node_number"][0]}, {config["node_number"][1]}, ! number of GPUs you are using
 n_threads = {config["n_threads"]}, ! number of threads per GPU
 tile_number = {config["tile_number"][0]}, {config["tile_number"][1]}, ! n_tiles_tot should be greater than n_cells_tot / 1024 and be a power of two. Refer to osiris cuda documentation
 if_periodic(1:2) = .true., .false.,
\u007d


!----------spatial grid----------
grid
\u007b
 nx_p = {config["nx_p"][0]}, {config["nx_p"][1]}, ! number of cells 
\u007d


!----------time step and global data dump timestep number----------
time_step
\u007b
 dt     = {config["dt"]}, ! time step in wpe^-1
 ndump  = {config["ndump"]}, ! number of time steps between data dumps,
\u007d


!----------restart information----------
restart
\u007b
  ndump_fac = -1,
  ndump_time = 3590, !once/hour
  if_restart = .false.,
  if_remold = .true.,
\u007d


!----------spatial limits ----------
space
\u007b
 xmin(1:2) = {config['xmax'][0]}, {int(lineout.osiris_length[0])},
 xmax(1:2) = {config['xmax'][1]}, {int(lineout.osiris_length[-1])},
\u007d


!----------time limits ----------
time
\u007b
 tmin = 0.0,
 tmax = {config["tmax"]}, ! Just to make your life easier, your upstream gyrotime for an rqm of {config["rqm_al"]} is {config["upstream_gyrotime"]}
\u007d

!----------field solver set up----------
el_mag_fld
\u007b
  type_init_b(1:3) = "math func", "math func", "math func",
  init_b_mfunc(1) = "{lineout.math_funcs['magx']}",
  init_b_mfunc(2) = "{lineout.math_funcs['magy']}",
  init_b_mfunc(3) = "{lineout.math_funcs['magz']}",
  type_init_e(1:3) = "math func", "math func", "math func",
  init_e_mfunc(1) = "{lineout.math_funcs['Ex']}",
  init_e_mfunc(2) = "{lineout.math_funcs['Ey']}",
  init_e_mfunc(3) = "{lineout.math_funcs['Ez']}", ! This is the most important component, can probably ignore others
\u007d

!----------boundary conditions for em-fields ----------
emf_bound
\u007b
 type(1:2,2) =  "{config["boundary_x2"][0]}", "{config["boundary_x2"][1]}", ! boundaries for x2
\u007d

!----------- electro-magnetic field diagnostics ---------
diag_emf
\u007b
 ndump_fac = 1,
 reports = 
   "b1", "b2", "b3",
   "e1", "e2", "e3",
\u007d

!----------number of particle species----------
particles
\u007b
  interpolation = "quadratic",
  num_species = 3,
\u007d

!----------information for electrons----------
species
\u007b
 name = "electrons",
 rqm=-1.0,
 num_par_x(1:2) = {config["num_par_x"][0]}, {config["num_par_x"][1]}, ! number of particles per cell in x and y directions
\u007d

!----------inital proper velocities - electrons-----------------
udist
\u007b
  use_spatial_uth = .true.,
  use_spatial_ufl = .true.,
  spatial_uth(1) = "{lineout.math_funcs['tele']}",
  spatial_uth(2) = "{lineout.math_funcs['tele']}",
  spatial_uth(3) = "{lineout.math_funcs['tele']}",

  spatial_ufl(1) = "{lineout.math_funcs['v_ex']}",
  spatial_ufl(2) = "{lineout.math_funcs['v_ey']}",
  spatial_ufl(3) = "{lineout.math_funcs['v_ez']}",
\u007d

!----------density profile for electrons----------
profile
\u007b
  profile_type = "math func",
  math_func_expr = "{lineout.math_funcs['edens']}",
\u007d

!----------boundary conditions for electrons----------
spe_bound
\u007b
 type(1:2,2) = "reflecting","open",
\u007d

!----------diagnostic for electrons----------
diag_species
\u007b
 ndump_fac = 1,
 reports = {config["reports"]},
 ndump_fac_pha = 1,
 ps_pmin(1:3) = {config["ps_pmin"][0]}, {config["ps_pmin"][1]}, {config["ps_pmin"][2]},
 ps_pmax(1:3) = {config["ps_pmax"][0]}, {config["ps_pmax"][1]}, {config["ps_pmax"][2]},
 ps_xmin(1:2) = {config['xmax'][0]}, {int(lineout.osiris_length[0])}, ! phase space covers the entire domain. change as needed
 ps_xmax(1:2) = {config['xmax'][1]}, {int(lineout.osiris_length[-1])},
 ps_np(1:3) = {config["ps_np"][0]}, {config["ps_np"][1]}, {config["ps_np"][2]},
 ps_nx(1:2) = {config["ps_nx"][0]}, {config["ps_nx"][1]},
 phasespaces = "p1x1", "p1x2", "p2x1", "p2x2",
\u007d
   
!----------information for Aluminum ions----------
species
\u007b
 name = "aluminum",
 rqm = {config["rqm_al"]},
 num_par_x(1:2) = {config["num_par_x"][0]}, {config["num_par_x"][1]}, ! number of particles per cell in x and y directions
\u007d

!----------inital proper velocities Aluminum ions-----------------
udist
\u007b
  use_spatial_uth = .true.,
  use_spatial_ufl = .true.,
  spatial_uth(1) = "{lineout.math_funcs['tion']}",
  spatial_uth(2) = "{lineout.math_funcs['tion']}",
  spatial_uth(3) = "{lineout.math_funcs['tion']}",

  spatial_ufl(1) = "{lineout.math_funcs['v_ix']}",
  spatial_ufl(2) = "{lineout.math_funcs['v_iy']}",
  spatial_ufl(3) = "{lineout.math_funcs['v_iz']}",
\u007d

!----------density profile for Aluminum ions----------
profile
\u007b
 profile_type = "math func",
 math_func_expr = "{lineout.math_funcs['aldens']}",
\u007d

!----------boundary conditions for Alumium ions----------
spe_bound
\u007b
 type(1:2,2) = "reflecting", "open",
\u007d

!----------diagnostic for Aluminum ions----------
diag_species
\u007b
 ndump_fac = 1,
 reports = {config["reports"]},
 ndump_fac_pha = 1,
 ps_pmin(1:3) = {config["ps_pmin"][0]}, {config["ps_pmin"][1]}, {config["ps_pmin"][2]}, 
 ps_pmax(1:3) = {config["ps_pmax"][0]}, {config["ps_pmax"][1]}, {config["ps_pmax"][2]},
 ps_xmin(1:2) = {config['xmax'][0]}, {int(lineout.osiris_length[0])},
 ps_xmax(1:2) = {config['xmax'][1]}, {int(lineout.osiris_length[-1])},
 ps_np(1:3) = {config["ps_np"][0]}, {config["ps_np"][1]}, {config["ps_np"][2]},
 ps_nx(1:2) = {config["ps_nx"][0]}, {config["ps_nx"][1]},
 phasespaces = "p1x1", "p1x2", "p2x1", "p2x2",
\u007d

!----------information for Silicon ions----------
species
\u007b
 name = "silicon",
 rqm = {config["rqm_si"]},
 num_par_x(1:2) = {config["num_par_x"][0]}, {config["num_par_x"][1]}, ! number of particles per cell in x and y directions
\u007d

!----------inital proper velocities Silicon ions-----------------
udist
\u007b
  use_spatial_uth = .true.,
  use_spatial_ufl = .true.,
  spatial_uth(1) = "{lineout.math_funcs['tion']}",
  spatial_uth(2) = "{lineout.math_funcs['tion']}",
  spatial_uth(3) = "{lineout.math_funcs['tion']}",

  spatial_ufl(1) = "{lineout.math_funcs['v_ix']}",
  spatial_ufl(2) = "{lineout.math_funcs['v_iy']}",
  spatial_ufl(3) = "{lineout.math_funcs['v_iz']}",
\u007d

!----------density profile for Silicon ions----------
profile
\u007b
  profile_type = "math func",
  math_func_expr = "{lineout.math_funcs['sidens']}",
\u007d

!----------boundary conditions for Silicon ions----------
spe_bound
\u007b
 type(1:2,2) = "reflecting","open",
\u007d

!----------diagnostic for Silicon ions----------
diag_species
\u007b
 ndump_fac = 1,
 reports = {config["reports"]},
 ndump_fac_pha = 1,
 ps_pmin(1:3) = {config["ps_pmin"][0]}, {config["ps_pmin"][1]}, {config["ps_pmin"][2]}, 
 ps_pmax(1:3) = {config["ps_pmax"][0]}, {config["ps_pmax"][1]}, {config["ps_pmax"][2]},
 ps_xmin(1:2) = {config['xmax'][0]}, {int(lineout.osiris_length[0])},
 ps_xmax(1:2) = {config['xmax'][1]}, {int(lineout.osiris_length[-1])},
 ps_np(1:3) = {config["ps_np"][0]}, {config["ps_np"][1]}, {config["ps_np"][2]},
 ps_nx(1:2) = {config["ps_nx"][0]}, {config["ps_nx"][1]},
 phasespaces = "p1x1", "p1x2", "p2x1", "p2x2",
\u007d
'''

def main(start_point, end_point, inputfile_name):
    """
    Main function to run the analysis.
    """

    ## Path to FLASH data. This data was chosen because it's generally a simple setup.
    ## We want to characterize the fundamental behavior of a pison expanding out into a magnetized background. 
    data_path = Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_06-2024/MAGON/MagShockZ_hdf5_chk_0005")

    # Use the plugin I built for yt to load in FLASH data and split up ion species.
    yt.enable_plugins()

    ds = yt.load_for_osiris(data_path, rqm_factor = 50)
    # Create a Ray object for the lineout
    lineout = Ray(ds, start_point, end_point)

    lineout.fit("magx", degree=6, fit_func="piecewise", plot=False)
    lineout.fit('magy', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('magz', degree=10, fit_func="piecewise", plot=False)

    lineout.fit('Ex', degree=5, fit_func="piecewise", plot=False)
    lineout.fit('Ey', degree=8, fit_func="piecewise", plot=False)
    lineout.fit('Ez', degree=10, fit_func="piecewise", plot=False)

    lineout.fit("sidens", degree=8, fit_func="piecewise", plot=False)
    lineout.fit("aldens", degree=8, fit_func="piecewise", plot=False)
    lineout.fit("edens", degree=8, fit_func="piecewise", plot=False)

    lineout.fit('v_ex', degree=5, fit_func="piecewise", plot=False)
    lineout.fit('v_ix', degree=15, fit_func="piecewise", plot=False)

    lineout.fit('v_iy', degree=15, fit_func="piecewise", plot=False)
    lineout.fit('v_ey', degree=15, fit_func="piecewise", plot=False)

    lineout.fit('v_iz', degree=8, fit_func="piecewise", plot=False)
    lineout.fit('v_ez', degree=8, fit_func="piecewise", plot=False)

    lineout.fit('tele', degree=5, fit_func="piecewise", plot=False)
    lineout.fit('tion', degree=5, fit_func="piecewise", plot=False)

    # Write the input file for OSIRIS
    input_file_content = write_input_file(lineout)
    with open(inputfile_name, 'w') as f:
        f.write(input_file_content)

if __name__ == "__main__":
    main(start_point = (0,0.05,0),end_point = (0,0.37,0), inputfile_name="testing_writeout.txt")