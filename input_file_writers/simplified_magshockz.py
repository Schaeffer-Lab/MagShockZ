## IMPORTS ##
from pathlib import Path
import yt
import numpy as np
import matplotlib.pyplot as plt
from fitting_functions import Ray 

def get_template_config(lineout: Ray, template_type: str, **kwargs):
    """
    Get configuration dictionary for different template types
    """
    # Base configurations for different simulation types
    templates = {
        "basic": {  # Fallback with reasonable defaults
            'xmax': [int(-4*np.sqrt(380/lineout.rqm_factor)), int(4*np.sqrt(380/lineout.rqm_factor))], # try to get 8 ion inertial lengths in x direction. sqrt(380) ~ 20
            'if_move': ".true.",
            'v_move': 0.01,
            "nx_p": None, # Get about the same resolution in x and y
            "num_par_x": [7, 7],
            "ndump": None,
            "dx": 0.07,
            "dt": None,
            "tmax": None,
            "node_number": [2, 1],
            "n_threads": 1,
            "tile_number": [32, 64],
            "emf_boundary_x2": ["open", "open"],
            "part_boundary_x2": ["thermal", "thermal"],
            "reports": '"charge"',
            "rep_udist": '', # I believe that this is broken for the gpu version as well
            "ps_pmin": [-0.15, -0.15, -0.15],
            "ps_pmax": [0.15, 0.15, 0.15],
            "ps_np": [128, 1024, 128],
            "ps_nx": [256, 1024],
            "emf_reports": '"e1", "e2", "e3", "b1", "b2", "b3"',
            "ps_xmin_x1": lineout.osiris_length[0],
            "smooth_type": "compensated",
            "smooth_order": "2",
            "interpolation": "quadratic",
        }
    }
    
    # Get base config and override with kwargs
    config = templates.get(template_type, templates["basic"]).copy()
    config.update(kwargs)
    
    # Add physics-based calculations
    mass_proton = 1836
    aluminum_mass_number = 27
    silicon_mass_number = 28
    al_charge_state = 13
    si_charge_state = 14
    B0 = 70_000  # Gauss, fully ionized aluminum ions

    config["upstream_gyrotime"] = int(mass_proton * aluminum_mass_number / al_charge_state / lineout.rqm_factor / (B0 / lineout.normalizations['magx'])) # 70k Gauss field, fully ionized alumium ions
    config["rqm_al"] = int(mass_proton * aluminum_mass_number / al_charge_state / lineout.rqm_factor)
    config["rqm_si"] = int(mass_proton * silicon_mass_number / si_charge_state / lineout.rqm_factor)
    config["tmax"] = config["upstream_gyrotime"] * 15 # want to run for 10 upstream gyroperiods
    config["nx_p"] = [int((config["xmax"][1] - config["xmax"][0]) / config['dx']), int((lineout.osiris_length[-1] - lineout.osiris_length[0]) / config['dx'])] # Get about the same resolution in x and y
    config["dt"] = config['dx'] * 0.99 / np.sqrt(2.0) # CFL condition. Factor of sqrt(2) to account for 2D simulation
    config["ndump"] = int(config["tmax"] / config['dt'] / 512) # 512 dumps total

    # # num_tiles must be a power of two and greater than n_cells_tot / 1024
    # n_cells_tot = config["nx_p"][0] * config["nx_p"][1]
    # num_tiles = 2 ** np.ceil(np.log2(n_cells_tot / 1024))
    # config["tile_number"] = [np.sqrt(int(num_tiles)), np.sqrt(int(num_tiles))]

    return config


def write_input_file(lineout: Ray, template_type: str, **kwargs):
    """
    Write OSIRIS input file with different template configurations
    
    Parameters:
    - lineout: Ray object with fitted data
    - template_type: "basic", etc.
    - **kwargs: Override any default values
    """
    # Default configuration
    config = get_template_config(lineout = lineout, template_type=template_type, **kwargs)

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
! It seems like restart is broken on the GPU version
!  ndump_fac = -1,
!  ndump_time = 3590, !once/hour
!  if_restart = .false.,
!  if_remold = .true.,
\u007d


!----------spatial limits ----------
space
\u007b
 xmin(1:2) = {config['xmax'][0]}, {int(lineout.osiris_length[0])},
 xmax(1:2) = {config['xmax'][1]}, {int(lineout.osiris_length[-1])},
 if_move(1:2) = .false., {config['if_move']},
 move_u = {-1*config['v_move']},
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
 type(1:2,2) =  "{config["emf_boundary_x2"][0]}", "{config["emf_boundary_x2"][1]}", ! boundaries for x2
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
  interpolation = "{config['interpolation']}",
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
  profile_type(1:2) = "uniform", "piecewise-linear",
  num_x = {lineout.math_funcs['edens']['x'].count(',') + 1},
  x(1:{lineout.math_funcs['edens']['x'].count(',') + 1},2) = {lineout.math_funcs['edens']['x']},
  fx(1:{lineout.math_funcs['edens']['dens'].count(',') + 1},2) = {lineout.math_funcs['edens']['dens']},
\u007d

!----------boundary conditions for electrons----------
spe_bound
\u007b
 type(1:2,2) = "{config["part_boundary_x2"][0]}","{config["part_boundary_x2"][1]}",
 uth_bnd(1:3,1,2) = {lineout._get_field_values('tele')[0]},{lineout._get_field_values('tele')[0]},{lineout._get_field_values('tele')[0]},
 uth_bnd(1:3,2,2) = {lineout._get_field_values('tele')[1]},{lineout._get_field_values('tele')[1]},{lineout._get_field_values('tele')[1]},
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
  profile_type(1:2) = "uniform", "piecewise-linear",
  num_x = {lineout.math_funcs['edens']['x'].count(',') + 1},
  x(1:{lineout.math_funcs['aldens']['x'].count(',') + 1},2) = {lineout.math_funcs['aldens']['x']},
  fx(1:{lineout.math_funcs['aldens']['dens'].count(',') + 1},2) = {lineout.math_funcs['aldens']['dens']},
\u007d

!----------boundary conditions for Alumium ions----------
spe_bound
\u007b
 type(1:2,2) = "thermal", "thermal",
 uth_bnd(1:3,1,2) = {lineout._get_field_values('tion')[0]},{lineout._get_field_values('tion')[0]},{lineout._get_field_values('tion')[0]},
 uth_bnd(1:3,2,2) = {lineout._get_field_values('tion')[1]},{lineout._get_field_values('tion')[1]},{lineout._get_field_values('tion')[1]},
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
  profile_type(1:2) = "uniform", "piecewise-linear",
  num_x = {lineout.math_funcs['edens']['x'].count(',') + 1},
  x(1:{lineout.math_funcs['sidens']['x'].count(',') + 1},2) = {lineout.math_funcs['sidens']['x']},
  fx(1:{lineout.math_funcs['sidens']['dens'].count(',') + 1},2) = {lineout.math_funcs['sidens']['dens']},
\u007d

!----------boundary conditions for Silicon ions----------
spe_bound
\u007b
 type(1:2,2) = "thermal","thermal",
 uth_bnd(1:3,1,2) = {lineout._get_field_values('tion')[0]},{lineout._get_field_values('tion')[0]},{lineout._get_field_values('tion')[0]},
 uth_bnd(1:3,2,2) = {lineout._get_field_values('tion')[1]},{lineout._get_field_values('tion')[1]},{lineout._get_field_values('tion')[1]},
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

!--------Current smoothing----------
smooth
\u007b
  type = "{config["smooth_type"]}",
  order = {config["smooth_order"]},
\u007d
'''

def main(FLASH_data, start_point, end_point, inputfile_name, rqm_factor, template_type, **kwargs):
    """
    Parameters:
    - FLASH_data: Path to the FLASH data directory
    - start_point: Start point of the lineout (x, y, z)
    - end_point: End point of the lineout (x, y, z)
    - inputfile_name: Name of the output input file for OSIRIS
    - rqm_factor: RQM factor to normalize by
    - template_type: Type of template configuration to use
    - **kwargs: Additional keyword arguments for configuration
    """
    
    # Create a Ray object for the lineout
    lineout = Ray(FLASH_data, start_point, end_point, rqm_factor=rqm_factor)

    lineout.fit("magx", degree=10, fit_func="piecewise", plot=False)
    lineout.fit('magy', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('magz', degree=10, fit_func="piecewise", plot=False)

    lineout.fit('Ex', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('Ey', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('Ez', degree=10, fit_func="piecewise", plot=False)

    lineout.fit_density("sidens")
    lineout.fit_density("aldens")
    lineout.fit_density("edens")

    lineout.fit('v_ex', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('v_ix', degree=10, fit_func="piecewise", plot=False)

    lineout.fit('v_iy', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('v_ey', degree=10, fit_func="piecewise", plot=False)

    lineout.fit('v_iz', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('v_ez', degree=10, fit_func="piecewise", plot=False)

    lineout.fit('tele', degree=10, fit_func="piecewise", plot=False)
    lineout.fit('tion', degree=10, fit_func="piecewise", plot=False)

    # Write the input file for OSIRIS
    input_file_content = write_input_file(lineout = lineout, template_type=template_type, **kwargs)
    with open(inputfile_name, 'w') as f:
        f.write(input_file_content)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run simplified MagShockZ analysis and generate OSIRIS input file.")
  parser.add_argument('-d', '--data_path', type=str, default="/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2024-06/MAGON/MagShockZ_hdf5_chk_0005", help="Path to the FLASH data directory")
  parser.add_argument('-s', '--start_point', type=float, nargs=3, default=(0, 0.07, 0), help="Start point of the lineout (x, y, z)")
  parser.add_argument('-e', '--end_point', type=float, nargs=3, default=(0, 0.3, 0), help="End point of the lineout (x, y, z)")
  parser.add_argument('-i', '--inputfile_name', type=str, default="testing_writeout.txt", help="Name of the output input file for OSIRIS")
  parser.add_argument('-t', '--template_type', type=str, default="basic", help="Type of template configuration to use")
  parser.add_argument('-m', '--rqm_factor', type=float, default=100, help="RQM factor to normalize by")
  args = parser.parse_args()
  print(args)

  print(f"Running MagShockZ analysis with data path: {args.data_path},\n start point: {args.start_point},\n end point: {args.end_point},\n input file name: {args.inputfile_name},\n rqm factor: {args.rqm_factor},\n template type: {args.template_type}\n")
  main(FLASH_data = args.data_path, start_point = args.start_point, end_point = args.end_point, inputfile_name = args.inputfile_name, rqm_factor=args.rqm_factor, template_type=args.template_type)