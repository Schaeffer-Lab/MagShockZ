## IMPORTS ##
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_template_config(template_type: str, **kwargs):
    """
    Get configuration dictionary for different template types
    """
    # Base configurations for different simulation types
    templates = {
        "basic": {  # Fallback with reasonable defaults
            'xmax': [-10, 10], 
            'ymax': [0, 400],
            'rqm': 100,
            "num_par_x": [5, 5],
            'n0_p': 50, # Density of the piston in n0
            'B0': 0.04, # Background magnetic field in osiris units
            'vthe_p': 0.001, # Thermal velocity of the piston electrons in c
            'vthe_b': 0.0001, # Thermal velocity of the background electrons in c
            'v_p': 0.01, # Velocity of the piston in c
            'dx': 0.5, # c/wpe / n_cells
            'piston_edge': 100, # Position of the piston edge in y
            "interpolation": "cubic",
            "nx_p": [None, None],
            "ndump": None,
            "dt": None,
            "tmax": None,
            "tile_number": [None, None],
            'vthi_p': None,
        },
        "debug": {  # Fallback with reasonable defaults
            'xmax': [-10, 10], 
            'ymax': [0, 400],
            'rqm': 100,
            "num_par_x": [5, 5],
            'n0_p': 50, # Density of the piston in n0
            'B0': 0.04, # Background magnetic field in osiris units
            'vthe_p': 0.03, # Thermal velocity of the piston electrons in c
            'vthe_b': 0.001, # Thermal velocity of the background electrons in c
            'v_p': 0.01, # Velocity of the piston in c
            'dx': 0.2, # c/wpe / n_cells
            'piston_edge': 100, # Position of the piston edge in y
            "interpolation": "cubic",
            "smooth_type": "none",
            "smooth_order": 1,
            "vpml_bnd_size": 5,
            "nx_p": [None, None],
            "ndump": None,
            "dt": None,
            "tmax": None,
            "tile_number": [None, None],
            'vthi_p': None,
        }
    }
    defaults = {
        "node_number": [2, 1],
        "emf_boundary_x2": ["open", "open"],
        "part_boundary_x2": ["thermal", "thermal"],
        "reports": '"charge"',
        "rep_udist": '', # I believe that this is broken for the gpu version as well
        "ps_pmin": [-0.15, -0.15, -0.15],
        "ps_pmax": [0.15, 0.15, 0.15],
        "ps_np": [128, 1024, 128],
        "ps_nx": [256, 1024],
        "emf_reports": '"e1", "e2", "e3", "b1", "b2", "b3"',
        "n_threads": 1,

    }
    config = templates.get(template_type, templates["basic"]).copy()
    config.update(kwargs)
    config.update(defaults)

    config['vthi_p'] = config['vthe_p'] / np.sqrt(config['rqm']) # Thermal velocity of the piston ions in c
    config['vthi_b'] = config['vthe_b'] / np.sqrt(config['rqm']) # Thermal velocity of the background ions in c
    config['nx_p'] = [int((config['xmax'][1]- config['xmax'][0]) / config['dx']), int((config['ymax'][1] - config['ymax'][0]) / config['dx'])]
    config['compression_length'] = config['piston_edge'] / 3 # Length of the compression region in y
    
    if config['v_p'] == 0.0:
        config['tmax'] = 10_000
    config['tmax'] = 1.5 * config['ymax'][1] / config['v_p']
    config['dt'] = config['ymax'][-1] / config["nx_p"][1] / np.sqrt(2.0)
    config["ndump"] = int(config["tmax"] / config['dt'] / 256) # 256 dumps total

    if template_type == "debug":
        config["ndump"] = 80
        config["tmax"] = config['tmax'] / 10.0

    n_tiles_min = config['nx_p'][0] * config['nx_p'][1] // 1024

    # Just keep typing in powers of two until you get n_tiles > n_tiles_min

    i = 0
    j = 0
    while True:
        n_tiles_x = 2**i
        n_tiles_y = 2**j
        n_tiles = n_tiles_x * n_tiles_y
        if n_tiles > n_tiles_min:
            break
        j += 1
        n_tiles_y = 2**j
        n_tiles = n_tiles_x * n_tiles_y
        if n_tiles > n_tiles_min:
            break
        j += 1
        n_tiles_y = 2**j
        n_tiles = n_tiles_x * n_tiles_y
        if n_tiles > n_tiles_min:
            break
        i += 1

    config['tile_number'] = [n_tiles_x, n_tiles_y]

    return config


def write_input_file(template_type: str, **kwargs):
    """
    Write OSIRIS input file with different template configurations
    
    Parameters:
    - lineout: Ray object with fitted data
    - template_type: "basic", etc.
    - **kwargs: Override any default values
    """
    # Default configuration
    config = get_template_config(template_type=template_type, **kwargs)

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
  xmin(1:2) = {config['xmax'][0]}, {config['ymax'][0]}, ! x and y min
  xmax(1:2) = {config['xmax'][1]}, {config['ymax'][1]}, ! x and y max
\u007d


!----------time limits ----------
time
\u007b
  tmin = 0.0,
  tmax = {config["tmax"]}, ! Just to make your life easier, your upstream gyrotime for an rqm of {config["rqm"]}
\u007d

!----------field solver set up----------
el_mag_fld
\u007b
  type_init_b(1:3) = "math func", "math func", "math func",
  init_b_mfunc(1) = "if(x2 < {config['piston_edge']}, 0.0, if(x2 < {config['piston_edge']+config['compression_length']}, {config['B0'] + config['B0'] * config['piston_edge'] / config['compression_length']}, {config['B0']}))", ! x profile
  init_b_mfunc(2) = "0.0",
  init_b_mfunc(3) = "0.0",

  type_init_e(1:3) = "math func", "math func", "math func",
  init_e_mfunc(1) = "0.0", ! x profile
  init_e_mfunc(2) = "0.0",
  init_e_mfunc(3) = "if(x2 > {config['piston_edge']}, if(x2 < {config['piston_edge'] + config['compression_length']}, {config['v_p'] * (config['B0'] + config['B0'] * config['piston_edge'] / config['compression_length'])}, 0.0), 0.0)", 
\u007d

!----------boundary conditions for em-fields ----------
emf_bound
\u007b
  type(1:2,2) = "{config["emf_boundary_x2"][0]}", "{config["emf_boundary_x2"][1]}", ! boundaries for x2
  vpml_bnd_size = {config['vpml_bnd_size']},
  vpml_diffuse(1:2,1) = .true., .true., ! Allows for damping of electrostatic fields in vpml
  vpml_diffuse(1:2,2) = .true., .true.,
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
  interpolation = "{config["interpolation"]}",
  num_species = 3,
\u007d

!----------information for electrons----------
species
\u007b
  name = "electrons",
  rqm = -1.0,
  num_par_x(1:2) = {config["num_par_x"][0]}, {config["num_par_x"][1]}, ! number of particles per cell in x and y directions
\u007d

!----------inital proper velocities electrons-----------------
udist
\u007b
  use_spatial_uth = .true.,
  use_spatial_ufl = .true.,
  spatial_uth(1) = "if(x2 < {config['piston_edge'] + config['compression_length']}, {config['vthe_p']}, {config['vthe_b']})",
  spatial_uth(2) = "if(x2 < {config['piston_edge'] + config['compression_length']}, {config['vthe_p']}, {config['vthe_b']})",
  spatial_uth(3) = "if(x2 < {config['piston_edge'] + config['compression_length']}, {config['vthe_p']}, {config['vthe_b']})",
  spatial_ufl(1:3) = "0.0", "if(x2 < {config['piston_edge'] + config['compression_length']}, {config['v_p']}, 0.0)", "0.0", ! drift velocity of the electrons in c
\u007d

!----------density profile for electrons----------
profile
\u007b
  profile_type = "math func",
  math_func_expr = "if(x2 < {config['piston_edge']}, {config['n0_p']}, if(x2 < {config['piston_edge'] + config['compression_length']}, {1.0 + 1.0*config['piston_edge'] / config['compression_length']}, 1.0))",
\u007d

!----------boundary conditions for electrons----------
spe_bound
\u007b
  type(1:2,2) = "{config["part_boundary_x2"][0]}","{config["part_boundary_x2"][1]}",
  thermal_type = "thermal",
  uth_bnd(1:3,1,2) = {config['vthe_p']},{config['vthe_p']},{config['vthe_p']},
  uth_bnd(1:3,2,2) = {config['vthe_b']},{config['vthe_b']},{config['vthe_b']},
\u007d

!----------diagnostic for electrons----------
diag_species
\u007b
  ndump_fac = 1,
  reports = {config["reports"]},
  ndump_fac_pha = 1,
  ps_pmin(1:3) = {config["ps_pmin"][0]}, {config["ps_pmin"][1]}, {config["ps_pmin"][2]},
  ps_pmax(1:3) = {config["ps_pmax"][0]}, {config["ps_pmax"][1]}, {config["ps_pmax"][2]},
  ps_xmin(1:2) = {config['xmax'][0]}, {config['ymax'][0]}, ! phase space covers the entire domain. change as needed
  ps_xmax(1:2) = {config['xmax'][1]}, {config['ymax'][1]}, ! phase space covers the entire domain. change as needed
  ps_np(1:3) = {config["ps_np"][0]}, {config["ps_np"][1]}, {config["ps_np"][2]},
  ps_nx(1:2) = {config["ps_nx"][0]}, {config["ps_nx"][1]},
  phasespaces = "p1x1", "p1x2", "p2x1", "p2x2",
\u007d
   
!----------information for background ions----------
species
\u007b
  name = "background_ions",
  rqm = {config["rqm"]},
  num_par_x(1:2) = {config["num_par_x"][0]}, {config["num_par_x"][1]}, ! number of particles per cell in x and y directions
\u007d

!----------inital proper velocities background ions-----------------
udist
\u007b
  uth_type = "thermal",
  uth(1:3) = {config['vthi_b']}, {config['vthi_b']}, {config['vthi_b']},
  use_spatial_ufl = .true.,
  spatial_ufl(1) = 0.0,
  spatial_ufl(2) = "if(x2 < {config['piston_edge'] + config['compression_length']}, {config['v_p']}, 0.0)", ! drift velocity of the background ions in c
  spatial_ufl(3) = 0.0,
\u007d

!----------density profile for background ions----------
profile
\u007b
  profile_type = "math func",
  math_func_expr = "if(x2 < {config['piston_edge']}, 0.0, if(x2 < {config['piston_edge'] + config['compression_length']}, {1.0 + 1.0*config['piston_edge'] / config['compression_length']}, 1.0))",
\u007d

!----------boundary conditions for background ions----------
spe_bound
\u007b
  type(1:2,2) = "thermal", "thermal",
  thermal_type = "thermal",
  uth_bnd(1:3,1,2) = {config['vthi_b']},{config['vthi_b']},{config['vthi_b']},
  uth_bnd(1:3,2,2) = {config['vthi_b']},{config['vthi_b']},{config['vthi_b']},
\u007d

!----------diagnostic for background ions----------
diag_species
\u007b
  ndump_fac = 1,
  reports = {config["reports"]},
  ndump_fac_pha = 1,
  ps_pmin(1:3) = {config["ps_pmin"][0]}, {config["ps_pmin"][1]}, {config["ps_pmin"][2]}, 
  ps_pmax(1:3) = {config["ps_pmax"][0]}, {config["ps_pmax"][1]}, {config["ps_pmax"][2]},
  ps_xmin(1:2) = {config['xmax'][0]}, {config['ymax'][0]},
  ps_xmax(1:2) = {config['xmax'][1]}, {config['ymax'][1]},
  ps_np(1:3) = {config["ps_np"][0]}, {config["ps_np"][1]}, {config["ps_np"][2]},
  ps_nx(1:2) = {config["ps_nx"][0]}, {config["ps_nx"][1]},
  phasespaces = "p1x1", "p1x2", "p2x1", "p2x2",
\u007d

!----------information for piston ions ----------
species
\u007b
  name = "piston_ions",
  rqm= {config["rqm"]},
  num_par_x = {config["num_par_x"][0]}, {config["num_par_x"][1]}, ! number of particles per cell in x and y directions
\u007d

!----------information for piston ions ----------

udist
\u007b
  uth(1:3) = {config['vthi_p']}, {config['vthi_p']}, {config['vthi_p']},
  ufl(1:3) = 0.00, {config['v_p']}, 0.00,
\u007d

!----------density profile for background ions----------
profile
\u007b
  profile_type = "math func",
  math_func_expr = "if(x2 < {config['piston_edge']}, {config['n0_p']}, 0.0)",
\u007d


!----------boundary conditions for piston ions ----------
spe_bound
\u007b
  type(1:2,2) = "{config["part_boundary_x2"][0]}","{config["part_boundary_x2"][1]}",
  thermal_type = "thermal",
  uth_bnd(1:3,1,2) = {config['vthi_p']},{config['vthi_p']},{config['vthi_p']},
  uth_bnd(1:3,2,2) = {config['vthi_b']},{config['vthi_b']},{config['vthi_b']},
  ufl_bnd(1:3,1,2) = 0.0, {config['v_p']}, 0.0,
\u007d

!----------diagnostic for piston ions----------
diag_species
\u007b
  ndump_fac = 1,
  reports = {config["reports"]},
  ndump_fac_pha = 1,
  ps_pmin(1:3) = {config["ps_pmin"][0]}, {config["ps_pmin"][1]}, {config["ps_pmin"][2]}, 
  ps_pmax(1:3) = {config["ps_pmax"][0]}, {config["ps_pmax"][1]}, {config["ps_pmax"][2]},
  ps_xmin(1:2) = {config['xmax'][0]}, {config['ymax'][0]},
  ps_xmax(1:2) = {config['xmax'][1]}, {config['ymax'][1]},
  ps_np(1:3) = {config["ps_np"][0]}, {config["ps_np"][1]}, {config["ps_np"][2]},
  ps_nx(1:2) = {config["ps_nx"][0]}, {config["ps_nx"][1]},
  phasespaces = "p1x1", "p1x2", "p2x1", "p2x2",
\u007d

! ------ Current smoothing -------
smooth
\u007b
  type = "{config['smooth_type']}",
  order = {config['smooth_order']},
\u007d
'''

def main(template_type, inputfile_name, **kwargs):
    """
    Parameters:
    - inputfile_name: Name for the input file to be generated
    - template_type: Type of template configuration to use
    - **kwargs: Additional keyword arguments for configuration
    """
  

    # Write the input file for OSIRIS
    input_file_content = write_input_file(template_type=template_type, **kwargs)
    with open(inputfile_name, 'w') as f:
        f.write(input_file_content)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run simplified MagShockZ analysis and generate OSIRIS input file.")
  parser.add_argument('-i', '--inputfile_name', type=str, default="testing_writeout.txt", help="Name of the output input file for OSIRIS")
  parser.add_argument('-t', '--template_type', type=str, default="debug", help="Type of template configuration to use")
  args = parser.parse_args()
  print("Template type:", args.template_type)
  main(template_type=args.template_type, inputfile_name=args.inputfile_name)