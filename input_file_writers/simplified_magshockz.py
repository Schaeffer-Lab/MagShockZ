## IMPORTS ##
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
import yt
import numpy as np
import matplotlib.pyplot as plt
from fitting_functions import Ray 
import plasmapy
import astropy

def get_template_config(lineout: Ray, template_type: str, dim = 2, **kwargs, ):
    """
    Get configuration dictionary for different template types
    """
    # Base configurations for different simulation types
    templates = {
      "basic": {  # Fallback with reasonable defaults
          'algorithm': "cuda",
          'xmax': [int(-4*np.sqrt(3800/lineout.rqm_factor)), int(4*np.sqrt(3800/lineout.rqm_factor))],
          "nx_p": None, # Get about the same resolution in x and y
          "num_par_x": [10, 10],
          "ndump": None,
          "dx": 0.3,
          "dt": None,
          "tmax": None,
          "node_number": [2, 1],
          "n_threads": 1,
          "tile_number": [None, None],
          "emf_boundary_x2": ["pmc", "vpml"],
          "vpml_bnd_size": 50,
          "vpml_diffuse": ".false.",
          "part_boundary_x2": ["thermal", "thermal"],
          "reports": '"charge"',
          "rep_udist": '', # I believe that this is broken for the gpu version as well
          "ps_pmin": [-0.5, -0.5, -0.15], 
          "ps_pmax": [0.5, 0.5, 0.15],
          "ps_np": [128, 1024, 128],
          "ps_nx": [256, 1024],
          "emf_reports": '"e1", "e2", "e3", "b1", "b2", "b3"',
          "ps_xmin_x1": lineout.osiris_length[0],
          "smooth_type": "none",
          "smooth_order": "1",
          "interpolation": "cubic",
      },
      "perlmutter": {
          'algorithm': "cuda",
          'xmax': [int(-1*np.sqrt(3800/lineout.rqm_factor)), int(1*np.sqrt(3800/lineout.rqm_factor))],
          "nx_p": None, # Get about the same resolution in x and y
          "num_par_x": [100, 100],
          "ndump": None,
          "dx": 0.2,
          "dt": None,
          "tmax": None,
          "tile_number": [None, None],
          "node_number": [25, 4], # Should be a multiple of 4 for Perlmutter, 4 GPUs per node
          "n_threads": 1,
          "emf_boundary_x2": ["pmc", "vpml"],
          "vpml_bnd_size": 100,
          "vpml_diffuse": ".true.",
          "part_boundary_x2": ["thermal", "thermal"],
          "reports": '"charge, savg", "j1, savg", "j2, savg", "j3, savg"',
          "rep_udist": '', # I believe that this is broken for the gpu version
          'e_ps_pmin': [-1, -1, -0.5],
          'e_ps_pmax': [1, 1, 0.5],
          'i_ps_pmin': [-0.1, -0.1, -0.05],
          'i_ps_pmax': [0.1, 0.1, 0.05],
          "ps_np": [8000, 64, 128],
          "ps_nx": [4096, 128],
          "emf_reports": '"e1, savg", "e2, savg", "e3, savg", "b1, savg", "b2, savg", "b3, savg"',
          "ps_xmin_x1": lineout.osiris_length[0],
          "smooth_type": "binomial",
          "smooth_order": "2",
          "interpolation": "cubic",
      }
    }
    
    # Get base config and override with kwargs
    config = templates.get(template_type, templates["basic"]).copy()
    config.update(kwargs)
    
    mass_proton = 1836
    aluminum_mass_number = 27
    silicon_mass_number = 28
    al_charge_state = 13
    si_charge_state = 14
    B0 = 100_000  # Gauss

    config["upstream_gyrotime"] = int(mass_proton * aluminum_mass_number / al_charge_state / lineout.rqm_factor / (B0 / lineout.normalizations['magx'])) # 100k Gauss field, fully ionized alumium ions
    config["rqm_al"] = int(mass_proton * aluminum_mass_number / al_charge_state / lineout.rqm_factor)
    config["rqm_si"] = int(mass_proton * silicon_mass_number / si_charge_state / lineout.rqm_factor)
    config["tmax"] = config["upstream_gyrotime"] * 15
    # Enforce that we are resolving the ion debye length
    # lambda_di = sqrt(epsilon_0 k_B T_i / n_i q^2) = sqrt(epsilon_0 k_B T_i / n_i q^2)
    # lambda_ci/(c/omega_pe) = sqrt(rqm) vthion_osiris
    upstream_vthion = lineout.get_upstream_value('vthion')
    print(f"Upstream vthion: {upstream_vthion}")
    config["dx"] = np.sqrt(config["rqm_al"]) * upstream_vthion
    
    # Handle grid size based on dimensionality
    if dim == 1:
        # For 1D, only use the lineout direction
        config["nx_p"] = [int((lineout.osiris_length[-1] - lineout.osiris_length[0]) / config['dx'])]
        config["dt"] = np.format_float_scientific(config['dx'] * 0.95, 3)  # CFL for 1D
        n_cells_tot = config["nx_p"][0]
    else:  # dim == 2
        config["nx_p"] = [int((config["xmax"][1] - config["xmax"][0]) / config['dx']), int((lineout.osiris_length[-1] - lineout.osiris_length[0]) / config['dx'])]
        config["dt"] = np.format_float_scientific(config['dx'] * 0.95 / np.sqrt(2.0), 3) # CFL condition for 2D
        n_cells_tot = config["nx_p"][0] * config["nx_p"][1]
    
    config["ndump"] = int(config["tmax"] / config['dt'] / 2048) # 512 dumps total

    # num_tiles must be a power of two and greater than n_cells_tot / 1024
    if dim == 1:
        # For 1D, use single tile dimension
        i = 0
        while 2**i < n_cells_tot / 1024:
            i += 1
        # Ensure at least 7 cells per tile
        while config['nx_p'][0] / 2**i < 7 and i > 0:
            i -= 1
        config["tile_number"] = [2**i]
    else:  # dim == 2
        i, j = 0, 0
        while 2**i * 2**j < n_cells_tot / 1024:
            if 2**i <= 2**j:
                i += 1
            else:
                j += 1
        config["tile_number"] = [2**i, 2**j]

        while config['nx_p'][0] / config['tile_number'][0] < 7 or config['nx_p'][1] / config['tile_number'][1] < 7:
            if config['nx_p'][0] / config['tile_number'][0] < 7:
                i -= 1
                j += 1
                config["tile_number"] = [2**i, 2**j]
            if config['nx_p'][1] / config['tile_number'][1] < 7:
                i += 1
                j -= 1
                config["tile_number"] = [2**i, 2**j]
    

    n_species = 3
    if dim == 2:
        n_particles = n_cells_tot * config['num_par_x'][0] * config['num_par_x'][1] * n_species
        n_bytes_particles = n_particles* 2 * 70 # maria says ~70 bytes per particle. I don't know if this is single or double precision, we also need to allocate for twice as many particles

    elif dim ==1:
        n_particles = config['nx_p'][0] * config['num_par_x'][0] * n_species
        n_bytes_particles = n_particles* 2 * 70 # maria says ~70 bytes per particle. I don't know if this is single or double precision, we also need to allocate for twice as many particles
    mem_per_GPU = 40e9
    max_bytes_per_GPU = mem_per_GPU * .8 # 80% of 16GB
    print("Number of particles: ", np.format_float_scientific(n_particles,3))

    print("Recommended number of GPUs: ", np.ceil(n_bytes_particles/max_bytes_per_GPU))
    print("Recommended number of nodes:", np.ceil(n_bytes_particles/max_bytes_per_GPU/4))

 
    return config

def write_input_file_2D(lineout: Ray, template_type: str, **kwargs):
  config = get_template_config(lineout = lineout, template_type=template_type, **kwargs)
  
  # Load and render template
  template_dir = Path(__file__).parent
  env = Environment(loader=FileSystemLoader(template_dir))
  template = env.get_template('MagShockZ_quasi1D_TEMPLATE.jinja')
  content = template.render(config=config, lineout=lineout)

  return content

def write_input_file1D(lineout: Ray, template_type: str, **kwargs):
   config = get_template_config(lineout = lineout, template_type=template_type, dim = 1, **kwargs)

   template_dir = Path(__file__).parent
   env = Environment(loader=FileSystemLoader(template_dir))
   template = env.get_template('MagShockZ_1D_TEMPLATE.jinja')
   content = template.render(config=config, lineout=lineout)

   return content


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

    # For 1D simulations, use dim_var="x1" since the lineout is along the y-axis
    # and maps to the simulation x1 axis
    lineout.fit("magx", degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('magy', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('magz', degree=10, fit_func="piecewise", plot=False, dim_var="x1")

    lineout.fit('Ex', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('Ey', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('Ez', degree=10, fit_func="piecewise", plot=False, dim_var="x1")

    lineout.fit_density("sidens")
    lineout.fit_density("aldens")
    lineout.fit_density("edens")

    lineout.fit('v_ex', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('v_ix', degree=10, fit_func="piecewise", plot=False, dim_var="x1")

    lineout.fit('v_iy', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('v_ey', degree=10, fit_func="piecewise", plot=False, dim_var="x1")

    lineout.fit('v_iz', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('v_ez', degree=10, fit_func="piecewise", plot=False, dim_var="x1")

    lineout.fit('vthele', degree=10, fit_func="piecewise", plot=False, dim_var="x1")
    lineout.fit('vthion', degree=10, fit_func="piecewise", plot=False, dim_var="x1")


    # Write the input file for OSIRIS
    input_file_content = write_input_file1D(lineout = lineout, template_type=template_type, **kwargs)
    with open(inputfile_name, 'w') as f:
        f.write(input_file_content)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run simplified MagShockZ analysis and generate OSIRIS input file.")
  parser.add_argument('-d', '--data_path', type=str, default="/pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_plt_cnt_0009", help="Path to the FLASH data directory")
  parser.add_argument('-s', '--start_point', type=float, nargs=3, default=(0, 0.07, 0), help="Start point of the lineout (x, y, z)")
  parser.add_argument('-e', '--end_point', type=float, nargs=3, default=(0, 0.7, 0), help="End point of the lineout (x, y, z)")
  parser.add_argument('-i', '--inputfile_name', type=str, default="testing_writeout.txt", help="Name of the output input file for OSIRIS")
  parser.add_argument('-t', '--template_type', type=str, default="basic", help="Type of template configuration to use")
  parser.add_argument('-m', '--rqm_factor', type=float, default=100, help="RQM factor to normalize by")
  args = parser.parse_args()
  print(args)

  print(f"Writing MagShockZ input deck from FLASH data: {args.data_path},\n start point: {args.start_point},\n end point: {args.end_point},\n input file name: {args.inputfile_name},\n rqm factor: {args.rqm_factor},\n template type: {args.template_type}\n")
  main(FLASH_data = args.data_path, start_point = args.start_point, end_point = args.end_point, inputfile_name = args.inputfile_name, rqm_factor=args.rqm_factor, template_type=args.template_type)