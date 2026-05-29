## IMPORTS ##
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import yt
import numpy as np
import matplotlib.pyplot as plt
from fitting_functions import Ray 
import plasmapy
import astropy
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# import pydantic #TODO include
# from ..src import analysis_utils

def get_template_config(lineout: Ray, args, **kwargs):
    
    mass_proton = 1836 
    aluminum_mass_number = 27
    silicon_mass_number = 28
    al_charge_state = args.al_charge_state
    si_charge_state = 13
    B0 = args.B0_Gauss  # Gauss #TODO user input
    
    config = {}

    config["upstream_gyrotime"] = int(mass_proton * aluminum_mass_number / al_charge_state / lineout.rqm_factor / (B0 / lineout.normalizations['magx']))
    config["rqm_al"] = int(mass_proton * aluminum_mass_number / al_charge_state / lineout.rqm_factor)
    config["rqm_si"] = int(mass_proton * silicon_mass_number / si_charge_state / lineout.rqm_factor)
    config["tmax"] = config["upstream_gyrotime"] * args.tmax
    config['algorithm'] = args.algorithm
    config['num_par_x'] = args.num_par_x
    config['ps_nx'] = args.ps_nx
    config['n_ave'] = args.n_ave

    config["ps_np"] = args.ps_np
    config["i_ps_pmin"] = args.i_ps_pmin
    config["i_ps_pmax"] = args.i_ps_pmax
    config["e_ps_pmin"] = args.e_ps_pmin
    config["e_ps_pmax"] = args.e_ps_pmax

    config["interpolation"] = args.interpolation
    config["smooth_type"] = args.smooth_type
    config["smooth_order"] = args.smooth_order
    config["emf_boundary"] = args.emf_boundary
    config["part_boundary"] = args.part_boundary


    # Enforce that we are resolving the ion debye length
    # lambda_di = sqrt(epsilon_0 k_B T_i / n_i q^2) = sqrt(epsilon_0 k_B T_i / n_i q^2)
    # lambda_ci/(c/omega_pe) = sqrt(rqm) vthion_osiris
    upstream_vthion = lineout.get_upstream_value('vthion')
    print(f"Upstream vthion: {upstream_vthion}")
    if args.dx == "ion_debye":
        config["dx"] = np.sqrt(config["rqm_al"]) * upstream_vthion
    else:
        config["dx"] = float(args.dx)

    # Handle grid size based on dimensionality
    if args.dim == 1:
        # For 1D, only use the lineout direction
        config["nx_p"] = [int((lineout.osiris_length[-1] - lineout.osiris_length[0]) / config['dx'])]
        config["dt"] = np.format_float_scientific(config['dx'] * 0.95, 3)  # CFL for 1D
        n_cells_tot = config["nx_p"][0]
    else:
        config["nx_p"] = [int((config["xmax"][1] - config["xmax"][0]) / config['dx']), int((lineout.osiris_length[-1] - lineout.osiris_length[0]) / config['dx'])]
        config["dt"] = np.format_float_scientific(config['dx'] * 0.95 / np.sqrt(2.0), 3) # CFL condition for 2D
        n_cells_tot = config["nx_p"][0] * config["nx_p"][1]
    
    config["ndump"] = int(config["tmax"] / float(config['dt']) / args.ndump_tot)

    # num_tiles must be a power of two and greater than n_cells_tot / 1024
    if args.dim == 1:
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
    if args.dim == 2:
        n_particles = n_cells_tot * config['num_par_x'][0] * config['num_par_x'][1] * n_species
        n_bytes_particles = n_particles* 2 * 70 # maria says ~70 bytes per particle. I don't know if this is single or double precision, we also need to allocate for twice as many particles

    elif args.dim ==1:
        n_particles = config['nx_p'][0] * config['num_par_x'][0] * n_species
        n_bytes_particles = n_particles* 2 * 70 # maria says ~70 bytes per particle. I don't know if this is single or double precision, we also need to allocate for twice as many particles
    mem_per_GPU = 40e9
    max_bytes_per_GPU = mem_per_GPU * .8 # 80% of 16GB
    print("Number of particles: ", np.format_float_scientific(n_particles,3))

    print("Recommended number of GPUs: ", np.ceil(n_bytes_particles/max_bytes_per_GPU))
    print("Recommended number of nodes:", np.ceil(n_bytes_particles/max_bytes_per_GPU/4))

    # prepare reports
    config['reports'] = '"'
    for report in args.reports:
        config['reports'] += report + ', savg"'

    config['rep_udist'] = '"'
    for udist in args.rep_udist:
        config['rep_udist'] += udist + ', savg"'

    config['emf_reports'] = '"'
    for emf_report in args.emf_reports:
        config['emf_reports'] += emf_report + ', savg"'

    config['phasespaces'] = '"'
    for ps in args.phasespaces:
        config['phasespaces'] += ps + '",'

    return config

def write_input_file_2D(lineout: Ray, args):
  config = get_template_config(lineout = lineout, args = args)
  
  # Load and render template
  template_dir = Path(__file__).parent
  env = Environment(loader=FileSystemLoader(template_dir))
  template = env.get_template('MagShockZ_quasi1D_TEMPLATE.jinja')
  content = template.render(config=config, lineout=lineout)

  return content

def write_input_file1D(lineout: Ray, args):
   config = get_template_config(lineout = lineout, args = args)

   template_dir = Path(__file__).parent
   env = Environment(loader=FileSystemLoader(template_dir))
   template = env.get_template('MagShockZ_1D_TEMPLATE.jinja')
   content = template.render(config=config, lineout=lineout)

   return content


def main(args):
    """
    Parameters:
    - args: Namespace object containing the configuration for the input file and lineout.
    """
    
    # Create a Ray object for the lineout
    lineout = Ray(ds=args.data_path, start_pt=args.start_point, end_pt=args.end_point, rqm_factor=args.rqm_factor)

    if args.dim == 1:
        dim_var = "x1"
    elif args.dim == 2:
        dim_var = "x2"
    logging.info("Fitting magnetic field data with piecewise polynomials...")
    lineout.fit("magx", degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('magy', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('magz', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)

    logging.info("Fitting electric field data with piecewise polynomials...")
    lineout.fit('Ex', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('Ey', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('Ez', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)

    logging.info("Fitting density data with piecewise polynomials...")
    lineout.fit_density("sidens")
    lineout.fit_density("aldens")
    lineout.fit_density("edens")

    logging.info("Fitting velocity data with piecewise polynomials...")
    lineout.fit('v_ex', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('v_ix', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)

    lineout.fit('v_iy', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('v_ey', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)

    lineout.fit('v_iz', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('v_ez', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)

    lineout.fit('vthele', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)
    lineout.fit('vthion', degree=args.piecewise_degree, fit_func="piecewise", plot=False, dim_var=dim_var)


    # Write the input file for OSIRIS
    if args.dim == 1:
        input_file_content = write_input_file1D(lineout = lineout, args=args)
    elif args.dim == 2:
        input_file_content = write_input_file_2D(lineout = lineout, args=args)
    with open(args.inputfile_name, 'w') as f:
        f.write(input_file_content)

if __name__ == "__main__":
    debug = True # For now
    import argparse
    parser = argparse.ArgumentParser(description="Run simplified MagShockZ analysis and generate OSIRIS input file.")
    parser.add_argument('--data_path', type=str, help="Path to the FLASH data")
    parser.add_argument('--dim', type=int, choices=[1, 2], help="Dimensionality of the OSIRIS simulation (1 or 2)")
    parser.add_argument('--node_number', type=int, nargs = '+', help="Number of nodes to run on for OSIRIS simulation")
    parser.add_argument('--num_threads', type=int, help="Number of threads to use for OSIRIS simulation")
    parser.add_argument('--rqm_factor', type=int, help="RQM factor to normalize by")
    parser.add_argument('--algorithm', type=str, choices=["cuda", "cpu"], help="Algorithm to use for OSIRIS simulation")
    parser.add_argument('--start_point', type=float, nargs=3, help="Start point of the lineout (x, y, z) in cm")
    parser.add_argument('--end_point', type=float, nargs=3, help="End point of the lineout (x, y, z) in cm")
    parser.add_argument('--inputfile_name', type=str, help="Name that the OSIRIS input file will use")
    parser.add_argument('--dx', type=str, help="dx for OSIRIS, can say 'ion_debye' to instead set dx to resolve the ion debye length")
    parser.add_argument('--num_par_x', type=int, nargs='+', help="Number of particles per cell in each dimension (1 or 2 values depending on --dim)")
    parser.add_argument('--tmax', type=int, help="Maximum time for OSIRIS simulation in upstream gyroperiods")
    parser.add_argument('--emf_boundary', nargs = 2, choices=["pmc", "vpml"], type = str)
    parser.add_argument('--part_boundary', nargs = 2, choices=["thermal", "open"], type = str)
    parser.add_argument('--reports', type = str, nargs = '+', help = "List of particle diagnostics to use, ex: charge j1 j3 q2")
    parser.add_argument('--rep_udist', nargs = '+', type = str, help="List of distribution functions to report, ex: ufl1 ufl2 uth2")
    parser.add_argument('--emf_reports', type = str, nargs = '+', help = "List of EMF diagnostics to use, ex: e1 e2 part_e3 b1 b2 b3")
    parser.add_argument('--phasespaces', nargs = '+', type = str, help="List of phase space diagnostics to use, ex: p1x1 p2x1 p1p2x1")
    parser.add_argument('--interpolation', type=str, choices=["linear", "quadratic", "cubic"], help="Particle interpolation for OSIRIS")
    parser.add_argument('--smooth_type', type = str, choices = ["binomial", ""], help = "Type of smoothing to apply to fields each timestep, if any")
    parser.add_argument('--smooth_order', type = int, help = "Order of smoothing to apply to fields each timestep, if any")
    parser.add_argument('--ps_nx', nargs='+', type=int, help="Number of spatial bins for phase space diagnostics, (x1, x2) depending on --dim")
    parser.add_argument('--ps_np', nargs = 3, type=int, help="Number of momentum bins for phase space diagnostics (px, py, pz)")
    parser.add_argument('--i_ps_pmin', nargs = 3, type = float, help="Minimum momentum for ion phase space diagnostics (px, py, pz)")
    parser.add_argument('--i_ps_pmax', nargs = 3, type = float, help="Maximum momentum for ion phase space diagnostics (px, py, pz)")
    parser.add_argument('--e_ps_pmin', nargs = 3, type = float, help="Minimum momentum for electron phase space diagnostics (px, py, pz)")
    parser.add_argument('--e_ps_pmax', nargs = 3, type = float, help="Maximum momentum for electron phase space diagnostics (px, py, pz)")
    parser.add_argument('--ndump_tot', type=int, help="Total number of dumps to output during the simulation")
    parser.add_argument('--n_ave', nargs = '+', type=int, help="Number of cells to average over for diagnostics 1 or 2 values depending on --dim")
    parser.add_argument('--B0_Gauss', type=float, help="Upstream magnetic field strength in Gauss")
    parser.add_argument('--al_charge_state', type=int, help="Charge state of aluminum ions")
    parser.add_argument('--piecewise_degree', type=int, help="Degree of piecewise polynomial fit to apply to lineout data input profiles")
    args = parser.parse_args()
    print(args)

    assert Path(args.data_path).exists(), f"Data path does not exist: {args.data_path}"
    assert args.al_charge_state > 0 and args.al_charge_state <= 13, "Charge state of aluminum must be between 1 and 13"
    if args.piecewise_degree:
        assert args.piecewise_degree > 0, "Piecewise polynomial degree must be a positive integer"
    
    if args.dx == "ion_debye":
        pass
    else:
        try:
            float(args.dx)
        except ValueError:
            raise ValueError("dx must be a float or 'ion_debye'")
    
    print(f"Writing MagShockZ input deck from FLASH data: {args.data_path},\n start point: {args.start_point},\n end point: {args.end_point},\n input file name: {args.inputfile_name},\n rqm factor: {args.rqm_factor},\n")
    main(args)