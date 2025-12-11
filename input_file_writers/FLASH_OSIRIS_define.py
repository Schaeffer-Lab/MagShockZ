"""
FLASH-OSIRIS Interface

This package provides tools to process 3D FLASH simulation data, apply user-defined edits and normalization, 
and generate input files for OSIRIS simulations.

It needs my yt plugin in order to work... If you are a person who isn't David and you are reading this,
you probably don't have it. Email me at dschneidinger@g.ucla.edu

Author: David Schneidinger
Version: 0.1.0
"""

# Standard library imports
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Union, Optional

# Third-party imports
import numpy as np
import yt
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
E_CHARGE = 4.80320425e-10  # [statC] = [cm^3/2⋅g^1/2⋅s^−1]
ELECTRON_MASS = 9.1093837139e-28  # [g]
SPEED_OF_LIGHT = 2.99792458e10  # [cm/s]
KB = 8.617e-5  # eV/K
ERGS_PER_EV = 1.602e-12  # erg/eV

# Add the path to the yt plugin
sys.path.append("../src")
yt.enable_plugins()

class FLASH_OSIRIS:
    """
    Main class for converting FLASH simulation data to OSIRIS input files.
    
    Parameters
    ----------
    FLASH_data : str
        Path to the FLASH data file.
    inputfile_name : str
        Name of the OSIRIS input file to generate.
    reference_density : float
        Reference density in cm^-3.
    B_background : float, optional
        Background magnetic field strength in Gauss. This field will be applied externally in OSIRIS. Default is 0.
    rqm_factor : int, optional
        Factor by which the real mass ratios will be divided. Default is 10.
    osiris_dims : int, optional
        Dimensions for OSIRIS simulation (1 or 2). Default is 2.
    ppc : int, optional
        Particles per cell. Default is 80.
    start_point : List[float], optional
        Starting point coordinates in OSIRIS units. Default is [0, 240].
    theta : float, optional
        Angle that ray makes with the x axis in radians. Default is π/2.
    xmax : float, optional
        Maximum x-coordinate in OSIRIS units. Default is 7100.
    species_rqms : Dict[str, float], optional
        Dictionary mapping species names to their mass ratios.
    dx_ndebye : float, optional
        Spatial resolution in units of upstream electron debye lengths. Default is 7.14.
    tmax_gyroperiods : int, optional
        Number of upstream ion gyroperiods to simulate. Default is 10.
    algorithm : str, optional
        Algorithm to use for OSIRIS simulation. Default is "cpu".
    normalizations_override : Dict[str, float], optional
        Dictionary of normalization factors to multiply existing default factor by.
        Note: this factor will be multiply the normalization. For example, if you set ['b2_ext': 10],
        the external magnetic field in y will be 0.1 times its default value.
    """
    def __init__(self, 
                path_to_FLASH_data: str, 
                desired_inputfile_name: str, 
                reference_density_cc: float,
                B_background: float = 0.0,
                rqm_normalization_factor: int = 10,
                osiris_dims: int = 2,
                ppc: int = 80,
                start_point: List[float] = [0, 240], 
                theta: float = np.pi/2, 
                xmax: float = 7100,
                ymax: float = None,
                species_rqms: Dict[str, int] = None,
                dx: float = 7.14,
                tmax_gyroperiods: int = 10,
                algorithm: str = "cpu",
                normalizations_override: Dict[str, float] = {}):
        """Initialize the FLASH-OSIRIS interface."""
        # Validate inputs
        if osiris_dims not in [1, 2]:
            raise ValueError("osiris_dims must be either 1 or 2")
            
        if species_rqms is None:
            species_rqms = {"al": 7257, "si": 3899}

        if algorithm not in ["cpu", "cuda", "tiles"]:
            raise ValueError("algorithm must be either 'cpu', 'cuda', or 'tiles'")

        if not isinstance(reference_density_cc, (int, float)):
            raise TypeError("reference_density_cc must be a number")
        
        # Store basic parameters
        self.osiris_dims = osiris_dims
        self.FLASH_data = Path(path_to_FLASH_data)
        self.inputfile_name = desired_inputfile_name + f".{self.osiris_dims}d"
        self.B0 = B_background
        self.rqm_factor = rqm_normalization_factor
        self.n0 = reference_density_cc
        self.ppc = ppc

        if osiris_dims == 1:
            if not isinstance(xmax, (int, float)):
                raise TypeError("xmax must be a number for 1D simulations")
            self.xmax = xmax  # TODO: Make this configurable based on data
            if not isinstance(start_point, (list, tuple)) or len(start_point) != 2:
                raise ValueError("start_point must be a list or tuple of two numbers for 1D simulations")
            self.start_point = start_point  # TODO: Make this configurable based on data
            if not isinstance(theta, (int, float)):
                raise TypeError("theta must be a number for 1D simulations")
            self.theta = theta  # TODO: Make this configurable based on data
        elif osiris_dims == 2:
            if not isinstance(xmax, (list, tuple)) or len(xmax) != 2:
                raise ValueError("xmax must be a list of two numbers for 2D simulations")
            if not isinstance(ymax, (list, tuple)) or len(ymax) != 2:
                raise ValueError("ymax must be a list of two numbers for 2D simulations")
            self.xmax = xmax
            self.ymax = ymax
        elif osiris_dims == 3:
            raise NotImplementedError("3D simulations are not supported yet")
        
        self.species_rqms = species_rqms
        self.normalizations = None
        self.gyrotime = None
        self.dx_ndebye = dx
        self.tmax_gyroperiods = tmax_gyroperiods
        self.algorithm = algorithm
        self.normalizations_override = normalizations_override
        
        # Physical constants (moved to module level)
        self.e = E_CHARGE
        self.m_e = ELECTRON_MASS
        self.c = SPEED_OF_LIGHT
        
        # Validate FLASH data file exists
        if not self.FLASH_data.exists():
            raise FileNotFoundError(f"FLASH data file not found: {self.FLASH_data}")
        
        # Set up project directories
        self.proj_dir = Path("/home/dschneidinger/MagShockZ")
        self.output_dir = self.proj_dir / "input_files" / self.inputfile_name
        
        # Calculate plasma frequency
        self.omega_pe = np.sqrt(4 * np.pi * self.n0 * self.e**2 / self.m_e)  # in rad/s
        
        # Load FLASH data
        logger.info(f"Loading FLASH data from {self.FLASH_data}")
        self.ds = yt.load_for_osiris(self.FLASH_data, rqm_factor=self.rqm_factor) # would be much cleaner if this just used yt.load
        
        # Get covering grid
        level = 2 # If write times are long, change this to 0. If data looks too coarse, change this to 2.
        self.dims = self.ds.domain_dimensions * self.ds.refine_by**level
        self.all_data = self.ds.covering_grid(
            level,
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1,
        )

        
        # Log initialization parameters
        self._log_initialization_params()
        self.calculate_numbers()
    def _log_initialization_params(self):
        """Log initialization parameters."""
        logger.info("\n" + str(self))

    def __str__(self):
        """String representation of the FLASH-OSIRIS interface."""
        lines = [
            "=" * 50,
            "FLASH-OSIRIS INTERFACE",
            f"FLASH data: {self.FLASH_data}",
            f"Input file: {self.inputfile_name}",
            f"Reference density: {self.n0:.2e} cm^-3",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"Background B-field: {self.B0} G",
            f"OSIRIS dimensions: {self.osiris_dims}D",
            f"Particles per cell: {self.ppc}",
        ]
        
        if self.osiris_dims == 1:
            lines.extend([
                f"Start point: {self.start_point} [c/ωpe]",
                f"Ray angle: {self.theta:.4f} rad",
                f"Xmax: {self.xmax} [c/ωpe]"
            ])
        elif self.osiris_dims == 2:
            lines.extend([
                f"X range: {self.xmax} [c/ωpe]",
                f"Y range: {self.ymax} [c/ωpe]"
            ])
            
        lines.extend([
            f"Output directory: {self.output_dir}",
            "=" * 50
        ])
        
        return "\n".join(lines)


    def calculate_numbers(self):
        """
        Calculate parameters needed for OSIRIS simulation.
        
        Parameters
        ----------
        n_gyroperiods : int, optional
            Number of gyroperiods to simulate. Default is 10.
        n_debye : float, optional
            Number of Debye lengths per cell. Default is 7.14.
        """
        # Convert coordinate arrays to OSIRIS units
        self.x = self.all_data['flash', 'x'][:, 0, 0] * self.omega_pe / self.c
        self.y = self.all_data['flash', 'y'][0, :, 0] * self.omega_pe / self.c
        self.z = self.all_data['flash', 'z'][0, 0, :] * self.omega_pe / self.c
        
        # Calculate Debye length in OSIRIS units
        self.debye_osiris = np.sqrt(
            self.all_data['flash', 'tele'][-1, -1, 0] * KB * ERGS_PER_EV / (self.m_e * self.c**2) #TODO: there must be a better way to pick the background temperature
        )
        
        logger.info(f"Debye length: {self.debye_osiris.value} osiris units")
        logger.info(f"Background temperature: {round(self.all_data['flash', 'tele'][-1, -1, 0].value * KB)} eV")
        
        # Calculate spatial and temporal resolution
        self.dx = self.debye_osiris * self.dx_ndebye
        self.dt = self.dx * 0.98 / np.sqrt(self.osiris_dims) # CFL condition
        # Get real mass ratio for reference
        rqm_real = 1836 / self.all_data['flash', 'ye'][-1, -1, 0]
        logger.info(f"{'*'*10} real mass ratio: {rqm_real} {'*'*10}")
        
        # Create normalizations dictionary
        self._create_normalizations()
        
        # Calculate gyrotime and simulation duration
        self._calculate_tmax()
    
    def _create_normalizations(self):
        """Create dictionary of normalization factors for different fields."""
        # Base normalization factors
        B_norm = (self.omega_pe * self.m_e * self.c) / self.e
        E_norm = B_norm * self.c / np.sqrt(self.rqm_factor)
        v_norm = self.c / np.sqrt(self.rqm_factor)
        vth_ele_norm = np.sqrt(self.m_e * self.c**2)
        
        self.normalizations = {
            # Density normalizations
            'edens': self.n0,
            
            # Magnetic field normalizations
            'Bx_int': B_norm, 'By_int': B_norm, 'Bz_int': B_norm,
            'magx': B_norm, 'magy': B_norm, 'magz': B_norm,
            
            # Electric field normalizations
            'Ex': E_norm, 'Ey': E_norm, 'Ez': E_norm,
            
            # Velocity normalizations
            'v_ix': v_norm, 'v_iy': v_norm, 'v_iz': v_norm,
            'v_ex': v_norm, 'v_ey': v_norm, 'v_ez': v_norm,
            
            # Thermal velocity for electrons
            'vthele': vth_ele_norm,
        }
        
        # Add species-specific normalizations
        for species, rqms in self.species_rqms.items():
            self.normalizations[species + 'dens'] = self.n0
            self.normalizations[f'vth{species}'] = vth_ele_norm * np.sqrt(rqms / self.rqm_factor)
    
    def _calculate_tmax(self):
        """Calculate gyrotime and simulation duration."""
        if self.B0 != 0:
            self.gyrotime = (self.species_rqms['al']/ self.rqm_factor) / (self.B0 / self.normalizations['Bx_int'])
        else:
            self.gyrotime = self.species_rqms['al']/ self.rqm_factor / (self.all_data['flash', 'magx'][-1, -1, 0] / self.normalizations['magx'])

        self.tmax = int(self.gyrotime * self.tmax_gyroperiods)

    def save_slices(self, normal_axis="z"):
        """
        Process and save field data slices for OSIRIS.
        
        Parameters
        ----------
        normal_axis : str, optional
            Axis normal to the slice plane ('x', 'y', or 'z'). Default is 'z'. # TODO allow for other normal axes
            
        Note:
            - Density data is output as a numpy array because OSIRIS uses its own interpolator
            - Other fields are saved as pickle files with interpolators
            - Thermal velocities require special handling because we need to take sqrt of temperature
        """
        import pickle
        from scipy.interpolate import RegularGridInterpolator
        
        # Create output directory
        interp_dir = self.output_dir / "interp"
        if not interp_dir.exists():
            interp_dir.mkdir(parents=True)
            
        # Validate normal_axis
        axis_map = {"x": 0, "y": 1, "z": 2}
        if normal_axis not in axis_map.keys():
            raise ValueError("normal_axis must be one of 'x', 'y', or 'z'")

        normal = axis_map[normal_axis]
        
        # Get the middle index of the chosen axis
        middle_index = self.dims[normal] // 2
        chunk_size = 128  # Adjust based on memory constraints
        
        # Process each field
        for field, normalization in self.normalizations.items():
            if field in self.normalizations_override.keys():
                normalization = normalization * self.normalizations_override[field]
                logger.info(f"{field} is normalized by additional factor of {np.format_float_scientific(self.normalizations_override[field],3)}")
            logger.info(f"Processing {field} with normalization {np.format_float_scientific(normalization, 3)}")
            
            # Special handling for thermal velocity fields
            if field.startswith('vth'):
                self._save_thermal_field(field, normalization, middle_index, chunk_size, interp_dir)
                continue
                
            # Process regular fields
            self._save_regular_field(field, normalization, middle_index, chunk_size, interp_dir)
    
    def _save_thermal_field(self, field, normalization, middle_index, chunk_size, interp_dir):
        """Save thermal velocity field data."""
        import pickle
        from scipy.interpolate import RegularGridInterpolator
        
        # Choose the appropriate temperature field
        if field == 'vthele':
            temp_field = 'tele'
        else:
            temp_field = 'tion'
        
        # Initialize field data array
        field_data = np.zeros(self.all_data['flash', temp_field][:, :, middle_index].shape)
        
        # Process data in chunks to save memory
        for i in range(0, self.all_data['flash', temp_field].shape[0], chunk_size):
            end = min(i + chunk_size, self.all_data['flash', temp_field].shape[0])
            field_data_chunk = np.array(
                np.sqrt(self.all_data['flash', temp_field][i:end, :, middle_index] * yt.units.kb_cgs)
            ) / normalization
            field_data[i:end, :] = field_data_chunk
        
        # Create and save interpolator
        interp = RegularGridInterpolator(
            (self.y, self.x), field_data.T, 
            method='linear', bounds_error=True, fill_value=0
        )
        
        with open(f"{interp_dir}/{field}.pkl", "wb") as f:
            pickle.dump(interp, f)
    
    def _save_regular_field(self, field, normalization, middle_index, chunk_size, interp_dir):
        """Save regular field data."""
        import pickle
        from scipy.interpolate import RegularGridInterpolator
        
        # Initialize field data array
        field_data = np.zeros(self.all_data['flash', field][:, :, middle_index].shape)
        
        # Process data in chunks to save memory
        for i in range(0, self.all_data['flash', field].shape[0], chunk_size):
            end = min(i + chunk_size, self.all_data['flash', field].shape[0])
            field_data_chunk = np.array(
                self.all_data['flash', field][i:end, :, middle_index]
            ) / normalization
            field_data[i:end, :] = field_data_chunk
        
        # Special handling for density fields
        if field.endswith('dens'):
            # Remove small density values
            lower_bound = 0.0001
            field_data[field_data < lower_bound] = 0
            np.save(f"{interp_dir}/{field}.npy", field_data)
        else:
            # Create and save interpolator for other fields
            interp = RegularGridInterpolator(
                (self.y, self.x), field_data.T,
                method='linear', bounds_error=True, fill_value=0
            )
            
            with open(f"{interp_dir}/{field}.pkl", "wb") as f:
                pickle.dump(interp, f)

    def write_input_file(self):
        """
        Generate and write OSIRIS input file using Jinja2 templates.
        
        Reads thermal velocity bounds and generates the input file
        with appropriate parameters for all species.
        """
        # Calculate the endpoint position
        end_point = self._calculate_endpoint()
        
        # Read thermal velocity bounds for all species
        thermal_bounds = self._read_thermal_bounds(end_point)
        
        # Prepare template context
        context = self._prepare_template_context(thermal_bounds)
        
        # Load and render template
        template_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('OSIRIS_TEMPLATE.jinja')
        content = template.render(**context)
        
        # Write the actual input file
        input_file_path = self.output_dir / self.inputfile_name
        logger.info(f"Writing OSIRIS input file to {input_file_path}")
        
        with open(input_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"OSIRIS input file written successfully")
    
    def _prepare_template_context(self, thermal_bounds):
        """Prepare the context dictionary for Jinja2 template rendering."""
        # Calculate grid dimensions
        if self.osiris_dims == 1:
            nx = int(self.xmax / self.dx)
            ny = None
            xmin, xmax = 0, self.xmax
            ymin, ymax = None, None
        else:
            nx = int((self.xmax[1] - self.xmax[0]) / self.dx)
            ny = int((self.ymax[1] - self.ymax[0]) / self.dx)
            xmin, xmax = self.xmax[0], self.xmax[1]
            ymin, ymax = self.ymax[0], self.ymax[1]
        
        # Calculate external B-field components
        if self.osiris_dims == 1:
            ext_b0_x = np.format_float_scientific(
                np.cos(self.theta) * self.B0 / self.normalizations['Bx_int'], 4
            )
            ext_b0_y = np.format_float_scientific(
                -np.sin(self.theta) * self.B0 / self.normalizations['Bx_int'], 4
            )
        else:
            ext_b0_x = np.format_float_scientific(
                self.B0 / self.normalizations['Bx_int'], 4
            )
            ext_b0_y = 0
        
        # Calculate tile numbers for GPU algorithms
        tile_numbers = self._calculate_tile_numbers() if self.algorithm in ["cuda", "tiles"] else []
        
        # Prepare species list
        species_list = self._prepare_species_list(thermal_bounds)
        
        # Build context dictionary
        context = {
            'dims': self.osiris_dims,
            'inputfile_name': self.inputfile_name,
            'algorithm': self.algorithm,
            'nx': nx,
            'ny': ny,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'dt': np.format_float_scientific(self.dt, 4),
            'ndump': int(self.tmax / (400 * self.dt)),
            'tmax': self.tmax,
            'ext_b0_x': ext_b0_x,
            'ext_b0_y': ext_b0_y,
            'tile_numbers': tile_numbers,
            'num_species': len(self.species_rqms) + 1,
            'species_list': species_list,
        }
        
        return context
    
    def _calculate_tile_numbers(self):
        """Calculate tile numbers for GPU algorithms."""
        if self.osiris_dims == 1:
            n_tiles_min = self.xmax / self.dx / 1024
            i = 0
            while True:
                n_tiles_x = 2**i
                if n_tiles_x > n_tiles_min:
                    break
                i += 1
            return [n_tiles_x]
        else:
            n_tiles_min = (self.xmax[1] - self.xmax[0]) * (self.ymax[1] - self.ymax[0]) / self.dx**2 / 1024
            i = j = 0
            while True:
                n_tiles_x = 2**i
                n_tiles_y = 2**j
                if n_tiles_x * n_tiles_y > n_tiles_min:
                    break
                i += 1
                j += 1
            return [n_tiles_x, n_tiles_y]
    
    def _prepare_species_list(self, thermal_bounds):
        """Prepare species data for template rendering."""
        species_list = []
        
        # Add electrons
        electron_config = self._get_species_config('electrons', thermal_bounds['electron'], is_electron=True)
        species_list.append(electron_config)
        
        # Add ions
        for ion, bounds in thermal_bounds['ions'].items():
            ion_config = self._get_species_config(ion, bounds, is_electron=False)
            species_list.append(ion_config)
        
        return species_list
    
    def _get_species_config(self, species_name, thermal_bounds, is_electron=False):
        """Get configuration dictionary for a single species."""
        config = {
            'name': species_name,
            'short_name': 'e' if is_electron else species_name,
            'fluid_func': 'e' if is_electron else 'i',
            'rqm': -1.0 if is_electron else self.species_rqms[species_name] / self.rqm_factor,
            'ps_pmin': [-0.1, -0.1, -0.02] if is_electron else [-0.05, -0.05, -0.02],
            'ps_pmax': [0.1, 0.1, 0.02] if is_electron else [0.05, 0.05, 0.02],
        }
        
        # Particles per cell configuration
        if self.osiris_dims == 1:
            config['ppc'] = self.ppc
        else:
            ppc_per_dim = int(np.sqrt(self.ppc))
            config['ppc_x'] = ppc_per_dim
            config['ppc_y'] = ppc_per_dim
        
        # Thermal velocity bounds (dimension-dependent)
        if self.osiris_dims == 1:
            vth_start, vth_end = thermal_bounds
            config['vth_start'] = np.format_float_scientific(vth_start, 4)
            config['vth_end'] = np.format_float_scientific(vth_end, 4)
        else:
            config['vth_x_start'] = np.format_float_scientific(thermal_bounds['x'][0], 4)
            config['vth_x_end'] = np.format_float_scientific(thermal_bounds['x'][1], 4)
            config['vth_y_start'] = np.format_float_scientific(thermal_bounds['y'][0], 4)
            config['vth_y_end'] = np.format_float_scientific(thermal_bounds['y'][1], 4)
        
        return config
    
    def _calculate_endpoint(self):
        """Calculate the endpoint position based on simulation parameters."""
        if self.osiris_dims == 1:
            end_x = self.xmax * np.cos(self.theta) + self.start_point[0]
            end_y = self.xmax * np.sin(self.theta) + self.start_point[1]
        elif self.osiris_dims == 2:
            end_x = self.xmax[-1]
            end_y = self.ymax[-1]
        return [end_x, end_y]
    
    def _read_thermal_bounds(self, end_point):
        """
        Read thermal velocity bounds for electrons and all ion species.
        
        Parameters
        ----------
        end_point : list
            Endpoint coordinates [x, y]
            
        Returns
        -------
        dict
            Thermal velocity bounds for electrons and all ion species
        """
        import pickle
        

        if self.osiris_dims == 1: 
            bounds = {
            'electron': None,
            'ions': {}
            }
            # Read electron thermal velocity bounds
            with open(self.output_dir / "interp/vthele.pkl", "rb") as f:
                vthele = pickle.load(f)
                bounds['electron'] = [
                    vthele((self.start_point[1], self.start_point[0])),
                    vthele((end_point[1], end_point[0]))
                ]
                logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")
            
            # Read ion thermal velocity bounds for each species
            for ion in self.species_rqms.keys():
                with open(self.output_dir / f"interp/vth{ion}.pkl", "rb") as f:
                    vthion = pickle.load(f)
                    bounds['ions'][ion] = [
                        vthion((self.start_point[1], self.start_point[0])),
                        vthion((end_point[1], end_point[0]))
                    ]
                    logger.info(f"{ion} thermal velocity bounds: {bounds['ions'][ion]}")
        elif self.osiris_dims == 2:             
            bounds = {
                'electron': {},
                'ions': {}
            }
            num_samples = 10  # Number of points to sample
            x_samples = np.linspace(self.xmax[0], self.xmax[1], num_samples)
            y_samples = np.linspace(self.ymax[0], self.ymax[1], num_samples)
            with open(self.output_dir / "interp/vthele.pkl", "rb") as f:
                    vthele = pickle.load(f)
                    x_lower_bound = np.mean([vthele((y, self.xmax[0])) for y in y_samples])
                    x_upper_bound = np.mean([vthele((y, self.xmax[1])) for y in y_samples])

                    y_lower_bound = np.mean([vthele((self.ymax[0], x)) for x in x_samples])
                    y_upper_bound = np.mean([vthele((self.ymax[1], x)) for x in x_samples])

                    bounds['electron']['x'] = [
                        x_lower_bound,
                        x_upper_bound
                    ]
                    bounds['electron']['y'] = [
                        y_lower_bound,
                        y_upper_bound
                    ]
                    logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")
                
                # Read ion thermal velocity bounds for each species
            for ion in self.species_rqms.keys():
                bounds['ions'][ion] = {}
                with open(self.output_dir / f"interp/vth{ion}.pkl", "rb") as f:
                    vthion = pickle.load(f)
                    x_lower_bound = np.mean([vthion((y, self.xmax[0])) for y in y_samples])
                    x_upper_bound = np.mean([vthion((y, self.xmax[1])) for y in y_samples])

                    y_lower_bound = np.mean([vthion((self.ymax[0], x)) for x in x_samples])
                    y_upper_bound = np.mean([vthion((self.ymax[1], x)) for x in x_samples])

                    bounds['ions'][ion]['x'] = [
                        x_lower_bound,
                        x_upper_bound
                    ]
                    bounds['ions'][ion]['y'] = [
                        y_lower_bound,
                        y_upper_bound
                    ]
                    logger.info(f"{ion} thermal velocity bounds: {bounds['ions'][ion]}")        
        return bounds
    
    def save_instance(self):
        """
        Save the instance to disk, excluding attributes that can't be pickled properly.
        """
        import pickle
        
        # Ensure output directory exists
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        try:
            # Store attributes that might cause pickling issues
            ds = self.ds
            all_data = self.all_data
            
            # Temporarily remove these attributes
            self.ds = None
            self.all_data = None
            
            # Save the instance to a file
            with open(self.output_dir / "instance.pkl", "wb") as f:
                pickle.dump(self, f)
            
            # Restore the attributes
            self.ds = ds
            self.all_data = all_data
            
            logger.info(f"Instance saved to {self.output_dir / 'instance.pkl'}")
        except Exception as e:
            # Ensure attributes are restored even if pickling fails
            if 'ds' in locals():
                self.ds = ds
            if 'all_data' in locals():
                self.all_data = all_data
            
            logger.error(f"Failed to save instance: {e}")
    def write_everything(self):
        # Main function to run the interface
        self.save_slices()
        self.write_input_file()
        if self.osiris_dims == 1:
            self.write_python_script1D()
        elif self.osiris_dims == 2:
            self.write_python_script2D()
        else:
            raise ValueError("osiris_dims must be 1 or 2")
        self.save_instance()

        print(f"Input file {self.inputfile_name} and python script input.py have been generated in {self.output_dir}")

    def show_lineout_in_plane(self, field):
        import matplotlib.pyplot as plt
        end_point = self._calculate_endpoint()
        if field.endswith('dens'):
            data = np.load(self.output_dir / f'interp/{field}.npy')

        else:
            import pickle
            with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                f = pickle.load(p)
                x1,x2 = np.meshgrid(self.y,self.x)
                data = f((x1,x2))

        plt.figure()

        plt.imshow(data.T, origin='lower',
                    extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],vmax = 10) # TODO add some logic for vmax

        plt.plot([self.start_point[0], end_point[0]], [self.start_point[1],end_point[1]],color='r')

        plt.show()

    def show_box_in_plane(self, field):
        import matplotlib.pyplot as plt
        if field.endswith('dens'):
            data = np.load(self.output_dir / f'interp/{field}.npy')

        else:
            import pickle
            with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                f = pickle.load(p)
                x1,x2 = np.meshgrid(self.y,self.x)
                data = f((x1,x2))

        plt.figure()

        plt.imshow(data.T, origin='lower',
                    extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],vmax = 10) # TODO add some logic for vmax

        plt.plot([self.xmax[0],self.xmax[-1]],[self.ymax[0],self.ymax[0]],color='r')
        plt.plot([self.xmax[0], self.xmax[-1]], [self.ymax[-1],self.ymax[-1]],color='r')
        plt.plot([self.xmax[0], self.xmax[0]], [self.ymax[0],self.ymax[-1]],color='r')
        plt.plot([self.xmax[-1], self.xmax[-1]], [self.ymax[0],self.ymax[-1]],color='r')

        plt.show()

    def plot_1D_lineout(self, fields):
        from scipy.interpolate import RegularGridInterpolator
        import matplotlib.pyplot as plt
        import pickle
        end_point = self._calculate_endpoint()
        x_osiris = np.linspace(self.start_point[0], end_point[0],1000)
        y_osiris = np.linspace(self.start_point[1], end_point[1],1000)
            
        results = {}

        # Check if fields is a string (single field) or an iterable
        if isinstance(fields, str):
            fields_to_plot = [fields]  # Convert to a list with a single element
        else:
            fields_to_plot = fields  # Use as is
        plt.figure()
        for field in fields_to_plot:

            if field.endswith('dens'):
                data = np.load(self.output_dir / f'interp/{field}.npy')
                f = RegularGridInterpolator(
                    (self.y, self.x), data.T, 
                    method='linear', bounds_error=True, fill_value=0)
            else:
                with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                    f = pickle.load(p)

            plt.plot(np.linspace(0, self.xmax, 1000), f((y_osiris,x_osiris)).T, label=field)
            results[field] = (f((y_osiris,x_osiris)).T)
        plt.xlabel(r'x [$c/\omega_{pe}$]')

        plt.grid(visible=True) 
        plt.legend()
        plt.show()
        return results
    
    def plot_2D_lineouts(self, fields):
        from scipy.interpolate import RegularGridInterpolator
        import matplotlib.pyplot as plt
        import pickle
        
        # Plot the boundaries
        boundary_segments = [
            ('Bottom_edge', self.xmax[0], self.xmax[-1], self.ymax[0], self.ymax[0]),
            ('Top_edge', self.xmax[0], self.xmax[-1], self.ymax[-1], self.ymax[-1]),
            ('Left_edge', self.xmax[0], self.xmax[0], self.ymax[0], self.ymax[-1]),
            ('Right_edge', self.xmax[-1], self.xmax[-1], self.ymax[0], self.ymax[-1])
        ]
                 
        results = {}


        # Check if fields is a string (single field) or an iterable
        if isinstance(fields, str):
            fields_to_plot = [fields]  # Convert to a list with a single element
        else:
            fields_to_plot = fields  # Use as is
        plt.figure()
        for edge, xmin, xmax, ymin, ymax in boundary_segments:
            results[edge] = {}
            x_osiris = np.linspace(xmin, xmax,1000)
            y_osiris = np.linspace(ymin, ymax,1000)
            for field in fields_to_plot:

                if field.endswith('dens'):
                    data = np.load(self.output_dir / f'interp/{field}.npy')
                    f = RegularGridInterpolator(
                        (self.y, self.x), data.T, 
                        method='linear', bounds_error=True, fill_value=0)
                else:
                    with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                        f = pickle.load(p)
                if xmax == xmin:
                    plt.plot(np.linspace(ymin, ymax, 1000), f((y_osiris,x_osiris)).T, 'r-', linewidth=2, label=field)
                    plt.xlabel('Y [c/ωpe]')
                elif ymax == ymin:
                    plt.plot(np.linspace(xmin, xmax, 1000), f((y_osiris,x_osiris)).T, 'r-', linewidth=2, label=field)
                    plt.xlabel('X [c/ωpe]')

                results[edge][field] = f((y_osiris,x_osiris)).T
            
            plt.grid(visible=True) 
            plt.legend()
            plt.title(f'Fields on {edge}')
            plt.show()
        return results


if __name__ == "__main__":
    # Example usage
    interface = FLASH_OSIRIS(
        path_to_FLASH_data=FLASH_data="/home/dschneidinger/shared/data/VAC_DEREK3D_20um/MagShockZ_hdf5_chk_0006",
        desired_inputfile_name=="magshockz-v3.2",
        osiris_dims=2,
        reference_density=5e18,
        ppc=16,
        xmax = [-1000,2500],
        ymax = [300,6000],
        dx_ndebye=200,
        B_background=75000,
    )
    interface.write_everything()
    interface.show_box_in_plane('edens')
    interface.plot_2D_lineouts('edens')