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
from pathlib import Path
from typing import Dict, List, Union, Optional

# Third-party imports
import numpy as np
import yt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the path to the yt plugin
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
        Reference density in cm^-3. OSIRIS will set this to be a density of 1.
    rqm_factor : int, optional
        Factor by which the real Z_i/m_i will be divided. Default is 10.
    osiris_dims : int, optional
        Dimensions for OSIRIS simulation (1 or 2). Default is 2.
    start_point : List[float], optional
        Starting point coordinates in REAL units. Only used for 1D simulations.
    theta : float, optional
        Angle that ray makes with the x axis in radians. Default is π/2. Only used for 1D simulations.
    xmax : float, optional
        Maximum x-coordinate in REAL units. In 1D, this is the maximum distance from the start point.
        In 2D, this is a list of two floats representing xmin and xmax.
    ymax : float, optional
        Maximum y-coordinate in REAL units. In 1D, this is the maximum distance from the start point.
        In 2D, this is a list of two floats representing ymin and ymax.
    species_rqms : Dict[str, float], optional
        Dictionary mapping species names to their mass ratios.
    algorithm : str, optional
        Algorithm to use for OSIRIS simulation. Default is "cpu".
    resolution : int, optional
        Level of refinement used for the FLASH data. Default is 1.
        If write times are too long, change to zero. If data looks too coarse, change to 2.
    normalizations_override : Dict[str, float], optional
        Dictionary of normalization factors that will override defaults.
    """
    def __init__(self, 
                FLASH_data: str, 
                inputfile_name: str, 
                reference_density: float,
                rqm_factor: int = 10,
                osiris_dims: int = 2,
                start_point: List[float] = None, 
                theta: float = np.pi/2, 
                xmax: float = None,
                ymax: float = None,
                species_rqms: Dict[str, int] = None,
                ion_mass_thresholds: list = None,
                resolution: int = 1,
                algorithm: str = "cpu",
                normalizations_override: Dict[str, float] = {}):
        """Initialize the FLASH-OSIRIS interface."""
        # Validate inputs
        if osiris_dims not in [1, 2]:
            raise ValueError("osiris_dims must be either 1 or 2")
            
        if algorithm not in ["cpu", "cuda", "tiles"]:
            raise ValueError("algorithm must be either 'cpu', 'cuda', or 'tiles'")

        if not isinstance(reference_density, (int, float)):
            raise TypeError("reference_density must be a number")
        
        # Store basic parameters
        self.osiris_dims = osiris_dims
        self.FLASH_data = Path(FLASH_data)
        self.inputfile_name = inputfile_name + f".{self.osiris_dims}d"
        self.rqm_factor = rqm_factor
        self.n0 = reference_density

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
            self.xmax = np.array(xmax)
            self.ymax = np.array(ymax)
        elif osiris_dims == 3:
            raise NotImplementedError("3D simulations are not supported yet")
        
        self.species_rqms = species_rqms
        self.normalizations = None
        self.algorithm = algorithm
        self.normalizations_override = normalizations_override
 

        # Physical constants
        self.e = 4.80320425e-10 # [stacC] = [cm^3/2⋅g^1/2⋅s^−1]
        self.m_e = 9.1093837139e-28  # [g]
        self.c = 2.99792458e10  # [cm/s]
        

        # Validate FLASH data file exists
        if not self.FLASH_data.exists():
            raise FileNotFoundError(f"FLASH data file not found: {self.FLASH_data}")
        
        # Set up project directories
        self.proj_dir = Path.cwd() # TODO: Make this configurable
        self.output_dir = self.proj_dir / "input_files" / self.inputfile_name
        
        # Calculate plasma frequency
        self.omega_pe = np.sqrt(4 * np.pi * self.n0 * self.e**2 / self.m_e)  # in rad/s
        
        # Load FLASH data
        logger.info(f"Loading FLASH data from {self.FLASH_data}")
        self.ds = yt.load_for_osiris(self.FLASH_data)
        
        self._derive_fields() # TODO: create necessary fields and ion species

        # Get covering grid
        self.dims = self.ds.domain_dimensions * self.ds.refine_by**resolution
        self.all_data = self.ds.covering_grid(
            resolution,
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1, # Necessary when we take curl of magnetic field to get currents
        )

        
        # Log initialization parameters
        self._log_initialization_params()
        self._calculate_numbers()
    def _derive_fields(self):
        return

    def _log_initialization_params(self):
        """Log initialization parameters."""
        logger.info("=" * 50)
        logger.info("INITIALIZING FLASH-OSIRIS INTERFACE")
        logger.info(f"FLASH data: {self.FLASH_data}")
        logger.info(f"Input file name: {self.inputfile_name}")
        logger.info(f"Reference density: {self.n0} cm^-3")
        logger.info(f"species_rqms: {self.species_rqms}")
        logger.info(f"all rqms will be divided by {self.rqm_factor}")
        logger.info(f"OSIRIS dimensions: {self.osiris_dims}")
        if self.osiris_dims == 1:
            logger.info(f"Start point: {self.start_point} [c/wpe]")
            logger.info(f"Angle: {self.theta} (only used in 1D)")
            logger.info(f"Xmax: {self.xmax} (only used in 1D)")
        elif self.osiris_dims == 2:
            logger.info(f"Xmax: {self.xmax}")
            logger.info(f"Ymax: {self.ymax}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("INITIALIZING FLASH-OSIRIS INTERFACE COMPLETE")
        logger.info("=" * 50)

    def __str__(self):
        """String representation of the FLASH-OSIRIS interface."""
        output = "=" * 50 + "\n"
        output += "FLASH-OSIRIS INTERFACE\n"
        output += f"FLASH data: {self.FLASH_data}\n"
        output += f"Input file name: {self.inputfile_name}\n"
        output += f"Reference density: {self.n0} cm^-3\n"
        output += f"species_rqms: {self.species_rqms}\n"
        output += f"all rqms will be divided by {self.rqm_factor}\n"
        output += f"OSIRIS dimensions: {self.osiris_dims}\n"
        if self.osiris_dims == 1:
            output += (f"Start point: {self.start_point} [c/wpe]\n")
            output += (f"Angle: {self.theta} (only used in 1D)\n")
            output += (f"Xmax: {self.xmax} (only used in 1D)\n")
        elif self.osiris_dims == 2:
            output += (f"Xmax: {self.xmax}\n")
            output += (f"Ymax: {self.ymax}\n")
        output += f"Output directory: {self.output_dir}\n"
        output += "=" * 50
        return output


    def _calculate_numbers(self):
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

        self.xmax = self.xmax * self.omega_pe / self.c
        if self.osiris_dims == 2:
            self.ymax = self.ymax * self.omega_pe / self.c
        
        if self.osiris_dims == 1:
            self.start_point = [self.start_point[0] * self.omega_pe / self.c,
                                self.start_point[1] * self.omega_pe / self.c]

        # Calculate Debye length in OSIRIS units

        # Get real mass ratio for reference
        rqm_real = 1836 / self.all_data['flash', 'ye']
        
        # Create normalizations dictionary
        self._create_normalizations()
        

    
    def _create_normalizations(self):
        """Create dictionary of normalization factors for different fields."""
        # Base normalization factors
        B_norm = (self.omega_pe * self.m_e * self.c) / self.e
        E_norm = B_norm * self.c 
        v_norm = self.c
        vth_ele_norm = np.sqrt(self.m_e * self.c**2)
        
        self.normalizations = {
            # Density normalizations
            'edens': self.n0,
            
            # Magnetic field normalizations
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
    
    def _save_slices(self):
        """
        Process and save field data slices for OSIRIS.

        Note:
            - Density data is output as a numpy array because OSIRIS uses its own interpolator
            - Other fields are saved as pickle files with interpolators
            - Thermal velocities require special handling because we need to take sqrt of temperature
        """
        # Create output directory
        self.interp_dir = self.output_dir / "interp"
        if not self.interp_dir.exists():
            self.interp_dir.mkdir(parents=True)
            
        # Process each field
        for field, normalization in self.normalizations.items():
            # TODO Unsure if I should treat normalizations_override as an additional factor or a replacement
            if field in self.normalizations_override:
                logger.info(f"{field} normalization of {normalization} is overriden to be {np.format_float_scientific(self.normalizations_override[field],3)}")
                normalization = self.normalizations_override[field]
            else:
                logger.info(f"Processing {field} with normalization {np.format_float_scientific(normalization, 3)}")
            
            # Special handling for thermal velocity fields
            if field.startswith('vth'):
                self._save_thermal_field(field, normalization)
                continue
                
            # Process regular fields
            self._save_regular_field(field, normalization)
    
    def _save_thermal_field(self, field, normalization, normal_axis="z"):
        """Save thermal velocity field data."""
        from pickle import dump
        from scipy.interpolate import RegularGridInterpolator
        """
         Parameters
        ----------
        normal_axis : str, optional
            Axis normal to the slice plane ('x', 'y', or 'z'). Default is 'z'.
        """

        # Choose the appropriate temperature field
        if field == 'vthele':
            temp_field = 'tele'
        else:
            temp_field = 'tion'
        

        axis_map = {"x": 0, "y": 1, "z": 2}
        if normal_axis not in axis_map:
            raise ValueError("normal_axis must be one of 'x', 'y', or 'z'")

        normal = axis_map[normal_axis]
        
        # Get the middle index of the chosen axis
        chunk_size = 128  # Adjust based on memory constraints
        middle_index = self.dims[normal] // 2

        # Initialize field data array
        if normal_axis == "x":
            field_data = np.zeros(self.all_data['flash', temp_field][middle_index, :, :].shape)
        elif normal_axis == "y":
            field_data = np.zeros(self.all_data['flash', temp_field][:, middle_index, :].shape)
        elif normal_axis == "z":
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
        
        with open(f"{self.interp_dir}/{field}.pkl", "wb") as f:
            dump(interp, f)
    
    def _save_regular_field(self, field, normalization, normal_axis = 'z'):
        """Save regular field data."""
        from pickle import dump
        from scipy.interpolate import RegularGridInterpolator

        axis_map = {"x": 0, "y": 1, "z": 2}
        if normal_axis not in axis_map:
            raise ValueError("normal_axis must be one of 'x', 'y', or 'z'")

        normal = axis_map[normal_axis]
        
        # Get the middle index of the chosen axis
        chunk_size = 128  # Adjust based on memory constraints
        middle_index = self.dims[normal] // 2

       
        # Initialize field data array
        if normal_axis == "x":
            field_data = np.zeros(self.all_data['flash', field][middle_index, :, :].shape)
        elif normal_axis == "y":
            field_data = np.zeros(self.all_data['flash', field][:, middle_index, :].shape)
        elif normal_axis == "z":
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
            np.save(f"{self.interp_dir}/{field}.npy", field_data)
        else:
            # Create and save interpolator for other fields
            interp = RegularGridInterpolator(
                (self.y, self.x), field_data.T,
                method='linear', bounds_error=True, fill_value=0
            )
            
            with open(f"{self.interp_dir}/{field}.pkl", "wb") as f:
                dump(interp, f)

    def write_input_file(self):
        """
        Generate and write OSIRIS input file based on processed data.
        
        Reads thermal velocity bounds and generates the input file
        with appropriate parameters for all species.
        """
        # Calculate the endpoint position
        end_point = self._calculate_endpoint()
        
        # Read thermal velocity bounds for all species
        thermal_bounds = self._read_thermal_bounds(end_point)
        
        # Write the actual input file
        input_file_path = self.output_dir / self.inputfile_name
        logger.info(f"Writing OSIRIS input file to {input_file_path}")
        
        with open(input_file_path, "w") as f:
            f.write(self._generate_input_file_content(thermal_bounds))
        
        logger.info(f"OSIRIS input file written successfully")
    
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

        from pickle import load

        if self.osiris_dims == 1: 
            bounds = {
            'electron': None,
            'ions': {}
            }
            # Read electron thermal velocity bounds
            with open(self.output_dir / "interp/vthele.pkl", "rb") as f:
                vthele = load(f)
                bounds['electron'] = [
                    vthele((self.start_point[1], self.start_point[0])),
                    vthele((end_point[1], end_point[0]))
                ]
                logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")
            
            # Read ion thermal velocity bounds for each species
            for ion in self.species_rqms:
                with open(self.output_dir / f"interp/vth{ion}.pkl", "rb") as f:
                    vthion = load(f)
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
                    vthele = load(f)
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
                    vthion = load(f)
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
    
    def _generate_input_file_content(self, thermal_bounds):
        """
        Generate the content for the OSIRIS input file.
        
        Parameters
        ----------
        thermal_bounds : dict
            Thermal velocity bounds for all species
            
        Returns
        -------
        str
            Complete OSIRIS input file content
        """
        if self.osiris_dims == 1:
            # Header and simulation parameters
            content = self._generate_header()
            content += self._generate_simulation_params1D()
            content += self._generate_timespace_params()
            content += self._generate_field_params()
            
            # Species-specific sections
            content += self._generate_electrons_section1D(thermal_bounds['electron'])
            
            # Generate sections for each ion species
            for ion, bounds in thermal_bounds['ions'].items():
                content += self._generate_ion_section1D(ion, bounds)
        elif self.osiris_dims == 2:
            # Header and simulation parameters
            content = self._generate_header()
            content += self._generate_simulation_params2D()
            content += self._generate_timespace_params()
            content += self._generate_field_params()
            
            # Species-specific sections
            content += self._generate_electrons_section2D(thermal_bounds['electron'])
            
            # Generate sections for each ion species
            for ion, bounds in thermal_bounds['ions'].items():
                content += self._generate_ion_section2D(ion, bounds)    
        # Footer
        elif self.osiris_dims == 3:
            raise NotImplementedError("3D OSIRIS input file generation is not implemented yet.")
        content += """
!---------- end of osiris input file -------------"""
        
        return content
    
    def _generate_header(self):
        """Generate the header section of the input file."""
        return f'''!----------------- Input deck for FLASH-OSIRIS interface using python ------------------

! Before this simulation is able to run, you will need to edit certain values in the file that are marked <edit>.
! To run this input deck as is, put the input deck, OSIRIS executable, and the
! py-script-{self.osiris_dims}d.py file all in the same directory.  Next, do `export PYTHONPATH=.` to set the Python
! path to the directory that contains the py-script-{self.osiris_dims}d.py file (current directory). Finally,
! execute `./osiris-{self.osiris_dims}D.e {self.inputfile_name}` to run the simulation, which will use the
! py-script-{self.osiris_dims}d.py and interp.npy files to set various field and particle data.
!-----------------------------------------------------------------------------------------
'''

    def _generate_simulation_params1D(self):
        """Generate simulation parameters section."""
        if self.algorithm == 'cpu': 
            return f'''
!----------global simulation parameters----------
simulation 
\u007b
\u007d

!--------the node configuration for this simulation--------
node_conf 
\u007b
 node_number = <edit>, ! number of nodes you are using
 n_threads= <edit>, ! number of threads per node
\u007d

!----------spatial grid----------
grid
\u007b
 nx_p = <edit>, ! number of cells in x-direction
\u007d

'''
        elif self.algorithm == 'cuda' or self.algorithm == 'tiles':
            return f'''
!----------global simulation parameters----------
simulation
\u007b
 algorithm = "{self.algorithm}",
\u007d

node_conf
\u007b
 node_number = <edit>, ! number of GPUs
 n_threads = <edit>, ! number of threads per GPU
 tile_number = <edit>, ! n_tiles_x should be greater than n_cells_x / 1024 and be a power of two. Refer to osiris cuda documentation
\u007d

!----------spatial grid----------
grid
\u007b
 nx_p = <edit>, ! number of cells in x-direction
\u007d
'''
        
    def _generate_simulation_params2D(self):
        """Generate simulation parameters section."""
        if self.algorithm == 'cpu': 
            return f'''
!----------global simulation parameters----------
simulation 
\u007b
\u007d

!--------the node configuration for this simulation--------
node_conf 
\u007b
 node_number = <edit>,<edit>, ! edit this to the number of nodes you are using. total number of nodes is node_number_x * node_number_y
 n_threads=<edit>,
\u007d

!----------spatial grid----------
grid
\u007b
 nx_p(1:2) = <edit>, <edit>, ! number of cells in x-direction
\u007d

'''
        elif self.algorithm == 'cuda' or self.algorithm == 'tiles':
            return f'''
!----------global simulation parameters----------
simulation
\u007b
 algorithm = "{self.algorithm}",
\u007d

node_conf
\u007b
 node_number = <edit>, ! number of GPUs you are using
 tile_number(1:2) = <edit>, <edit>, ! n_tiles_x * n_tiles_y should be greater than n_cells_x * n_cells_y / 1024 and be a power of two. Refer to osiris cuda documentation
\u007d

!----------spatial grid----------
grid
\u007b
 nx_p(1:2) = <edit>, <edit>, ! number of cells in x-direction
\u007d
'''
    def _generate_timespace_params(self):
        """Generate grid parameters section."""
        result = f'''
!----------time step and global data dump timestep number----------
time_step
\u007b
 dt     = <edit>, ! time step in wpe^-1
 ndump  = <edit>, ! number of time steps between data dumps,
\u007d

!----------restart information----------
restart
\u007b
\u007d
'''
        if self.osiris_dims == 1:
            result += f'''
!----------spatial limits of the simulations----------
space
\u007b
 ! This is euclidean distance, not the span in any one direction
 ! Start point in {self.osiris_dims}D plane is specified in py-script-{self.osiris_dims}d
 xmin = 0, ! This should always be == 0
 xmax = {self.xmax},
\u007d
'''
        elif self.osiris_dims == 2:
            result += f'''
!----------spatial limits of the simulations----------
space
\u007b
 xmin(1:2) = {self.xmax[0]}, {self.ymax[0]},
 xmax(1:2) = {self.xmax[1]}, {self.ymax[1]},
\u007d
'''
        result += f'''
!----------time limits ----------
time
\u007b
 tmin = 0.0,
 tmax = <edit>,
\u007d
'''
        return result
    def _generate_field_params(self):
        """Generate field parameters section."""
        result = f'''
!----------field solver set up----------
el_mag_fld
\u007b
  ! Set two of the field components with the Python script
  ! Note, you need to set PYTHONPATH in the console to the folder containing py-script-{self.osiris_dims}d.py
  type_init_b(1:3) = "python", "python", "python",
  type_init_e(1:3) = "python", "python", "python",
  init_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
  init_py_func = "set_fld_int", ! Name of function in the Python file to call (same for all components)
'''
        if self.osiris_dims == 1:
            result += f'''
 ! You can also do this with external fields, as functions of time
\u007d
!----------boundary conditions for em-fields ----------
emf_bound
\u007b
 type(1:2,1) = <edit>, <edit>, ! field boundary conditions 
\u007d

!----------- electro-magnetic field diagnostics ---------
diag_emf
\u007b
  reports = 
    "b1", "b2", "b3",
    "e1", "e2", "e3",
\u007d
'''
        elif self.osiris_dims == 2:
            result += f'''
 ! You can also do this with external fields, as functions of time
\u007d
!----------boundary conditions for em-fields ----------
emf_bound
\u007b
 type(1:2,1) =   <edit>, <edit>, ! edit the field boundary conditions
 type(1:2,2) =   <edit>, <edit>,
\u007d

!----------- electro-magnetic field diagnostics ---------
diag_emf
\u007b
 reports = 
   "b1", "b2", "b3",
   "e1", "e2", "e3",
\u007d
'''
        return result
    
    def _generate_electrons_section1D(self, thermal_bounds):
        """Generate the section for electrons."""
        vthele_start = thermal_bounds[0]
        vthele_end = thermal_bounds[1]
        
        return f'''
!----------number of particle species----------
particles
\u007b
  interpolation = "quadratic", ! edit interpolation method as needed
  num_species = {len(self.species_rqms) + 1},
\u007d

!----------information for electrons----------
species
\u007b
 name = "electrons",
 rqm = -1.0,
 num_par_x = <edit>, ! number of particles per cell 
 init_type = "python",
\u007d

!----------inital proper velocities - electrons-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 uth_py_func = "set_uth_e", ! Name of function in the Python file to call
 
 ! use_spatial_ufl = .true.,
 ufl_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 ufl_py_func = "set_ufl_e", ! Name of function in the Python file to call
\u007d

!----------density profile for electrons----------
profile
\u007b
 py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 py_func = "set_density_e", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for electrons----------
spe_bound
\u007b
 type(1:2,1) = "thermal","thermal", ! Default is thermal boundary conditions. Feel free to change
 uth_bnd(1:3,1,1) = {np.format_float_scientific(vthele_start,4)}, {np.format_float_scientific(vthele_start,4)}, {np.format_float_scientific(vthele_start,4)}, 
 uth_bnd(1:3,2,1) = {np.format_float_scientific(vthele_end,4)}, {np.format_float_scientific(vthele_end,4)}, {np.format_float_scientific(vthele_end,4)}, 
\u007d

!----------diagnostic for electrons----------
diag_species
\u007b
 ndump_fac = 1,
 reports = "charge", <edit>, ! edit the reports as needed
 rep_udist = <edit>,
 ndump_fac_pha = 1,
 ps_pmin(1:3) = <edit>, <edit>, <edit>,
 ps_pmax(1:3) = <edit>, <edit>, <edit>,
 ps_xmin(1:1) = 0.0,
 ps_xmax(1:1) = {self.xmax},
 ps_np = <edit>,
 ps_nx = <edit>,
 phasespaces = "p1x1", "p2x1","p3x1",
\u007d
'''
 
    def _generate_electrons_section2D(self, thermal_bounds):
        """Generate the section for electrons in 2D"""
        
        return f'''
!----------number of particle species----------
particles
\u007b
  interpolation = "quadratic",
  num_species = {len(self.species_rqms) + 1},
\u007d

!----------information for electrons----------
species
\u007b
 name = "electrons",
 rqm=-1.0,
 num_par_x(1:2) = <edit>, <edit>, ! number of particles per cell in x and y directions
 init_type = "python",
\u007d

!----------inital proper velocities - electrons-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 uth_py_func = "set_uth_e", ! Name of function in the Python file to call
 
 ! use_spatial_ufl = .true.,
 ufl_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 ufl_py_func = "set_ufl_e", ! Name of function in the Python file to call
\u007d

!----------density profile for electrons----------
profile
\u007b
 py_mod = "py-script-2d", ! Name of Python file
 py_func = "set_density_e", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for electrons----------
spe_bound
\u007b
 ! Default is thermal boundary conditions, but change as needed
 type(1:2,1) = "thermal","thermal",
 type(1:2,2) = "thermal","thermal",
 uth_bnd(1:3,1,1) = {np.format_float_scientific(thermal_bounds['x'][0],4)}, {np.format_float_scientific(thermal_bounds['x'][0],4)}, {np.format_float_scientific(thermal_bounds['x'][0],4)}, 
 uth_bnd(1:3,2,1) = {np.format_float_scientific(thermal_bounds['x'][1],4)}, {np.format_float_scientific(thermal_bounds['x'][1],4)}, {np.format_float_scientific(thermal_bounds['x'][1],4)}, 
 uth_bnd(1:3,1,2) = {np.format_float_scientific(thermal_bounds['y'][0],4)}, {np.format_float_scientific(thermal_bounds['y'][0],4)}, {np.format_float_scientific(thermal_bounds['y'][0],4)}, 
 uth_bnd(1:3,2,2) = {np.format_float_scientific(thermal_bounds['y'][1],4)}, {np.format_float_scientific(thermal_bounds['y'][1],4)}, {np.format_float_scientific(thermal_bounds['y'][1],4)}, 
\u007d

!----------diagnostic for electrons----------
diag_species
\u007b
 ndump_fac = 1,
 reports = <edit>,
 rep_udist = <edit>, 
 ndump_fac_pha = 1,
 ps_pmin(1:3) = <edit>, <edit>, <edit>,
 ps_pmax(1:3) = <edit>, <edit>, <edit>,
 ps_xmin(1:2) = {self.xmax[0]}, {self.ymax[0]}, ! phase space covers the entire domain. change as needed
 ps_xmax(1:2) = {self.xmax[1]}, {self.ymax[1]},
 ps_np = <edit>,
 ps_nx(1:2) = <edit>, <edit>,
 phasespaces = <edit>,
\u007d
'''

    def _generate_ion_section1D(self, ion, thermal_bounds):
        """Generate the section for a specific ion species."""
        vthion_start = thermal_bounds[0]
        vthion_end = thermal_bounds[1]
        
        return f'''
!----------information for {ion} ions----------
species
\u007b
 name = "{ion}",
 rqm = {self.species_rqms[ion] / self.rqm_factor},
 num_par_x = <edit>,
 init_type = "python",
\u007d

!----------inital proper velocities - {ion} ions-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 uth_py_func = "set_uth_{ion}", ! Name of function in the Python file to call
 
 ! use_spatial_ufl = .true.,
 ufl_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 ufl_py_func = "set_ufl_i", ! Name of function in the Python file to call
\u007d

!----------density profile for {ion} ions----------
profile
\u007b
 py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 py_func = "set_density_{ion}", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for {ion} ions----------
spe_bound
\u007b
 type(1:2,1) =   "thermal","thermal",
 uth_bnd(1:3,1,1) = {np.format_float_scientific(vthion_start,4)}, {np.format_float_scientific(vthion_start,4)}, {np.format_float_scientific(vthion_start,4)}, 
 uth_bnd(1:3,2,1) = {np.format_float_scientific(vthion_end,4)}, {np.format_float_scientific(vthion_end,4)}, {np.format_float_scientific(vthion_end,4)}, 
\u007d

!----------diagnostic for {ion} ions----------
diag_species
\u007b
 ndump_fac = 1,
 reports = <edit>,
 rep_udist = <edit>, 
 ndump_fac_pha = 1,
 ps_pmin(1:3) = <edit>, <edit>, <edit>, 
 ps_pmax(1:3) = <edit>, <edit>, <edit>, 
 ps_xmin(1:1) = 0.0,
 ps_xmax(1:1) = {self.xmax},
 ps_np = <edit>,
 ps_nx = <edit>,
 phasespaces = <edit>,
\u007d'''
    
    def _generate_ion_section2D(self, ion, thermal_bounds):
        """Generate the section for a specific ion species."""
        
        return f'''
!----------information for {ion} ions----------
species
\u007b
 name = "{ion}",
 rqm = {self.species_rqms[ion] / self.rqm_factor},
 num_par_x(1:2) = {int(np.sqrt(self.ppc))}, {int(np.sqrt(self.ppc))},
 init_type = "python",
\u007d

!----------inital proper velocities - {ion} ions-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 uth_py_func = "set_uth_{ion}", ! Name of function in the Python file to call
 
 ! use_spatial_ufl = .true.,
 ufl_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 ufl_py_func = "set_ufl_i", ! Name of function in the Python file to call
\u007d

!----------density profile for {ion} ions----------
profile
\u007b
 py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 py_func = "set_density_{ion}", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for {ion} ions----------
spe_bound
\u007b
 ! Default is thermal boundary conditions, but change as needed
 type(1:2,1) = "thermal","thermal",
 type(1:2,2) = "thermal","thermal",
 uth_bnd(1:3,1,1) = {np.format_float_scientific(thermal_bounds['x'][0],4)}, {np.format_float_scientific(thermal_bounds['x'][0],4)}, {np.format_float_scientific(thermal_bounds['x'][0],4)}, 
 uth_bnd(1:3,2,1) = {np.format_float_scientific(thermal_bounds['x'][1],4)}, {np.format_float_scientific(thermal_bounds['x'][1],4)}, {np.format_float_scientific(thermal_bounds['x'][1],4)}, 
 uth_bnd(1:3,1,2) = {np.format_float_scientific(thermal_bounds['y'][0],4)}, {np.format_float_scientific(thermal_bounds['y'][0],4)}, {np.format_float_scientific(thermal_bounds['y'][0],4)}, 
 uth_bnd(1:3,2,2) = {np.format_float_scientific(thermal_bounds['y'][1],4)}, {np.format_float_scientific(thermal_bounds['y'][1],4)}, {np.format_float_scientific(thermal_bounds['y'][1],4)}, 
\u007d

!----------diagnostic for {ion} ions----------
diag_species
\u007b
 ndump_fac = 1,
 reports = <edit>,
 rep_udist = <edit>, 
 ndump_fac_pha = 1,
 ps_pmin(1:3) = <edit>, <edit>, <edit>, 
 ps_pmax(1:3) = <edit>, <edit>, <edit>,
 ps_xmin(1:2) = {self.xmax[0]}, {self.ymax[0]},
 ps_xmax(1:2) = {self.xmax[1]}, {self.ymax[1]},
 ps_np = <edit>,
 ps_nx(1:2) = <edit>, <edit>,
 phasespaces = <edit>,
\u007d
'''

    def write_python_script1D(self):
        # Write the python script to generate the input file
        with open(f'{self.output_dir}/py-script-{self.osiris_dims}d.py', "w") as f:
            f.write(f'''import numpy as np
import pickle

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------

# Define the start point for the ray in OSIRIS units
start_point = {self.start_point} # start point in OSIRIS units
theta = {self.theta} # angle that ray makes with the x axis [radians]

# Parameters of FLASH simulation
box_bounds = \u007b
    "xmin": {int(self.x[0].value)},
    "xmax": {int(self.x[-1].value)},
    "ymin": {int(self.y[0].value)},
    "ymax": {int(self.y[-1].value)},
\u007d

def set_fld_int( STATE ):
    """
    Function to set the field data in the STATE dictionary based on the field component.
    """
    
    # Parameters:
    # STATE (dict): Dictionary containing the state information, including field component and positional boundary data.

    # print("calling set_fld...")
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    print(f'x_bnd: \u007b x_bnd \u007d')

    # Shape of the data array
    nx = STATE["data"].shape

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace(x_bnd[0][0] * np.cos(theta), x_bnd[0][1] * np.cos(theta), nx[0], endpoint=True) + start_point[0]
    y = np.linspace(x_bnd[0][0] * np.sin(theta), x_bnd[0][1] * np.sin(theta), nx[0], endpoint=True) + start_point[1]

    # Dictionary to map field components to their respective filenames and operations
    field_map = \u007b 
        "e1": ("interp/Ex.pkl", "interp/Ey.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex.pkl", "interp/Ey.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez.pkl", None, lambda Ez, _: Ez),
        "b1": ("interp/Bx_int.pkl", "interp/By_int.pkl", lambda Bx, By: np.cos(theta) * Bx + np.sin(theta) * By),
        "b2": ("interp/Bx_int.pkl", "interp/By_int.pkl", lambda Bx, By: -np.sin(theta) * Bx + np.cos(theta) * By),
        "b3": ("interp/Bz_int.pkl", None, lambda Bz, _: Bz)
    \u007d

    # Determine the filenames and operation based on the field component
    filename1, filename2, operation = field_map.get(STATE['fld'], (None, None, None))

    if filename1:
        with open(filename1, "rb") as f:
            field1 = pickle.load(f)
        field2 = None
        if filename2:
            with open(filename2, "rb") as f:
                field2 = pickle.load(f)
        STATE["data"] = operation(field1((y, x)), field2((y, x)) if field2 else None)

    return

#-----------------------------------------------------------------------------------------
def set_fld_ext( STATE ):
    """
    Function to set external field data, STATE dictionary based on the field component.
    It seems like osiris will make repeated calls to this if it is used and the external fields are set to "dynamic" avoid that by setting fields to "static"
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including field component and positional boundary data.
    """
    # print("calling set_fld...")
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    print(f'x_bnd: \u007b x_bnd \u007d')

    # Shape of the data array
    nx = STATE["data"].shape

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace(x_bnd[0][0] * np.cos(theta), x_bnd[0][1] * np.cos(theta), nx[0], endpoint=True) + start_point[0]
    y = np.linspace(x_bnd[0][0] * np.sin(theta), x_bnd[0][1] * np.sin(theta), nx[0], endpoint=True) + start_point[1]

    # Dictionary to map field components to their respective filenames and operations
    field_map = \u007b 
        "e1": ("interp/Ex_ext.pkl", "interp/Ey_ext.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex_ext.pkl", "interp/Ey_ext.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez_ext.pkl", None, lambda Ez, _: Ez),
        "b1": ("interp/Bx_ext.pkl", "interp/By_ext.pkl", lambda Bx, By: np.cos(theta) * Bx + np.sin(theta) * By),
        "b2": ("interp/Bx_ext.pkl", "interp/By_ext.pkl", lambda Bx, By: -np.sin(theta) * Bx + np.cos(theta) * By),
        "b3": ("interp/Bz_ext.pkl", None, lambda Bz, _: Bz)
    \u007d

    # Determine the filenames and operation based on the field component
    filename1, filename2, operation = field_map.get(STATE['fld'], (None, None, None))

    if filename1:
        with open(filename1, "rb") as f:
            field1 = pickle.load(f)
        field2 = None
        if filename2:
            with open(filename2, "rb") as f:
                field2 = pickle.load(f)
        STATE["data"] = operation(field1((y, x)), field2((y, x)) if field2 else None)

    return



#-----------------------------------------------------------------------------------------

def set_uth_e(STATE):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """
    # Load the electron thermal velocity interpolator if not already loaded
    if "vthele" not in STATE:
        with open('interp/vthele.pkl', "rb") as f:
            STATE['vthele'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    # Define a chunk size for processing
    chunk_size = 1024

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE['vthele']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE['vthele']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE['vthele']((y_positions, x_positions))

    return
#-----------------------------------------------------------------------------------------''' + '\n'.join(f'''
def set_uth_{ion}( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """
    if f"vth{ion}" not in STATE.keys():
        with open(f'interp/vth{ion}.pkl', "rb") as f:
            STATE[f'vth{ion}'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE[f'vth{ion}']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE[f'vth{ion}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{ion}']((y_positions, x_positions))
    return
#-----------------------------------------------------------------------------------------''' for ion in self.species_rqms.keys()) + f'''

#-----------------------------------------------------------------------------------------
def set_ufl_e( STATE ):
    # print("calling set_ufl...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    with open("interp/v_ex.pkl", "rb") as f:
        velx = pickle.load(f)
    with open("interp/v_ey.pkl", "rb") as f:
        vely = pickle.load(f)
    with open("interp/v_ez.pkl", "rb") as f:
        velz = pickle.load(f)
        
    # Calculate positions
    x_positions = start_point[0] + np.cos(theta) * STATE["x"][:, 0]
    y_positions = start_point[1] + np.sin(theta) * STATE["x"][:, 0]

    # Set ufl_x1
    STATE["u"][:, 0] = (
        np.cos(theta) * velx((y_positions, x_positions)) +
        np.sin(theta) * vely((y_positions, x_positions))
    )

    # Set ufl_x2
    STATE["u"][:, 1] = (
        -np.sin(theta) * velx((y_positions, x_positions)) +
        np.cos(theta) * vely((y_positions, x_positions))
    )

    # Set ufl_x3
    STATE["u"][:, 2] = velz((y_positions, x_positions))

    return
#-----------------------------------------------------------------------------------------

def set_ufl_i( STATE ):
    # print("calling set_ufl...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    with open("interp/v_ix.pkl", "rb") as f:
        velx = pickle.load(f)
    with open("interp/v_iy.pkl", "rb") as f:
        vely = pickle.load(f)
    with open("interp/v_iz.pkl", "rb") as f:
        velz = pickle.load(f)
        
    # Calculate positions
    x_positions = start_point[0] + np.cos(theta) * STATE["x"][:, 0]
    y_positions = start_point[1] + np.sin(theta) * STATE["x"][:, 0]

    # Set ufl_x1
    STATE["u"][:, 0] = (
        np.cos(theta) * velx((y_positions, x_positions)) +
        np.sin(theta) * vely((y_positions, x_positions))
    )

    # Set ufl_x2
    STATE["u"][:, 1] = (
        -np.sin(theta) * velx((y_positions, x_positions)) +
        np.cos(theta) * vely((y_positions, x_positions))
    )

    # Set ufl_x3
    STATE["u"][:, 2] = velz((y_positions, x_positions))

    return

#-----------------------------------------------------------------------------------------
def load_and_interpolate_density(STATE, filename):
    """
    Helper function to load interpolator from a file and set the density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including positional boundary data.
    filename (str): Path to the file containing the interpolator.
    """

    # Free up a little bit of memory
    if "fld" in STATE.keys():
        del STATE["fld"]
    density_grid = np.load(filename)

    STATE["nx"] = np.array([4096])
    STATE["xmin"] = np.array([0.0])
    STATE["xmax"] = np.array([{int(self.xmax * 1.02)}]) # a little more than the final distance specified in input file

    from scipy.interpolate import RegularGridInterpolator
    loaded_interpolator = RegularGridInterpolator((np.linspace(box_bounds["xmin"], box_bounds['xmax'], density_grid.shape[0]), 
                                                   np.linspace(box_bounds['ymin'], box_bounds['ymax'], density_grid.shape[1])), 
                                                   density_grid, bounds_error=True, fill_value=None)

    x = np.linspace(STATE['xmin'][0]*np.cos(theta), STATE['xmax'][0]*np.cos(theta), STATE['nx'][0], endpoint=True ) + start_point[0]
    y = np.linspace(STATE['xmin'][0]*np.sin(theta), STATE['xmax'][0]*np.sin(theta), STATE['nx'][0], endpoint=True ) + start_point[1]
    
    # print(loaded_interpolator((x, y)).shape)
    STATE["data"] = loaded_interpolator((x, y)) # This one is reversed because it does not come pre-interpolated

    return

#-----------------------------------------------------------------------------------------
def set_density_e( STATE ):
    """
    Set the electron density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print("setting ELECTRON DENSITY...")

    load_and_interpolate_density(STATE, "interp/edens.npy")

#-----------------------------------------------------------------------------------------'''+'\n'.join(f'''

def set_density_{ion}( STATE ):
    """
    Set the {ion} density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting {str.upper(ion)} DENSITY...")
    load_and_interpolate_density(STATE, f"interp/{ion}dens.npy")

#-----------------------------------------------------------------------------------------'                   
''' for ion in self.species_rqms.keys()))
    def write_python_script2D(self):
        # Write the python script to generate the input file
        with open(f'{self.output_dir}/py-script-{self.osiris_dims}d.py', "w") as f:
            f.write(f'''
import numpy as np
import pickle


# Define bounds of box in osiris units, ensure that this is larger than the bounds specified in input file
box_bounds = \u007b
    "xmin": {self.x[0].value}, 
    "xmax": {self.x[-1].value},
    "ymin": {self.y[0].value},
    "ymax": {self.y[-1].value},
\u007d

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------
def set_fld_int( STATE ):
    """
    Function to set the field data in the STATE dictionary based on the field component.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including field component and positional boundary data.
    """
    # print("calling set_fld...")
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    # print(f"x_bnd = \u007b x_bnd \u007d")

    # Shape of the data array
    nx = STATE["data"].shape
    # print(f"nx = \u007b nx \u007d")

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' )

    # Determine the filename based on the field component
    match STATE['fld']:
        case "e1":
            filename = "interp/Ex.pkl"
        case "e2":
            filename = "interp/Ey.pkl"
        case "e3":
            filename = "interp/Ez.pkl"
        case "b1":
            filename = "interp/Bx_int.pkl"
        case "b2":
            filename = "interp/By_int.pkl"
        case "b3":
            filename = "interp/Bz_int.pkl"

    with open(filename, "rb") as f:
        loaded_interpolator = pickle.load(f)

    STATE["data"] = loaded_interpolator((X2, X1))


#-----------------------------------------------------------------------------------------
def set_fld_ext( STATE ):
    # print("calling set_fld_ext...")
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]

    # Time (in case fields are dynamic)

    # Could make decisions based on field component
    # match STATE['fld']:
    #     case "e1":
    #         filename = "interp/Ex.pkl"
    #     case "e2":
    #         filename = "interp/Ey.pkl"
    #     case "e3":
    #         filename = "interp/Ez.pkl"
    #     case "b1":
    #         filename = "interp/magx.pkl"
    #     case "b2":
    #         filename = "interp/magy.pkl"
    #     case "b3":
    #         filename = "interp/magz.pkl"

    # Create x arrays that indicate the position (remember indexing order is reversed)
    nx = STATE["data"].shape
    x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' ) # Matches Fortran array indexing

    # # Perform some function to fill in the field values based on the coordinates
    # with open(filename, "rb") as f:
    #     loaded_interpolator = pickle.load(f)

    # STATE["data"] = loaded_interpolator((X2, X1))



#-----------------------------------------------------------------------------------------

def set_uth_e( STATE ):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """
    # print("calling set_uth_e...")
    if "vthele" not in STATE.keys():
        with open('interp/vthele.pkl', "rb") as f:
            STATE['vthele'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))

    return

#-----------------------------------------------------------------------------------------
def set_ufl_e( STATE ):
    # print("calling set_ufl_e...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

        # Set ufl_x1
    with open("interp/v_ex.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,0] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    # Set ufl_x2
    with open("interp/v_ey.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,1] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0]))

        # Set ufl_x3
    with open("interp/v_ez.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,2] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    return

#-----------------------------------------------------------------------------------------
def set_ufl_i( STATE ):
    # print("calling set_ufl_i...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

        # Set ufl_x1
    with open("interp/v_ix.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,0] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    # Set ufl_x2
    with open("interp/v_iy.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,1] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0]))

        # Set ufl_x3
    with open("interp/v_iz.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,2] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    return



#-----------------------------------------------------------------------------------------
def set_density_e( STATE ):
    """
    Set the electron density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print("setting ELECTRON DENSITY...")

    load_and_interpolate_density(STATE, "interp/edens.npy")

#-----------------------------------------------------------------------------------------''' + '\n'.join(f'''
def set_uth_{ion}( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """

    if "vth{ion}" not in STATE.keys():
        with open(f'interp/vth{ion}.pkl', "rb") as f:
            STATE[f'vth{ion}'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE[f'vth{ion}']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE[f'vth{ion}']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE[f'vth{ion}']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
    return
#-----------------------------------------------------------------------------------------''' for ion in self.species_rqms.keys()) + f'''


#-----------------------------------------------------------------------------------------

def load_and_interpolate_density(STATE, filename):
    """
    Helper function to load interpolator from a file and set the density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including positional boundary data.
    filename (str): Path to the file containing the interpolator.
    """
    # Free up a little bit of memory
    # print(STATE.keys())
    if "fld" in STATE.keys():
        del STATE["fld"]
    density_grid = np.load(filename)

    STATE["nx"] = np.array(density_grid.shape)//2
    STATE["xmin"] = np.array([{self.x[0].value}, {self.y[0].value}])
    STATE["xmax"] = np.array([{self.x[-1].value},{self.y[-1].value}])
    STATE['data'] = density_grid[::2,::2].T # For memory constraints, we only use every second point in the grid

    return

#-----------------------------------------------------------------------------------------
def set_density_e( STATE ):
    """
    Set the electron density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print("setting ELECTRON DENSITY...")

    load_and_interpolate_density(STATE, "interp/edens.npy")

#-----------------------------------------------------------------------------------------'''+'\n'.join(f'''

def set_density_{ion}( STATE ):
    """
    Set the {ion} density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting {str.upper(ion)} DENSITY...")
    load_and_interpolate_density(STATE, f"interp/{ion}dens.npy")

#-----------------------------------------------------------------------------------------'                   
''' for ion in self.species_rqms.keys()))

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
        self._save_slices()
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
        FLASH_data="/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_11-2024/MagShockZ_hdf5_chk_0006",
        inputfile_name="magshockz-TEST",
        osiris_dims=2,
        reference_density=5e18,
        xmax = [-1000,2500],
        ymax = [300,6000],
        species_rqms= {'aluminum': 6000, 'silicon' : 3800, 'steel': 18000}
    )
    interface.write_everything()
    interface.show_box_in_plane('edens')
    interface.plot_2D_lineouts('edens')