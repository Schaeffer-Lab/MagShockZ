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
    """
    def __init__(self, 
                FLASH_data: str, 
                inputfile_name: str, 
                reference_density: float,
                B_background: float = 0, 
                rqm_factor: int = 10,
                osiris_dims: int = 2,
                ppc: int = 80,
                start_point: List[float] = [0, 240], 
                theta: float = np.pi/2, 
                xmax: float = 7100,
                species_rqms: Dict[str, int] = None,
                ion_mass_thresholds: list = [28,35],
                dx_ndebye: float = 7.14,
                tmax_gyroperiods: int = 10):
        """Initialize the FLASH-OSIRIS interface."""
        # Validate inputs
        if osiris_dims not in [1, 2]:
            raise ValueError("osiris_dims must be either 1 or 2")
            
        if species_rqms is None:
            species_rqms = {"channel": 3810, "sheathe": 6802, "background": 7257, "si": 3899}
        
        # Store basic parameters
        self.FLASH_data = Path(FLASH_data)
        self.inputfile_name = inputfile_name
        self.B0 = B_background
        self.rqm_factor = rqm_factor
        self.n0 = reference_density
        self.osiris_dims = osiris_dims
        self.ppc = ppc
        self.xmax = xmax  # TODO: Make this configurable based on data
        self.start_point = start_point  # TODO: Make this configurable based on data
        self.theta = theta  # TODO: Make this configurable based on data
        self.species_rqms = species_rqms
        self.normalizations = None
        self.gyrotime = None
        self.dx_ndebye = dx_ndebye
        self.tmax_gyroperiods = tmax_gyroperiods
        
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
        self.ds = yt.load_for_osiris(self.FLASH_data, rqm_factor=self.rqm_factor, B_background=self.B0, ion_mass_thresholds=ion_mass_thresholds)
        
        # Get covering grid
        level = 0
        self.dims = self.ds.domain_dimensions * self.ds.refine_by**level
        self.all_data = self.ds.covering_grid(
            level,
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1,
        )

        
        # Log initialization parameters
        self._log_initialization()
        self.calculate_numbers()
    
    def _log_initialization(self):
        """Log initialization parameters."""
        logger.info("=" * 50)
        logger.info("INITIALIZING FLASH-OSIRIS INTERFACE")
        logger.info(f"FLASH data: {self.FLASH_data}")
        logger.info(f"Input file name: {self.inputfile_name}")
        logger.info(f"Reference density: {self.n0} cm^-3")
        logger.info(f"species_rqms: {self.species_rqms}")
        logger.info(f"all rqms will be divided by {self.rqm_factor}")
        logger.info(f"External background magnetic field: {self.B0} Gauss")
        logger.info(f"OSIRIS dimensions: {self.osiris_dims}")
        logger.info(f"Particles per cell: {self.ppc}")
        logger.info(f"Start point: {self.start_point} [c/wpe]")
        logger.info(f"Angle: {self.theta} (only used in 1D)")
        logger.info(f"Xmax: {self.xmax} (only used in 1D)")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("INITIALIZING FLASH-OSIRIS INTERFACE COMPLETE")
        logger.info("=" * 50)


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
            self.all_data['flash', 'tele'][-1, -1, 0] * KB * ERGS_PER_EV / (self.m_e * self.c**2)
        )
        
        logger.info(f"Debye length: {self.debye_osiris.value} osiris units")
        logger.info(f"Background temperature: {round(self.all_data['flash', 'tele'][-1, -1, 0].value * KB)} eV")
        
        # Calculate spatial and temporal resolution
        self.dx = self.debye_osiris * self.dx_ndebye
        self.dt = self.dx * 0.98  # CFL condition
        
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
            self.gyrotime = (self.species_rqms['background']/ self.rqm_factor) / (self.B0 / self.normalizations['Bx_int'])
        else:
            self.gyrotime = self.species_rqms['background']/ self.rqm_factor / (self.all_data['flash', 'magx'][-1, -1, 0] / self.normalizations['magx'])
        
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
        end_x = self.xmax * np.cos(self.theta) + self.start_point[0]
        end_y = self.xmax * np.sin(self.theta) + self.start_point[1]
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
        
        bounds = {
            'electron': None,
            'ions': {}
        }
        
        try:
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
                    
        except FileNotFoundError as e:
            logger.error(f"Could not read thermal velocity files: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading thermal velocity data: {e}")
            raise
            
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
        # Header and simulation parameters
        content = self._generate_header()
        content += self._generate_simulation_params()
        content += self._generate_grid_params()
        content += self._generate_field_params()
        
        # Species-specific sections
        content += self._generate_electrons_section(thermal_bounds['electron'])
        
        # Generate sections for each ion species
        for ion, bounds in thermal_bounds['ions'].items():
            content += self._generate_ion_section(ion, bounds)
        
        # Footer
        content += """
!----------diagnostic for currents----------
diag_current
{
 !ndump_fac = 1,
 !reports = "j1", "j2", "j3" , 
}

!---------- end of osiris input file -------------"""
        
        return content
    
    def _generate_header(self):
        """Generate the header section of the input file."""
        return f'''!----------------- Input deck illustrating the Python-Fortran interface ------------------
! To run this input deck as is, first put the input deck, OSIRIS executable, and the
! py-script-{self.osiris_dims}d.py file all in the same directory.  Next, do `export PYTHONPATH=.` to set the Python
! path to the directory that contains the py-script-{self.osiris_dims}d.py file (current directory). Finally,
! execute `./osiris-{self.osiris_dims}D.e {self.inputfile_name}` to run the simulation, which will use the
! py-script-{self.osiris_dims}d.py and interp.npy files to set various field and particle data.
!-----------------------------------------------------------------------------------------
'''

    def _generate_simulation_params(self):
        """Generate simulation parameters section."""
        return '''
!----------global simulation parameters----------
simulation 
{
 parallel_io = "mpi",
}

!--------the node configuration for this simulation--------
node_conf 
{
 node_number = 16, ! edit this to the number of nodes you are using
 n_threads=2,
}
'''

    def _generate_grid_params(self):
        """Generate grid parameters section."""
        return f'''
!----------spatial grid----------
grid
\u007b
 nx_p = {int(self.xmax/self.dx)}, ! number of cells in x-direction
\u007d

!----------time step and global data dump timestep number----------
time_step
\u007b
  dt     =   {np.format_float_scientific(self.dt,4)},
  ndump  =   {int(self.tmax / (400 * self.dt))},
\u007d

!----------restart information----------
restart
\u007b
  ndump_fac = -1,
  ndump_time = 3500, !once/hour
  if_restart = .false.,
  if_remold = .true.,
\u007d

!----------spatial limits of the simulations----------
space
\u007b
  ! This is euclidean distance, not the span in y direction
  ! Start point in 2D plane is specified in py-script-1d
  xmin =  0, ! This should always be == 0
  xmax =  {self.xmax},
\u007d

!----------time limits ----------
time
\u007b
  tmin = 0.0,
  tmax  = {self.tmax},
\u007d
'''
    def _generate_field_params(self):
        """Generate field parameters section."""
        return f'''
!----------field solver set up----------
el_mag_fld
\u007b
  ! Set two of the field components with the Python script
  ! Note, you need to set PYTHONPATH in the console to the folder containing py-script-1d.py
  type_init_b(1:3) = "python", "python", "python",
  type_init_e(1:3) = "python", "python", "python",
  init_py_mod = "py-script-1d", ! Name of Python file
  init_py_func = "set_fld_int", ! Name of function in the Python file to call (same for all components)
  ! init_move_window = .false., ! May want to declare this for a moving-window simulation

  ! You can also do this with external fields, as functions of time
  ext_fld = "static",
  type_ext_b(1:3) = "uniform", "uniform", "uniform",
  ext_b0(1:3) = {np.format_float_scientific(np.cos(self.theta) * self.B0 / self.normalizations['Bx_int'], 4)}, {np.format_float_scientific(-np.sin(self.theta) * self.B0 / self.normalizations['Bx_int'])}, 0,
  \u007d

!----------boundary conditions for em-fields ----------
emf_bound
\u007b
  type(1:2,1) =   "open", "open",
\u007d

!----------- electro-magnetic field diagnostics ---------
diag_emf
\u007b
  reports = 
    "b1, savg",
    "b2, savg",
    "b3, savg",
    "e1, savg",
    "e2, savg",
    "e3, savg",
    
  !ndump_fac = 1,                     ! do full grid diagnostics at every 20 timesteps
  ndump_fac_ave = 1,                  ! do average/envelope grid diagnostics at every timestep
  ! ndump_fac_lineout = 1,              ! do lineouts/slices at every timestep
  ndump_fac_ene_int = 1,
  n_ave(1:1) = 4,                   ! average/envelope 8 cells (2x2x2)
  !n_tavg = 5,                         ! average 5 iterations for time averaged diagnostics 
\u007d
'''
    
    def _generate_electrons_section(self, thermal_bounds):
        """Generate the section for electrons."""
        vthele_start = thermal_bounds[0]
        vthele_end = thermal_bounds[1]
        
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
 num_par_x = {self.ppc},
 init_type = "python",
\u007d

!----------inital proper velocities - electrons-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-1d", ! Name of Python file
 uth_py_func = "set_uth_e", ! Name of function in the Python file to call
 
 ! use_spatial_ufl = .true.,
 ufl_py_mod = "py-script-1d", ! Name of Python file
 ufl_py_func = "set_ufl_e", ! Name of function in the Python file to call
\u007d

!----------density profile for electrons----------
profile
\u007b
 py_mod = "py-script-1d", ! Name of Python file
 py_func = "set_density_e", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for electrons----------
spe_bound
\u007b
 type(1:2,1) = "thermal","thermal",
 uth_bnd(1:3,1,1) = {np.format_float_scientific(vthele_start,4)}, {np.format_float_scientific(vthele_start,4)}, {np.format_float_scientific(vthele_start,4)}, 
 uth_bnd(1:3,2,1) = {np.format_float_scientific(vthele_end,4)}, {np.format_float_scientific(vthele_end,4)}, {np.format_float_scientific(vthele_end,4)}, 
\u007d

!----------diagnostic for electrons----------
diag_species
\u007b
 ndump_fac = 1,
 ndump_fac_temp = 1,
 ndump_fac_ene = 1,
 reports = "charge",
 rep_udist = "uth1", "uth2", "ufl1", "ufl2",
 ndump_fac_pha = 1,
 ps_pmin(1:3) = -0.1, -0.1, -0.02,
 ps_pmax(1:3) = 0.1,  0.1,  0.02,
 ps_xmin(1:1) = 0.0,
 ps_xmax(1:1) = {self.xmax},
 ps_np = 4096,
 ps_nx = 4096,
 phasespaces = "p1x1", "p2x1","p3x1",
\u007d
'''

    def _generate_ion_section(self, ion, thermal_bounds):
        """Generate the section for a specific ion species."""
        vthion_start = thermal_bounds[0]
        vthion_end = thermal_bounds[1]
        
        return f'''
!----------information for {ion} ions----------
species
\u007b
 name = "{ion}",
 rqm = {self.species_rqms[ion] / self.rqm_factor},
 num_par_x = {self.ppc},
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
 ndump_fac_temp = 1,
 ndump_fac_ene = 1,
 reports = "charge",
 rep_udist = "uth1", "uth2", "ufl1", "ufl2",
 ndump_fac_pha = 1,
 ps_pmin(1:3) = -0.05, -0.05, -0.02,
 ps_pmax(1:3) = 0.05,  0.05,  0.02,
 ps_xmin(1:1) = 0.0,
 ps_xmax(1:1) = {self.xmax},
 ps_np = 4096,
 ps_nx = 4096,
 !if_ps_p_auto(1:3) = .true., .true., .true.,
 phasespaces = "p1x1", "p2x1","p3x1",
\u007d'''

    def write_python_script(self):
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
            
    def write_everything(self):
        # Main function to run the interface
        self.save_slices()
        self.write_input_file()
        self.write_python_script()

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


if __name__ == "__main__":
    # Example usage
    interface = FLASH_OSIRIS(
        FLASH_data="/home/dschneidinger/shared/data/VAC_DEREK3D_20um/MagShockZ_hdf5_chk_0006",
        inputfile_name="magshockz-v3.2.1d",
        osiris_dims=1,
        reference_density=5e18,
        ppc=100,
        start_point=[0, 240],
        theta=np.pi / 2,
    )
    # interface.write_everything()
    interface.show_lineout_in_plane('edens')
    interface.plot_1D_lineout('edens')