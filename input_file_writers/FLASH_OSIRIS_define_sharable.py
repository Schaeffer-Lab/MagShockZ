import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import yt

E_CHARGE = 4.80320425e-10  # [statC] = [cm^3/2⋅g^1/2⋅s^−1]
ELECTRON_MASS = 9.1093837139e-28  # [g]
SPEED_OF_LIGHT = 2.99792458e10  # [cm/s]
KB = 8.617e-5  # eV/K
ERGS_PER_EV = 1.602e-12  # erg/eV

yt.enable_plugins()

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FLASH_OSIRIS_Base:
    def __init__(self, 
                 path_to_FLASH_data: str,
                 OSIRIS_inputfile_name: str,
                 reference_density_cc: float,
                 ppc: int,
                 dx: float,
                 rqm_normalization_factor: int = 10,
                 species_rqms: Dict[str, int] = None,
                 tmax_gyroperiods: int = 10,
                 algorithm: str = "cpu",
                 normalizations_override: Dict[str, float] = {}):

        if species_rqms is None:
            species_rqms = {"al": 7257, "si": 3899}

        if algorithm not in ["cpu", "cuda", "tiles"]:
            raise ValueError("algorithm must be either 'cpu', 'cuda', or 'tiles'")
        
        if not isinstance(ppc, int) or ppc <= 0:
            raise ValueError("ppc must be a positive integer")
        
        if not isinstance(dx, (int, float)) or dx <= 0:
            raise ValueError("dx must be a positive number")
        
        if not isinstance(rqm_normalization_factor, (int, float)) or rqm_normalization_factor <= 0:
            raise ValueError("rqm_normalization_factor must be a positive number")
        
        if not isinstance(tmax_gyroperiods, (int, float)) or tmax_gyroperiods <= 0:
            raise ValueError("tmax_gyroperiods must be a positive number")

        if not isinstance(reference_density_cc, (int, float)):
            raise TypeError("reference_density must be a number")

        if not path_to_FLASH_data.exists():
            raise FileNotFoundError(f"FLASH data file not found: {path_to_FLASH_data}")
        

        self.FLASH_data = Path(path_to_FLASH_data)
        self.inputfile_name = OSIRIS_inputfile_name + f".{self.osiris_dims}d"
        self.n0 = reference_density_cc
        self.ppc = ppc
        self.dx = dx
        self.rqm_factor = rqm_normalization_factor
        self.species_rqms = species_rqms
        self.tmax_gyroperiods = tmax_gyroperiods
        self.algorithm = algorithm

        self.proj_dir = Path.cwd()
        self.output_dir = self.proj_dir / "input_files" / self.inputfile_name
        

        self.omega_pe = np.sqrt(4 * np.pi * self.n0 * self.e**2 / self.m_e)

        E_CHARGE = 4.80320425e-10  # [statC] = [cm^3/2⋅g^1/2⋅s^−1]
        ELECTRON_MASS = 9.1093837139e-28  # [g]
        SPEED_OF_LIGHT = 2.99792458e10  # [cm/s]
        KB = 8.617e-5  # eV/K
        ERGS_PER_EV = 1.602e-12  # erg/eV

        self.e = E_CHARGE
        self.m_e = ELECTRON_MASS
        self.c = SPEED_OF_LIGHT

        logger.info(f"Loading FLASH data from {self.FLASH_data}")
        self.ds = yt.load_for_osiris(self.FLASH_data, rqm_factor=self.rqm_factor)

        level = 2 # If write times are long, change this to 0. If data looks too coarse, change this to 2.
        self.dims = self.ds.domain_dimensions * self.ds.refine_by**level

        self.all_data = self.ds.covering_grid(
            level,
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1,
        )

        self.x = self.all_data['flash', 'x'][:, 0, 0] * self.omega_pe / self.c
        self.y = self.all_data['flash', 'y'][0, :, 0] * self.omega_pe / self.c
        self.z = self.all_data['flash', 'z'][0, 0, :] * self.omega_pe / self.c
        
        debye_osiris = np.sqrt(
            self.all_data['flash', 'tele'][-1, -1, 0] * KB * ERGS_PER_EV / (self.m_e * self.c**2)
        )
        
        logger.info(f"Debye length: {debye_osiris.value} osiris units")
        logger.info(f"Background temperature: {round(self.all_data['flash', 'tele'][-1, -1, 0].value * KB)} eV")
        


        # Get real mass ratio for reference
        self.rqm_real = 1836 / self.all_data['flash', 'ye'][-1, -1, 0]
        logger.info(f"{'*'*10} real mass ratio: {self.rqm_real} {'*'*10}")


        # Get normalizations 
        B_norm = (self.omega_pe * self.m_e * self.c) / self.e
        E_norm = B_norm * self.c / np.sqrt(self.rqm_factor)
        v_norm = self.c / np.sqrt(self.rqm_factor)
        vth_ele_norm = np.sqrt(self.m_e * self.c**2)
        
        self.normalizations = {
            'edens': self.n0,
            
            'Bx_int': B_norm, 'By_int': B_norm, 'Bz_int': B_norm,
            'magx': B_norm, 'magy': B_norm, 'magz': B_norm,
            
            'Ex': E_norm, 'Ey': E_norm, 'Ez': E_norm,
            
            'v_ix': v_norm, 'v_iy': v_norm, 'v_iz': v_norm,
            'v_ex': v_norm, 'v_ey': v_norm, 'v_ez': v_norm,
            
            'vthele': vth_ele_norm,
        }
        
        # Add ion species specific normalizations
        for species, rqms in self.species_rqms.items():
            self.normalizations[species + 'dens'] = self.n0
            self.normalizations[f'vth{species}'] = vth_ele_norm * np.sqrt(rqms / self.rqm_factor)
        
        # Calculate gyrotime and simulation duration
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
        interp_dir = self.output_dir / "interp"
        if not interp_dir.exists():
            interp_dir.mkdir(parents=True)
            
        # Validate normal_axis
        axis_map = {"x": 0, "y": 1, "z": 2}
        if normal_axis not in axis_map.keys():
            raise ValueError("normal_axis must be one of 'x', 'y', or 'z'")
        normal = axis_map[normal_axis]
        
        middle_index = self.dims[normal] // 2
        chunk_size = 128
        
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
        
        if field == 'vthele':
            temp_field = 'tele'
        else:
            temp_field = 'tion'
        
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
            lower_bound = 0.001
            field_data[field_data < lower_bound] = 0
            np.save(f"{interp_dir}/{field}.npy", field_data)
        else:
            interp = RegularGridInterpolator(
                (self.y, self.x), field_data.T,
                method='linear', bounds_error=True, fill_value=0
            )
            
            with open(f"{interp_dir}/{field}.pkl", "wb") as f:
                pickle.dump(interp, f)
    # Shared methods like calculate_numbers, save_slices, etc.


class FLASH_OSIRIS_1D(FLASH_OSIRIS_Base):
    
    def __init__(self,
                 start_point: List[float],
                 distance: float,
                 theta: float = np.pi/2,
                 **kwargs):
        
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise TypeError("distance must be a positive number for 1D simulations")
        
        if not isinstance(start_point, (list, tuple)) or len(start_point) != 2:
            raise ValueError("start_point must be a list or tuple of two numbers for 1D simulations")

        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number for 1D simulations")
        
        self.osiris_dims = 1
        self.start_point = start_point
        self.theta = theta
        self.distance = distance
        self.dt = self.dx * 0.95 / np.sqrt(self.osiris_dims) # CFL condition
        self.xmax = self.start_point[0] * np.cos(self.theta) + self.start_point[0] # Have it match the form of 2D setup
        self.ymax = self.start_point[1] * np.sin(self.theta) + self.start_point[1]
        super().__init__(**kwargs)
        logger.info("\n" + str(self))
        
    def __str__(self):
        lines = [
            "=" * 50,
            "FLASH-OSIRIS INTERFACE",
            f"FLASH data: {self.FLASH_data}",
            f"Input file: {self.inputfile_name}",
            f"Reference density: {self.n0:.2e} cm^-3",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"OSIRIS dimensions: 1D",
            f"Particles per cell: {self.ppc}",
            f"Start point: {self.start_point} [c/ωpe]",
            f"Ray angle: {self.theta:.4f} rad",
            f"Lineout distance: {self.distance} [c/ωpe]",
            f"Output directory: {self.output_dir}",
            "=" * 50
        ]
        return "\n".join(lines)
    
    def _calculate_endpoint(self):
        """Calculate endpoint from start + angle + distance."""
        return [
            self.start_point[0] + self.distance * np.cos(self.theta),
            self.start_point[1] + self.distance * np.sin(self.theta)
        ]
    
    def _generate_simulation_params(self):
        """1D-specific simulation parameters."""
        # Your current _generate_simulation_params1D logic
        pass


class FLASH_OSIRIS_2D(FLASH_OSIRIS_Base):
    """2D box configuration."""
    
    osiris_dims = 2
    
    def __init__(self,
                 xmin: float,
                 xmax: float,
                 ymin: float,
                 ymax: float,
                 **kwargs):
        for param_name, param_value in [('xmin', xmin), ('xmax', xmax), ('ymin', ymin), ('ymax', ymax)]:
            if not isinstance(param_value, (int, float)):
                raise TypeError(f"{param_name} must be a number for 2D simulations")

        self.osiris_dims = 2
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dt = self.dx * 0.95 / np.sqrt(self.osiris_dims) # CFL condition
        super().__init__(**kwargs)
        logger.info("\n" + str(self))

    def __str__(self):
        lines = [
            "=" * 50,
            "FLASH-OSIRIS INTERFACE",
            f"FLASH data: {self.FLASH_data}",
            f"Input file: {self.inputfile_name}",
            f"Reference density: {self.n0:.2e} cm^-3",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"OSIRIS dimensions: 2D",
            f"Particles per cell: {self.ppc}",
            f"X range: {self.xmin} to {self.xmax} [c/ωpe]",
            f"Y range: {self.ymin} to {self.ymax} [c/ωpe]",
            f"Output directory: {self.output_dir}",
            "=" * 50
        ]
        return "\n".join(lines)
    