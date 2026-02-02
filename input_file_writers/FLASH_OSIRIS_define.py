from logging import config
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yt
from jinja2 import Environment, FileSystemLoader


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
                 osiris_dims: int,
                 xmin: float,
                 xmax: float,
                 ymin: float,
                 ymax: float,
                 theta: float = None,
                 distance: float = None,
                 rqm_normalization_factor: int = 10,
                 species_rqms: Dict[str, int] = {"al": 7257, "si": 3899}, # Defaults for MagShockZ
                 tmax_gyroperiods: int = 10,
                 algorithm: str = "cpu",
                 normalizations_override: Dict[str, float] = {}):

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
        
        if osiris_dims not in [1, 2]:
            raise ValueError("osiris_dims must be either 1 or 2")
 

        self.osiris_dims = osiris_dims
        self.FLASH_data = Path(path_to_FLASH_data)
        self.inputfile_name = OSIRIS_inputfile_name + f".{self.osiris_dims}d"
        self.n0 = reference_density_cc * yt.units.cm**-3
        self.ppc = ppc
        self.interpolation = 'cubic'
        self.dx = dx 
        self.rqm_factor = rqm_normalization_factor
        self.species_rqms = species_rqms
        self.tmax_gyroperiods = tmax_gyroperiods
        self.algorithm = algorithm
        self.dt = self.dx * 0.95 / np.sqrt(self.osiris_dims) # CFL condition
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.theta = theta
        self.distance = distance
        self.normalizations_override = normalizations_override

        self.proj_dir = Path.cwd().parent
        logger.info(f"Project directory: {self.proj_dir}")
        self.output_dir = self.proj_dir / "input_files" / self.inputfile_name
        logger.info(f"Output directory: {self.output_dir}")

        self.omega_pe = np.sqrt(4 * np.pi * self.n0 * yt.units.electron_charge**2 / (yt.units.electron_mass))
        logger.info(f"Plasma frequency omega_pe: {np.format_float_scientific(self.omega_pe.to('1/s'), 2)} 1/s")
        logger.info(f"1000 c/w_pe is {np.format_float_scientific((1000 * yt.units.speed_of_light / self.omega_pe).to('cm'),2)} cm")

        logger.info(f"Loading FLASH data from {self.FLASH_data}")
        self.ds = yt.load_for_osiris(self.FLASH_data.as_posix(), rqm_factor=self.rqm_factor)

        level = 1 # If you are getting inexplicable crashes, you are probably running out of memory. This is probably the culprit.
        self.dims = self.ds.domain_dimensions * self.ds.refine_by**level

        logger.info(f"Creating covering grid at level {level} with dims {self.dims}")

        self.all_data = self.ds.covering_grid(
            level,
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1,
        )

        logger.info("Covering grid created successfully")
        logger.info("Extracting coordinate arrays")
        self.x = self.all_data['flash', 'x'][:, 0, 0] * self.omega_pe / yt.units.speed_of_light
        self.y = self.all_data['flash', 'y'][0, :, 0] * self.omega_pe / yt.units.speed_of_light
        self.z = self.all_data['flash', 'z'][0, 0, :] * self.omega_pe / yt.units.speed_of_light

        logger.info(f"x bounds: {np.round(self.x[[0, -1]], 2)} c/w_pe")
        logger.info(f"y bounds: {np.round(self.y[[0, -1]], 2)} c/w_pe")
        logger.info(f"z bounds: {np.round(self.z[[0, -1]], 2)} c/w_pe")


        debye_osiris = np.sqrt(
            self.all_data['flash', 'tele'][-1, -1, 0] * yt.units.boltzmann_constant / (yt.units.electron_mass * yt.units.speed_of_light**2)
        )
        
        # logger.info(f"Debye length: {debye_osiris.value} osiris units")
        logger.info(f"Background temperature: {self.all_data['flash', 'tele'][-1, -1, 0].to('K'):.3e}")
        logger.info(f"Background temperature: {(self.all_data['flash', 'tele'][-1, -1, 0] * yt.units.boltzmann_constant).to('eV'):.3e}")


        self.rqm_real = 1836 / self.all_data['flash', 'ye'][-1, -1, 0] # ye has units of 
        logger.info(f"{'*'*10} real mass ratio: {self.rqm_real} {'*'*10}")


        logger.info("normalizing plasma parameters")
        ######## NORMALIZATIONS ######## 
        B_norm = self.omega_pe * yt.units.electron_mass * yt.units.speed_of_light / yt.units.elementary_charge
        v_norm = yt.units.speed_of_light / np.sqrt(self.rqm_factor)
        E_norm = self.omega_pe * yt.units.electron_mass * yt.units.speed_of_light / yt.units.elementary_charge / np.sqrt(self.rqm_factor)
        # E_norm must equal v_norm * B_norm so that E = -v x B holds in normalized units
        # E_norm = v_norm * B_norm

        logger.info(f"Electric field normalization: {E_norm.to('statV/cm'):.3e}")
        logger.info(f"Magnetic field normalization: {B_norm.to('Gauss'):.3e}")
        logger.info(f"Velocity normalization: {v_norm.to('cm/s'):.3e} cm/s")
        
        self.normalizations = {
            'edens': self.n0,
            'aldens': self.n0,
            'sidens': self.n0,
            
            'magx': B_norm, 'magy': B_norm, 'magz': B_norm,
            
            'Ex': E_norm, 'Ey': E_norm, 'Ez': E_norm,
            
            'v_ix': v_norm, 'v_iy': v_norm, 'v_iz': v_norm,
            'v_ex': v_norm, 'v_ey': v_norm, 'v_ez': v_norm,
            
            'vthele': yt.units.speed_of_light,
            'vthal': v_norm,
            'vthsi': v_norm,
        }
        
        # Calculate gyrotime and simulation duration
        self.gyrotime = self.species_rqms['al']/ self.rqm_factor / (self.all_data['flash', 'magx'][-1, -1, 0] / self.normalizations['magx'])
        self.tmax = int(self.gyrotime * self.tmax_gyroperiods)

        # print(f"Normalizations: {self.normalizations}")
        n_species = 3
        if self.osiris_dims == 1:
            n_particles = (self.xmax-self.xmin) / self.dx  * n_species * self.ppc
        elif self.osiris_dims == 2:
            n_particles = (self.xmax-self.xmin) * (self.ymax - self.ymin) / self.dx**2  * n_species * self.ppc**2
        n_bytes_particles = n_particles* 2 * 70 # maria says ~70 bytes per particle. I don't know if this is single or double precision, we also need to allocate for twice as many particles

        mem_per_GPU = 40e9
        max_bytes_per_GPU = mem_per_GPU * .8 # 80% of 16GB
        print("Number of particles: ", np.format_float_scientific(n_particles,3))

        logger.info(f"Recommended number of GPUs: {np.ceil(n_bytes_particles/max_bytes_per_GPU)}")
        logger.info(f"Recommended number of nodes: {np.ceil(n_bytes_particles/max_bytes_per_GPU/4)}")

 

    
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
        chunk_size = 32 
        
        for field, normalization in self.normalizations.items():
            if field in self.normalizations_override.keys():
                normalization = normalization * self.normalizations_override[field]
                logger.info(f"{field} is normalized by additional factor of {np.format_float_scientific(self.normalizations_override[field],3)}")
            logger.info(f"Processing {field} with normalization {np.format_float_scientific(normalization, 3)}")
            self._save_field(field, normalization, middle_index, chunk_size, interp_dir)
            
    def _save_field(self, field, normalization, middle_index, chunk_size, interp_dir):
        """Save regular field data."""
        import pickle
        from scipy.interpolate import RegularGridInterpolator
        
        # Initialize field data array
            
        field_data = np.zeros(self.all_data['flash', field][:, :, middle_index].shape)
        
        # Process data in chunks to save memory
        for i in range(0, self.all_data['flash', field].shape[0], chunk_size):
            end = min(i + chunk_size, self.all_data['flash', field].shape[0])
            field_data_chunk = self.all_data['flash', field][i:end, :, middle_index] / normalization
            field_data[i:end, :] = field_data_chunk

        # Special handling for density fields
        if field.endswith('dens'):
            # Remove small density values
            lower_bound = 0.001
            field_data[field_data < lower_bound] = 0
            np.save(f"{interp_dir}/{field}.npy", field_data)
        else:
            interp = RegularGridInterpolator(
                (self.x, self.y), field_data,
                method='linear', bounds_error=True, fill_value=0)
            with open(f"{interp_dir}/{field}.pkl", "wb") as f:
                pickle.dump(interp, f)

        # Tried to include memory cleanup here, but it just doesn't work

    def write_input_file(self):
        """
        Generate and write OSIRIS input file using Jinja2 templates.
        
        Reads thermal velocity bounds and generates the input file
        with appropriate parameters for all species.
        """
        
        # Read thermal velocity bounds for all species
        thermal_bounds = self._read_thermal_bounds()
        
        # Prepare the context dictionary for Jinja2 template rendering
        if self.osiris_dims == 1:
            nx = int(self.distance / self.dx)
            ny = None
            xmin, xmax = 0, self.distance 
            ymin, ymax = None, None
        else:
            nx = int((self.xmax - self.xmin) / self.dx)
            ny = int((self.ymax - self.ymin) / self.dx)
            xmin, xmax = self.xmin, self.xmax
            ymin, ymax = self.ymin, self.ymax
            
        n_tiles_x, n_tiles_y = self._calculate_tile_numbers()
        
        species_list = self._prepare_species_list(thermal_bounds) # Type: List[Dictionary]

        # preset params. Feel free to modify.
        n_dump_total = 256
        vpml_bnd_size = 100

        if self.osiris_dims == 1:
            num_par_max = int(self.ppc*nx/n_tiles_x/4) # Factor of 4 is random. Otherwise it's way too large idk
        else:
            num_par_max = int(nx*ny/(n_tiles_x*n_tiles_y)*self.ppc**2/4)
        
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
            'interpolation': self.interpolation,
            'ppc': self.ppc,
            'num_par_max': num_par_max,
            'dt': np.format_float_scientific(self.dt, 4),
            'ndump': int(self.tmax / (n_dump_total * self.dt)),
            'tmax': self.tmax,
            'tile_numbers': [n_tiles_x, n_tiles_y],
            'num_species': len(self.species_rqms) + 1,
            'species_list': species_list,
            'vpml_bnd_size': vpml_bnd_size,
            'ps_pmin': [-0.1, -0.1, -0.05],
            'ps_pmax': [0.1, 0.1, 0.05],
            'ps_np': [4096, 4096, 64],
            'ps_nx': 32,
            'ps_ny': 4096,
            'smooth_type': 'binomial',
            'smooth_order': 2,
        }

        # Load and render template
        template_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('MagShockZ_python_TEMPLATE.jinja')
        content = template.render(**context)
        
        # Write the actual input file
        input_file_path = self.output_dir / self.inputfile_name
        logger.info(f"Writing OSIRIS input file to {input_file_path}")
        
        with open(input_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"OSIRIS input file written successfully")
   
    def _calculate_tile_numbers(self):
        precision = 8 # bytes per number (double precision)
        match self.interpolation:
            case 'linear':
                interp = 1
            case 'quadratic':
                interp = 2
            case 'cubic':
                interp = 3
            case 'quartic':
                interp = 4
            case _:
                raise ValueError("Unsupported interpolation type")
        ### FOR PERLMUTTER:
        shmemsize = 163e3 * 0.8 # 80% of shared memory per block in bytes

        ### FOR LOCAL MACHINES:
        # TBD: adjust shared memory size accordingly
        max_tile_size = int((shmemsize/(2*3*precision))**(1/self.osiris_dims) - (2 * interp - 1))
        print("Max tile size per dimension: ", max_tile_size)
        if self.osiris_dims == 1:
            i = 0
            n_tiles_x = 2**i
            tile_size_x = (self.xmax - self.xmin) / self.dx / n_tiles_x
            while tile_size_x > max_tile_size:
                i += 1
                n_tiles_x = 2**i
                tile_size_x = (self.xmax - self.xmin) / self.dx / n_tiles_x
        if self.osiris_dims == 2:
            i, j = 0, 0
            n_tiles_x, n_tiles_y = 2**i, 2**j
            tile_size_x = (self.xmax - self.xmin) / self.dx / n_tiles_x
            tile_size_y = (self.ymax - self.ymin) / self.dx / n_tiles_y
            while tile_size_x > max_tile_size:
                i += 1
                n_tiles_x = 2**i
                tile_size_x = (self.xmax - self.xmin) / self.dx / n_tiles_x
            while tile_size_y > max_tile_size:
                j += 1
                n_tiles_y = 2**j
                tile_size_y = (self.ymax - self.ymin) / self.dx / n_tiles_y
        logger.info(f"Calculated tile numbers: n_tiles_x = {n_tiles_x}, n_tiles_y = {n_tiles_y}")
        return n_tiles_x, n_tiles_y
    
    def _prepare_species_list(self, thermal_bounds):
        """Prepare species data for template rendering."""
        species_list = []
        
        # Add electrons
        electron_config = self._get_species_config('e', thermal_bounds['electron'], is_electron=True)
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
            'rqm': -1.0 if is_electron else int(self.species_rqms[species_name] / self.rqm_factor),
            'ps_pmin': [-0.1, -0.1, -0.02] if is_electron else [-0.05, -0.05, -0.02],
            'ps_pmax': [0.1, 0.1, 0.02] if is_electron else [0.05, 0.05, 0.02],
        }
        
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

    def _read_thermal_bounds(self) -> Dict:
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
                    vthele((self.xmin, self.ymin)),
                    vthele((self.xmax, self.ymax))
                ]
                logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")
            
            # Read ion thermal velocity bounds for each species
            for ion in self.species_rqms.keys():
                with open(self.output_dir / f"interp/vth{ion}.pkl", "rb") as f:
                    vthion = pickle.load(f)
                    bounds['ions'][ion] = [
                        vthion((self.xmin, self.ymin)),
                        vthion((self.xmax, self.ymax))
                    ]
                    logger.info(f"{ion} thermal velocity bounds: {bounds['ions'][ion]}")
        elif self.osiris_dims == 2:             
            bounds = {
                'electron': {},
                'ions': {}
            }
            num_samples = 16  # Number of points to sample
            x_samples = np.linspace(self.xmin, self.xmax, num_samples)
            y_samples = np.linspace(self.ymin, self.ymax, num_samples)
            with open(self.output_dir / "interp/vthele.pkl", "rb") as f:
                    vthele = pickle.load(f)
                    x_lower_bound = np.mean([vthele((self.xmin, y)) for y in y_samples])
                    x_upper_bound = np.mean([vthele((self.xmax, y)) for y in y_samples])

                    y_lower_bound = np.mean([vthele((x, self.ymin)) for x in x_samples])
                    y_upper_bound = np.mean([vthele((x, self.ymax)) for x in x_samples])
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
                    x_lower_bound = np.mean([vthion((self.xmin, y)) for y in y_samples])
                    x_upper_bound = np.mean([vthion((self.xmax, y)) for y in y_samples])

                    y_lower_bound = np.mean([vthion((x, self.ymin)) for x in x_samples])
                    y_upper_bound = np.mean([vthion((x, self.ymax)) for x in x_samples])
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
    
    def write_python_file(self):
        """
        Generate and write OSIRIS python initialization file using Jinja2 templates.
        """
        # Prepare the context dictionary for Jinja2 template rendering
        context = {
            "dims": self.osiris_dims,
            "start_point": [self.xmin, self.ymin], 
            "distance": self.distance,
            "theta": self.theta,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "species_list": list(self.species_rqms.keys()),
            "box_bounds": {
                "xmin": np.min(self.x).value,
                "xmax": np.max(self.x).value,
                "ymin": np.min(self.y).value,
                "ymax": np.max(self.y).value,
            },
        }
        
        # Load and render template
        template_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('MagShockZ_py-script_TEMPLATE.jinja')
        content = template.render(**context)
        
        # Write the actual python initialization file
        python_file_path = self.output_dir / f"py-script-{self.osiris_dims}d.py"
        logger.info(f"Writing OSIRIS python initialization file to {python_file_path}")
        
        with open(python_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"OSIRIS python initialization file written successfully")
    
    def plot1D(self, fields):
        import matplotlib.pyplot as plt
        from scipy.interpolate import RegularGridInterpolator
        import pickle

        if isinstance(fields, str):
            fields = [fields]  # Convert to a list with a single element
        else:
            fields = fields  # Use as is

        for field in fields:
            if field.endswith('dens'):
                data = np.load(self.output_dir / f'interp/{field}.npy')
                f = RegularGridInterpolator(
                    (self.x, self.y), data, 
                    method='linear', bounds_error=True, fill_value=0)
            else:
                with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                    f = pickle.load(p)
                    x1,x2 = np.meshgrid(self.x, self.y)
                    data = f((x1,x2))

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left subplot: 2D plane with lineout
            if field.endswith('dens'):
                im = ax1.imshow(np.log(data), origin='lower',
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
            else:
                im = ax1.imshow(data, origin='lower',
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
            ax1.plot([self.xmin, self.xmax], [self.ymin, self.ymax], color='r', linewidth=2, label='Lineout')
            ax1.set_xlabel(r'x [$c/\omega_{pe}$]')
            ax1.set_ylabel(r'y [$c/\omega_{pe}$]')
            ax1.set_title(f'{field} - 2D plane')
            ax1.legend()
            plt.colorbar(im, ax=ax1)

            # Create points along the lineout from (xmin, ymin) to (xmax, ymax)
            n_points = 10000
            
            x_points = np.linspace(self.xmin, self.xmax, n_points)
            y_points = np.linspace(self.ymin, self.ymax, n_points)
            data_line = f(np.column_stack([x_points, y_points]))
            
            # Right subplot: 1D lineout
            ax2.plot(np.linspace(0, self.distance, n_points), data_line, color='r', linewidth=2)
            ax2.set_xlabel(r'Distance along lineout [$c/\omega_{pe}$]')
            ax2.set_ylabel(field)
            ax2.set_title(f'{field} - Lineout')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{field}_1D.png', dpi=150)
            plt.close()

    def plot2D(self, fields):
        import matplotlib.pyplot as plt
        from scipy.interpolate import RegularGridInterpolator
        import pickle
        output = {}
        if isinstance(fields, str):
            fields = [fields]  # Convert to a list with a single element
        else:
            fields = fields  # Use as is
        for field in fields:
            if field.endswith('dens'):
                data = np.load(self.output_dir / f'interp/{field}.npy')
                f = RegularGridInterpolator(
                    (self.x, self.y), data, 
                    method='linear', bounds_error=True, fill_value=0)
            else:
                with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                    f = pickle.load(p)
                    x1,x2 = np.meshgrid(self.x,self.y)
                    data = f((x1,x2))

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


            # plt.imshow(data.T, origin='lower',
            #             extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],vmax = 10) # TODO add some logic for vmax

            ax1.plot([self.xmin, self.xmax], [self.ymin, self.ymin], color='r')
            ax1.plot([self.xmin, self.xmax], [self.ymax, self.ymax], color='r')
            ax1.plot([self.xmin, self.xmin], [self.ymin, self.ymax], color='r')
            ax1.plot([self.xmax, self.xmax], [self.ymin, self.ymax], color='r')

            # Left subplot: 2D plane with lineout
            if field.endswith('dens'):
                im = ax1.imshow(np.log(data).T, origin='lower',
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
            else:
                im = ax1.imshow(data, origin='lower',
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
            ax1.set_xlabel(r'x [$c/\omega_{pe}$]')
            ax1.set_ylabel(r'y [$c/\omega_{pe}$]')
            ax1.set_title(f'{field} - 2D plane')
            ax1.legend()
            fig.colorbar(im, ax=ax1)

            # Create points along the lineout from (xmin, ymin) to (xmax, ymax)
            n_points = 10000
            
            bottom_line_x = np.linspace(self.xmin, self.xmax, n_points)
            bottom_line_y = self.ymin * np.ones_like(bottom_line_x)
            bottom_line = np.column_stack([bottom_line_x, bottom_line_y])
            bottom_data = f(bottom_line)

            top_line_x = np.linspace(self.xmin, self.xmax, n_points)
            top_line_y = self.ymax * np.ones_like(top_line_x)
            top_line = np.column_stack([top_line_x, top_line_y])
            top_data = f(top_line)

            left_line_x = self.xmin * np.ones_like(bottom_line_x)
            left_line_y = np.linspace(self.ymin, self.ymax, n_points)
            left_line = np.column_stack([left_line_x, left_line_y])
            left_data = f(left_line)

            right_line_x = self.xmax * np.ones_like(bottom_line_x)
            right_line_y = np.linspace(self.ymin, self.ymax, n_points)
            right_line = np.column_stack([right_line_x, right_line_y])
            right_data = f(right_line)
            
            # Right subplot: 1D lineout
            bottom_distance = np.linspace(0, self.xmax - self.xmin, n_points)
            top_distance = np.linspace(0, self.xmax - self.xmin, n_points)
            left_distance = np.linspace(0, self.ymax - self.ymin, n_points)
            right_distance = np.linspace(0, self.ymax - self.ymin, n_points)
            
            ax2.plot(bottom_distance, bottom_data, color='r', linewidth=2, label = 'Bottom edge')
            ax2.plot(top_distance, top_data, color='b', linewidth=2, label = 'Top edge')
            ax2.plot(left_distance, left_data, color='g', linewidth=2, label = 'Left edge')
            ax2.plot(right_distance, right_data, color='m', linewidth=2, label = 'Right edge')
            ax2.set_xlabel(r'[$c/\omega_{pe}$]')
            ax2.set_ylabel(field)
            ax2.set_title(f'{field} - Lineout')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{field}_2D.png', dpi=150)
            plt.close()

            output[field] = [bottom_data, top_data, left_data, right_data]

        return output
    
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
        

        xmax = distance * np.cos(theta) + start_point[0] # Have it match the form of 2D setup
        ymax = distance * np.sin(theta) + start_point[1]
         # CFL condition
        super().__init__(osiris_dims=1, theta = theta, distance=distance, xmax=xmax, ymax=ymax,
                         ymin=start_point[1], xmin=start_point[0], **kwargs)


        logger.info(str(self))
        
    def __str__(self):
        lines = [
            "=" * 50,
            "FLASH-OSIRIS INTERFACE",
            f"FLASH data: {self.FLASH_data}",
            f"Input file: {self.inputfile_name}",
            f"Reference density: {self.n0}",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"OSIRIS dimensions: 1D",
            f"Particles per cell: {self.ppc}",
            f"Start point: [{self.xmin}, {self.ymin}] [c/ωpe]",
            f"Ray angle: {self.theta} rad",
            f"Lineout distance: {self.distance} [c/ωpe]",
            f"Output directory: {self.output_dir}",
            "=" * 50
        ]
        return "\n".join(lines)
    

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

        super().__init__(osiris_dims=2,xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, **kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dt = self.dx * 0.95 / np.sqrt(self.osiris_dims) # CFL condition
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
    

if __name__ == "__main__":
    # test_1d = FLASH_OSIRIS_1D(
    #     path_to_FLASH_data=Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2024-06/MAGON/MagShockZ_hdf5_chk_0005"),
    #     OSIRIS_inputfile_name="test_1d",
    #     reference_density_cc=5e18,
    #     ppc=20,
    #     dx=0.3,
    #     start_point=[0, 300],
    #     distance=2000,
    #     theta=np.pi/2,
    #     rqm_normalization_factor=500,
    #     tmax_gyroperiods=20,
    #     algorithm="cuda"
    # )

    # test_1d.save_slices()
    # test_1d.write_input_file()
    # test_1d.plot1D(['edens', 'aldens','sidens', 'magx', 'magy', 'magz', 'Ex', 'Ey', 'Ez', 'vthele', 'vthal', 'v_ix', 'v_iy','v_ey'])
    # test_1d.write_python_file()

    # test_2d = FLASH_OSIRIS_2D(
    #     path_to_FLASH_data=Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2024-06/MAGON/MagShockZ_hdf5_chk_0005"),
    #     OSIRIS_inputfile_name="test_2d",
    #     reference_density_cc=5e18,
    #     ppc=2,
    #     dx=0.6,
    #     xmin=-700,
    #     xmax=700,
    #     ymin=300,
    #     ymax=2000,
    #     rqm_normalization_factor=500,
    #     tmax_gyroperiods=10,
    #     algorithm="cuda"
    # )
    test_2d = FLASH_OSIRIS_2D(
        path_to_FLASH_data=Path("/pscratch/sd/d/dschnei/MagShockZ_hdf5_chk_0005"),
        # path_to_FLASH_data=Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2024-06/MAGON/MagShockZ_hdf5_chk_0005"),
        OSIRIS_inputfile_name="perlmutter_2d",
        reference_density_cc=1e18,
        ppc=5,
        dx=0.2,
        xmin=-1000,
        xmax=1000,
        ymin=200,
        ymax=1800,
        rqm_normalization_factor=50,
        tmax_gyroperiods=50,
        algorithm="cuda"
    )

    test_2d.save_slices()
    test_2d.write_input_file()
    test_2d.plot2D(['edens', 'aldens','sidens', 'magx', 'magy', 'magz', 'Ex', 'Ey', 'Ez', 'vthele', 'vthal', 'vthsi', 'v_ix', 'v_iy','v_ey'])
    test_2d.write_python_file()
