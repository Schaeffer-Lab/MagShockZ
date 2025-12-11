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
        self.n0 = reference_density_cc
        self.ppc = ppc
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

        self.proj_dir = Path.cwd()
        self.output_dir = self.proj_dir / "input_files" / self.inputfile_name


        self.omega_pe = np.sqrt(4 * np.pi * self.n0 * yt.units.cm**-3 * yt.units.electron_charge_cgs**2 / yt.units.electron_mass_cgs)
        logger.info(f"Plasma frequency omega_pe: {np.format_float_scientific(self.omega_pe)} rad/s")

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

        logger.info(f"x bounds: {self.x[[0, -1]]} c/w_pe")
        logger.info(f"y bounds: {self.y[[0, -1]]} c/w_pe")
        logger.info(f"z bounds: {self.z[[0, -1]]} c/w_pe")

        debye_osiris = np.sqrt(
            self.all_data['flash', 'tele'][-1, -1, 0] * yt.units.kB * yt.units.erg / (yt.units.electron_mass_cgs * yt.units.speed_of_light**2)
        )
        
        logger.info(f"Debye length: {debye_osiris.value} osiris units")
        logger.info(f"Background temperature: {round(self.all_data['flash', 'tele'][-1, -1, 0].value * yt.units.kB)} eV")


        self.rqm_real = 1836 / self.all_data['flash', 'ye'][-1, -1, 0]
        logger.info(f"{'*'*10} real mass ratio: {self.rqm_real} {'*'*10}")


        logger.info("normalizing plasma parameters")
        ######## NORMALIZATIONS ######## 
        B_norm = (self.omega_pe * self.m_e * self.c) / self.e
        E_norm = B_norm * self.c / np.sqrt(self.rqm_factor)
        v_norm = self.c / np.sqrt(self.rqm_factor)
        vth_ele_norm = np.sqrt(self.m_e * self.c**2)
        
        self.normalizations = {
            'edens': self.n0,
            
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
        chunk_size = 32 
        
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

        # Tried to include memory cleanup here, but it just doesn't work
    
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
        
        # Prepare template context
        context = self._prepare_template_context(thermal_bounds)
        
        # Load and render template
        template_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('MagShockZ_OSIRIS_TEMPLATE.jinja')
        content = template.render(**context)
        
        # Write the actual input file
        input_file_path = self.output_dir / self.inputfile_name
        logger.info(f"Writing OSIRIS input file to {input_file_path}")
        
        with open(input_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"OSIRIS input file written successfully")

    def _prepare_template_context(self, thermal_bounds):
        """Prepare the context dictionary for Jinja2 template rendering."""
        if self.osiris_dims == 1:
            nx = int(self.xmax / self.dx)
            ny = None
            xmin, xmax = 0, self.xmax
            ymin, ymax = None, None
        else:
            nx = int((self.xmax - self.xmin) / self.dx)
            ny = int((self.ymax - self.ymin) / self.dx)
            xmin, xmax = self.xmin, self.xmax
            ymin, ymax = self.ymin, self.ymax
            

        
        tile_numbers = self._calculate_tile_numbers() if self.algorithm in ["cuda", "tiles"] else []
        
        species_list = self._prepare_species_list(thermal_bounds) # Type: List[Dictionary]

        # preset params. Feel free to modify.
        n_dump_total = 400
        vpml_bnd_size = 50
        interpolation = 'cubic'
        
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
            'interpolation': interpolation,
            'ppc': self.ppc,
            'dt': np.format_float_scientific(self.dt, 4),
            'ndump': int(self.tmax / (n_dump_total * self.dt)),
            'tmax': self.tmax,
            'tile_numbers': tile_numbers,
            'num_species': len(self.species_rqms) + 1,
            'species_list': species_list,
            'vpml_bnd_size': vpml_bnd_size,
            'ps_pmin': [-0.1, -0.1, -0.02],
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
            n_tiles_min = (self.xmax - self.xmin) * (self.ymax - self.ymin) / self.dx**2 / 1024
            i = j = 0
            while True:
                n_tiles_x = 2**i
                n_tiles_y = 2**j
                if n_tiles_x * n_tiles_y > n_tiles_min:
                    break
                j += 1
                n_tiles_x = 2**i
                n_tiles_y = 2**j
                if n_tiles_x * n_tiles_y > n_tiles_min:
                    break
                i += 1
            return [n_tiles_x, n_tiles_y]
    
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
                    vthele((self.ymin, self.xmin)),
                    vthele((self.ymax, self.xmax))
                ]
                logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")
            
            # Read ion thermal velocity bounds for each species
            for ion in self.species_rqms.keys():
                with open(self.output_dir / f"interp/vth{ion}.pkl", "rb") as f:
                    vthion = pickle.load(f)
                    bounds['ions'][ion] = [
                        vthion((self.ymin, self.xmin)),
                        vthion((self.ymax, self.xmax))
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
                    x_lower_bound = np.mean([vthele((y, self.xmin)) for y in y_samples])
                    x_upper_bound = np.mean([vthele((y, self.xmax)) for y in y_samples])

                    y_lower_bound = np.mean([vthele((self.ymin, x)) for x in x_samples])
                    y_upper_bound = np.mean([vthele((self.ymax, x)) for x in x_samples])

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
                    x_lower_bound = np.mean([vthion((y, self.xmin)) for y in y_samples])
                    x_upper_bound = np.mean([vthion((y, self.xmax)) for y in y_samples])

                    y_lower_bound = np.mean([vthion((self.ymin, x)) for x in x_samples])
                    y_upper_bound = np.mean([vthion((self.ymax, x)) for x in x_samples])

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
    
    def show_lineout_in_plane(self, fields):
        import matplotlib.pyplot as plt
        for field in fields:
            if field.endswith('dens'):
                data = np.log(np.load(self.output_dir / f'interp/{field}.npy')) # I take the log here because density has a large numerical range

            else:
                import pickle
                with open(self.output_dir / f'interp/{field}.pkl', "rb") as p:
                    f = pickle.load(p)
                    x1,x2 = np.meshgrid(self.x, self.y, indexing='ij')
                    data = f((x2,x1))

            plt.figure()

            plt.imshow(data.T, origin='lower',
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])

            plt.plot([self.xmin, self.xmax], [self.ymin, self.ymax],color='r')

            plt.savefig(f'{self.output_dir}/{field}_lineout.png')

    def show_box_in_plane(self, field):
        import matplotlib.pyplot as plt
        if self.osiris_dims != 2:
            raise ValueError("show_box_in_plane is only applicable for 2D simulations.")
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

        plt.plot([self.xmin, self.xmax], [self.ymin, self.ymin], color='r')
        plt.plot([self.xmin, self.xmax], [self.ymax, self.ymax], color='r')
        plt.plot([self.xmin, self.xmin], [self.ymin, self.ymax], color='r')
        plt.plot([self.xmax, self.xmax], [self.ymin, self.ymax], color='r')

        plt.show()

    def plot_1D_lineout(self, fields):
        from scipy.interpolate import RegularGridInterpolator
        import matplotlib.pyplot as plt
        import pickle
        x_osiris = np.linspace(self.xmin, self.xmax,1000)
        y_osiris = np.linspace(self.ymin, self.ymax,1000)

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
            f"Reference density: {self.n0:.2e} cm^-3",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"OSIRIS dimensions: 1D",
            f"Particles per cell: {self.ppc}",
            f"Start point: [{self.xmin}, {self.ymin}] [c/ωpe]",
            f"Ray angle: {self.theta:.4f} rad",
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

        super().__init__(osiris_dims=2, **kwargs)
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
    test_1d = FLASH_OSIRIS_1D(
        path_to_FLASH_data=Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2024-06/MAGON/MagShockZ_hdf5_chk_0005"),
        OSIRIS_inputfile_name="test_1d",
        reference_density_cc=5e18,
        ppc=8,
        dx=0.1,
        start_point=[0, 100],
        distance=1000,
        theta=np.pi/2,
        rqm_normalization_factor=10,
        tmax_gyroperiods=10,
        algorithm="cpu"
    )

    test_1d.save_slices()
    test_1d.write_input_file()
    test_1d.show_lineout_in_plane(['edens', 'magx', 'magy'])

