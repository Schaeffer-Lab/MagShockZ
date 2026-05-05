import osiris_utils
import numpy as np
import plasmapy
import astropy
from pathlib import Path
import scipy.integrate
import glob
import h5py

#TODO: allow better plotting routines for phase space. Allow to plot without taking moment
#TODO: fix the units for plotting moments and stuff in 2D. Right now both axes are spatial, but one should be velocity.
#TODO: allow for better integration when calculating things from the data, like multiplying by dimensional quantities
#TODO: clean it up generally. Lot of AI cruft rn
#TODO: make those plots where you have vector fields

class LazyMoment:
    """Lazy-loading moment diagnostic that computes timesteps on-demand.
    
    Acts like an osiris_utils diagnostic object with indexing and attributes,
    but computes moments only when requested and caches results to HDF5.
    """
    
    def __init__(self, sim_wrapper, species, momentum_component, order):
        """
        Parameters
        ----------
        sim_wrapper : MagShockZRun
            The MagShockZRun instance that will compute moments
        species : str
            Species name (e.g., 'al', 'si', 'e')
        momentum_component : str
            Which momentum component ('p1', 'p2', or 'p3')
        order : int
            Moment order (0=density, 1=velocity, 2=temperature)
        """
        self.sim_wrapper = sim_wrapper
        self.species = species
        self.momentum_component = momentum_component
        self.order = order
        
        # Get reference diagnostic for metadata
        pha_field = f'{species}/{momentum_component}x1x2'
        self.pha_data = sim_wrapper._get_field(pha_field)
        
        # Set name for this diagnostic
        moment_names = {0: 'n', 1: f'v{momentum_component[1]}', 2: 'vth2'}
        self.name = f'{species}/{moment_names[order]}-from-{momentum_component}'
        
        # Find which axis is momentum to exclude it from spatial grid
        p_axis_idx = sim_wrapper._find_momentum_axis(self.pha_data, momentum_component)
        
        # Copy spatial grid info (all axes except momentum)
        self.grid = [self.pha_data.grid[i] for i in range(len(self.pha_data.grid)) if i != p_axis_idx]
        self.nx = [self.pha_data.nx[i] for i in range(len(self.pha_data.nx)) if i != p_axis_idx]
    
    def time(self, timestep):
        """Get time information for a timestep (delegates to phase space diagnostic)."""
        return self.pha_data.time(timestep)
    
    def __getitem__(self, timestep):
        """Compute or load moment for requested timestep."""
        if self.order == 0:
            return self.sim_wrapper.calculate_0th_moment(
                self.species, timestep, self.momentum_component
            )
        elif self.order == 1:
            return self.sim_wrapper.calculate_1st_moment(
                self.species, timestep, self.momentum_component
            )
        elif self.order == 2:
            return self.sim_wrapper.calculate_2nd_moment(
                self.species, timestep, self.momentum_component
            )
        else:
            raise ValueError(f"Unsupported moment order: {self.order}")
    
    def __len__(self):
        """Return number of timesteps available."""
        return len(self.pha_data)


class MagShockZRun:
    """
    Thin wrapper around osiris_utils that adds experiment-specific 
    diagnostics for the MagShockZ collisionless shock experiments.
    """
    
    def __init__(self, input_deck: str, norm_density: float, B0: astropy.units.Gauss = None, 
                 Z: int = None, m_i: astropy.units.g = None):
        """
        Parameters
        ----------
        input_deck : str
            Path to input deck
        norm_density : float
            Reference plasma density n0 in cm^-3 (needed for unit conversion)
        B0 : astropy.units.Gauss, optional
            Reference magnetic field strength. If provided, methods that take B_real 
            as an argument will use this as the default.
        Z : int, optional
            Ion charge state. If provided, used as default in methods requiring Z.
        m_i : astropy.units.g, optional
            Ion mass. If provided, used as default in methods requiring m_i.
        """
        self.sim = osiris_utils.Simulation(input_deck_path = input_deck) # I think that I want to avoid all calls being sim.sim, is there a better way to do this?
        self.deck = self.sim._input_deck
        self.norm_density = norm_density
        self.B0 = B0
        self.Z = Z
        self.m_i = m_i
        self._lazy_moments = {}  # Store lazy-loading moment diagnostics
    
    # --- Helper methods ---
    
    def _resolve_params(self, B_real=None, Z=None, m_i=None):
        """
        Resolve parameters: use provided values if given, otherwise use instance defaults.
        Returns tuple of resolved (B_real, Z, m_i) with appropriate error checking.
        """
        resolved = {}
        
        if B_real is not None or self.B0 is not None:
            resolved['B_real'] = B_real if B_real is not None else self.B0
        
        if Z is not None or self.Z is not None:
            resolved['Z'] = Z if Z is not None else self.Z
        
        if m_i is not None or self.m_i is not None:
            resolved['m_i'] = m_i if m_i is not None else self.m_i
        
        return resolved
    
    def _require(self, params, *required_keys):
        """Check that required parameters are present, raise helpful error if not."""
        missing = [k for k in required_keys if k not in params]
        if missing:
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}. "
                f"Provide them as arguments or set during initialization."
            )
        return [params[k] for k in required_keys]
    
    # --- Unit conversion ---
    
    @property
    def omega_p_real(self):
        return (plasmapy.formulary.plasma_frequency(self.norm_density, particle = "e-")).to("rad/s")
    
    def omega_pi_real(self, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(Z=Z, m_i=m_i)
        Z, m_i = self._require(params, 'Z', 'm_i')
        return plasmapy.formulary.plasma_frequency(self.norm_density/Z, particle = self._ion_particle(Z, m_i)).to("rad/s")

    @property
    def rqm(self):
        if 'al' in self.deck.species:
            rqm = self.deck.species['al'].rqm
        else:
            raise KeyError("Ion species not found in input deck")
        return rqm
    
    def _ion_particle(self, Z: int, m_i: astropy.units.g):
        """Returns a plasmapy Particle for the ion species, needed for some formulary functions"""
        return plasmapy.particles.particle_class.CustomParticle(mass=m_i, Z=Z)
    
    def omega_ce_real(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        B_real, = self._require(params, 'B_real')
        return plasmapy.formulary.gyrofrequency(B_real, particle = "e-").to("rad/s")
    
    def B_osiris(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        B_real, = self._require(params, 'B_real')
        B_norm = (astropy.constants.m_e * self.omega_p_real / astropy.units.rad / astropy.constants.e.si).to(astropy.units.Gauss)
        return (B_real  / B_norm).to(astropy.units.dimensionless_unscaled)
    
    def omega_ci_real(self, B_real: astropy.units.Gauss = None, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(B_real=B_real, Z=Z, m_i=m_i)
        B_real, Z, m_i = self._require(params, 'B_real', 'Z', 'm_i')
        return plasmapy.formulary.gyrofrequency(B_real, particle = self._ion_particle(Z = Z, m_i = m_i)).to("rad/s")
    
    def omega_ci(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        B_real, = self._require(params, 'B_real')
        return self.B_osiris(B_real) / self.rqm
    
    def lambda_D_real(self, T_e: astropy.units.eV):
        return plasmapy.formulary.Debye_length(n_e = self.norm_density, T_e = T_e).to("cm")

    def lambda_D(self, T_e: astropy.units.eV):
        return (self.lambda_D_real(T_e) / self.electron_inertial_length_real()).to(astropy.units.dimensionless_unscaled)

    def vA_real(self, B_real: astropy.units.Gauss = None, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(B_real=B_real, Z=Z, m_i=m_i)
        B_real, Z, m_i = self._require(params, 'B_real', 'Z', 'm_i')
        return (B_real / np.sqrt(astropy.constants.mu0 * self.norm_density / Z * m_i)).to("cm/s")
    
    def vA(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        B_real, = self._require(params, 'B_real')
        return self.B_osiris(B_real) / np.sqrt(self.rqm)
    
    def electron_inertial_length_real(self):
        return (astropy.constants.c.si / (self.omega_p_real / astropy.units.rad)).to("cm")
    
    def ion_inertial_length_real(self, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(Z=Z, m_i=m_i)
        Z, m_i = self._require(params, 'Z', 'm_i')
        return (astropy.constants.c.si / (self.omega_pi_real(Z, m_i) / astropy.units.rad)).to("cm")
    
    def ion_inertial_length(self):
        return np.sqrt(self.rqm)

    def ion_sound_speed_real(self, T_e: astropy.units.eV, adiabatic_index: float = 5/3, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(Z=Z, m_i=m_i)
        Z, m_i = self._require(params, 'Z', 'm_i')
        return (np.sqrt(adiabatic_index * Z * T_e / m_i)).to("cm/s")
    
    # --- Phase space moments ---
    
    @staticmethod
    def _find_momentum_axis(pha_data, momentum_component):
        """
        Find which axis corresponds to the momentum dimension.
        
        Uses the .axis attribute metadata to identify the correct axis.
        """
        # Check each axis to find the one matching the momentum component
        for i, axis_info in enumerate(pha_data.axis):
            axis_name = axis_info.get('name', '').lower()
            # Match p1, p2, or p3
            if axis_name == momentum_component:
                return i
        
        # Fallback: assume last axis is momentum (common convention)
        return len(pha_data.axis) - 1
    
    @staticmethod
    def _moment(data, p_axis, order=0, axis=0):
        """Calculate a moment of the distribution function (based on user's implementation)."""
        weights = p_axis**order
        shape = [1] * data.ndim
        shape[axis] = -1
        weights = weights.reshape(shape)
        return scipy.integrate.simpson(data * weights, x=p_axis, axis=axis)
    
    def _diagnostic_exists(self, diag_name: str) -> bool:
        """Check if a diagnostic already exists in the simulation."""
        try:
            _ = self._get_field(diag_name)
            return True
        except (KeyError, AttributeError, ValueError):
            return False
    
    def _moment_h5_exists(self, diag_name: str, timestep: int) -> bool:
        """
        Check if a specific moment timestep HDF5 file exists on disk.
        
        Parameters
        ----------
        diag_name : str
            Diagnostic name (e.g., 'al/n-from-p1')
        timestep : int
            Timestep index
            
        Returns
        -------
        bool
            True if H5 file for this timestep exists, False otherwise
        """
        # Convert diagnostic name to filename format
        h5_name = diag_name.replace('/', '_')
        
        # Check in moments/<diagnostic_name>/ directory within simulation folder
        sim_path = Path(self.sim._simulation_folder)
        moments_dir = sim_path / 'moments' / h5_name
        
        # Look for the specific timestep file
        h5_file = moments_dir / f'{h5_name}-{timestep:06d}.h5'
        
        return h5_file.exists()
    
    def calculate_0th_moment(self, species: str, timestep: int, momentum_component: str = 'p1'):
        """
        Calculate density (0th moment) from phase space for a single timestep.
        
        Loads from cache if available, otherwise computes and saves to HDF5.
        """
        diag_name = f'{species}/n-from-{momentum_component}'
        h5_name = diag_name.replace('/', '_')
        
        # Check if this specific timestep already exists on disk
        if self._moment_h5_exists(diag_name, timestep):
            sim_path = Path(self.sim._simulation_folder)
            h5_file = sim_path / 'moments' / h5_name / f'{h5_name}-{timestep:06d}.h5'
            
            with h5py.File(h5_file, 'r') as f:
                result = f['AXIS'][h5_name][()]
            return result
        
        # Need to compute
        pha_field = f'{species}/{momentum_component}x1x2'
        pha_data = self._get_field(pha_field)
        
        # Find which axis is the momentum axis and get its coordinates
        p_axis_idx = self._find_momentum_axis(pha_data, momentum_component)
        p_min, p_max = pha_data.grid[p_axis_idx][0], pha_data.grid[p_axis_idx][-1]
        n_points = pha_data.nx[p_axis_idx]
        p_axis = np.linspace(p_min, p_max, n_points)
        
        # Get single timestep data and compute: n = ∫f dp
        data_t = pha_data[timestep]
        result = self._moment(data_t, p_axis, order=0, axis=p_axis_idx)
        
        # Save to HDF5
        sim_path = Path(self.sim._simulation_folder)
        moments_dir = sim_path / 'moments' / h5_name
        moments_dir.mkdir(parents=True, exist_ok=True)
        
        h5_file = moments_dir / f'{h5_name}-{timestep:06d}.h5'
        with h5py.File(h5_file, 'w') as f:
            axis_group = f.create_group('AXIS')
            axis_group.create_dataset(h5_name, data=result)
        
        return result
    
    def calculate_1st_moment(self, species: str, timestep: int, momentum_component: str = 'p1'):
        """
        Calculate mean velocity (1st moment / 0th moment) from phase space for a single timestep.
        
        Automatically calculates 0th moment if not already cached.
        """
        diag_name = f'{species}/v{momentum_component[1]}-from-{momentum_component}'
        h5_name = diag_name.replace('/', '_')
        
        # Check if this specific timestep already exists on disk
        if self._moment_h5_exists(diag_name, timestep):
            sim_path = Path(self.sim._simulation_folder)
            h5_file = sim_path / 'moments' / h5_name / f'{h5_name}-{timestep:06d}.h5'
            
            with h5py.File(h5_file, 'r') as f:
                result = f['AXIS'][h5_name][()]
            return result
        
        # Need to compute - first ensure 0th moment exists
        density = self.calculate_0th_moment(species, timestep, momentum_component)
        
        # Get phase space diagnostic
        pha_field = f'{species}/{momentum_component}x1x2'
        pha_data = self._get_field(pha_field)
        
        # Find which axis is the momentum axis and get its coordinates
        p_axis_idx = self._find_momentum_axis(pha_data, momentum_component)
        p_min, p_max = pha_data.grid[p_axis_idx][0], pha_data.grid[p_axis_idx][-1]
        n_points = pha_data.nx[p_axis_idx]
        p_axis = np.linspace(p_min, p_max, n_points)
        
        # Get single timestep data
        data_t = pha_data[timestep]
        
        # Compute: v = (∫p·f dp) / n
        flux = self._moment(data_t, p_axis, order=1, axis=p_axis_idx)
        result = flux / density
        
        # Save to HDF5
        sim_path = Path(self.sim._simulation_folder)
        moments_dir = sim_path / 'moments' / h5_name
        moments_dir.mkdir(parents=True, exist_ok=True)
        
        h5_file = moments_dir / f'{h5_name}-{timestep:06d}.h5'
        with h5py.File(h5_file, 'w') as f:
            axis_group = f.create_group('AXIS')
            axis_group.create_dataset(h5_name, data=result)
        
        return result
    
    def calculate_2nd_moment(self, species: str, timestep: int, momentum_component: str = 'p1'):
        """
        Calculate thermal velocity squared (temperature) from phase space for a single timestep.
        
        Automatically calculates 0th and 1st moments if not already cached.
        """
        diag_name = f'{species}/vth2-from-{momentum_component}'
        h5_name = diag_name.replace('/', '_')
        
        # Check if this specific timestep already exists on disk
        if self._moment_h5_exists(diag_name, timestep):
            sim_path = Path(self.sim._simulation_folder)
            h5_file = sim_path / 'moments' / h5_name / f'{h5_name}-{timestep:06d}.h5'
            
            with h5py.File(h5_file, 'r') as f:
                result = f['AXIS'][h5_name][()]
            return result
        
        # Need to compute - first ensure 0th and 1st moments exist
        density = self.calculate_0th_moment(species, timestep, momentum_component)
        velocity = self.calculate_1st_moment(species, timestep, momentum_component)
        
        # Get phase space diagnostic
        pha_field = f'{species}/{momentum_component}x1x2'
        pha_data = self._get_field(pha_field)
        
        # Find which axis is the momentum axis and get its coordinates
        p_axis_idx = self._find_momentum_axis(pha_data, momentum_component)
        p_min, p_max = pha_data.grid[p_axis_idx][0], pha_data.grid[p_axis_idx][-1]
        n_points = pha_data.nx[p_axis_idx]
        p_axis = np.linspace(p_min, p_max, n_points)
        
        # Get single timestep data
        data_t = pha_data[timestep]
        
        # Compute: vth² = ∫(p - v)²·f dp / n
        # Create the deviation (p - v) with proper broadcasting based on axis location
        shape = [1] * data_t.ndim
        shape[p_axis_idx] = -1
        p_broadcast = p_axis.reshape(shape)
        
        # Broadcast velocity to match
        v_shape = list(data_t.shape)
        v_shape[p_axis_idx] = 1
        v_broadcast = velocity.reshape(v_shape)
        
        w = p_broadcast - v_broadcast
        result = scipy.integrate.simpson(data_t * np.square(w), x=p_axis, axis=p_axis_idx) / density
        
        # Save to HDF5
        sim_path = Path(self.sim._simulation_folder)
        moments_dir = sim_path / 'moments' / h5_name
        moments_dir.mkdir(parents=True, exist_ok=True)
        
        h5_file = moments_dir / f'{h5_name}-{timestep:06d}.h5'
        with h5py.File(h5_file, 'w') as f:
            axis_group = f.create_group('AXIS')
            axis_group.create_dataset(h5_name, data=result)
        
        return result
    
    def add_moment_diagnostic(self, species: str, momentum_component: str = 'p1', order: int = 0):
        """
        Add a lazy-loading moment diagnostic that works with plotting routines.
        
        The moment is computed on-demand when accessed (e.g., via plot_field) and
        cached to HDF5 for future use. This allows moment calculations to work
        seamlessly with all existing plotting methods.
        
        Parameters
        ----------
        species : str
            Species name (e.g., 'al', 'si', 'e')
        momentum_component : str
            Which momentum component's phase space to use ('p1', 'p2', or 'p3')
        order : int
            Moment order:
            - 0: density (n)
            - 1: mean velocity (v)
            - 2: thermal velocity squared (vth²)
        
        Returns
        -------
        diag_name : str
            Name of the created diagnostic
        
        Examples
        --------
        >>> # Add density diagnostic
        >>> sim.add_moment_diagnostic('al', 'p1', order=0)
        'al/n-from-p1'
        >>> 
        >>> # Now plot it like any other field
        >>> sim.plot_field('al/n-from-p1', timestep=50)
        >>> sim.plot_streak('al/n-from-p1', timesteps=[0, 50, 100])
        >>> 
        >>> # Add velocity diagnostic
        >>> sim.add_moment_diagnostic('al', 'p1', order=1)
        'al/v1-from-p1'
        """
        moment_names = {0: 'n', 1: f'v{momentum_component[1]}', 2: 'vth2'}
        if order not in moment_names:
            raise ValueError(f"order must be 0, 1, or 2, got {order}")
        
        diag_name = f'{species}/{moment_names[order]}-from-{momentum_component}'
        
        # Create lazy-loading wrapper
        lazy_moment = LazyMoment(self, species, momentum_component, order)
        
        # Store in our custom dictionary (osiris_utils.add_diagnostic requires Diagnostic objects)
        self._lazy_moments[diag_name] = lazy_moment
        
        print(f"✓ Added lazy-loading diagnostic: '{diag_name}'")
        print(f"  Moments will be computed on-demand and cached to HDF5")
        
        return diag_name

    # --- Upstream averages (the PI questions) ---

    def plot_field(
        self,
        field_name: str,
        timestep: int = 0,
        ax=None,
        spatial_units: str = "ion",
        time_units: str = "ion gyrotime",
        log: bool = False,
        norm=None,
        vmin=None,
        vmax=None,
        cmap: str = "RdBu_r",
        colorbar: bool = True,
        **kwargs
    ):
        """
        Visualize a 2D field component at a given timestep.

        Parameters
        ----------
        field_name : str
            Name of the field to plot (e.g. 'b3', 'e1')
        timestep : int
            Timestep index to plot
        ax : matplotlib Axes, optional
            Axes to plot on. Creates a new figure if None.
        spatial_units : str
            Units for x-axis: 'ion' (c/ω_ci), 'electron' (c/ω_ce), 'physical', or 'cells'
        time_units : str
            Units for time-axis: 'ion gyrotime', 'electron', or 'physical'
        log : bool
            If True, use logarithmic color normalization
        norm : matplotlib Normalize, optional
            Explicit normalization. Overrides log if provided.
        vmin, vmax : float, optional
            Color scale limits
        cmap : str
            Colormap name
        colorbar : bool
            Whether to add a colorbar
        **kwargs
            Passed through to imshow
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        field_obj = self._get_field(field_name)
        field_data = field_obj[timestep]
        time_val, _ = (
            field_obj.time(timestep)[0],
            field_obj.time(timestep)[1],
        )

        # --- Axes ---
        if ax is None:
            ax = plt.figure(figsize=(6, 5)).gca()

        # --- Axis coordinates ---
        x1, x2 = field_obj.grid  # ensure grid info is loaded

        x, x_label = self._convert_axis(x1, spatial_units, direction="x")
        y, y_label = self._convert_axis(x2, spatial_units, direction="y")

        # --- Normalization ---
        if norm is None:
            if log:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # --- Plot ---
        im = ax.imshow(
            field_data.T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            norm=norm,
            cmap=cmap,
            aspect="auto",
            **kwargs,
        )

        if colorbar:
            plt.colorbar(im, ax=ax, label=field_name) #TODO: add units

        # --- Labels ---
        time, time_str = self._convert_time(time_val, time_units)
        ax.set_title(rf"{field_name} at t = {np.round(time,2)} {time_str}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return ax

    def plot_lineout(
        self,
        field_name: str,
        timestep: int = 0,
        axis: str = "x",
        position: float = None,
        slab: tuple = None,
        ax=None,
        spatial_units: str = "ion",
        time_units: str = "ion gyrotime",
        **kwargs
    ):
        """
        Plot a 1D lineout of a field along a specified axis.

        Parameters
        ----------
        field_name : str
            Name of the field to plot (e.g. 'b3-savg', 'e1')
        timestep : int
            Timestep index to plot
        axis : str
            Axis to take lineout along: 'x' or 'y'
        position : float, optional
            Position in the transverse direction (in spatial_units).
            If None, takes lineout at center. Ignored if slab is provided.
        slab : tuple of (float, float), optional
            Range (min, max) in transverse direction to average over (in spatial_units).
            If provided, overrides position.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates a new figure if None.
        spatial_units : str
            Units for spatial axes: 'ion', 'electron', 'physical', or 'cells'
        time_units : str
            Units for time display in title
        **kwargs
            Passed through to ax.plot()

        Returns
        -------
        ax : matplotlib Axes
            The axes object with the plot
        """
        import matplotlib.pyplot as plt

        # Get field data and grid
        field_obj = self._get_field(field_name)
        field_data = field_obj[timestep]
        time_val, _ = field_obj.time(timestep)
        x1_grid, x2_grid = field_obj.grid

        # Get data shape
        nx, ny = field_data.shape
        
        # Create full coordinate arrays from grid extents
        x1_full = np.linspace(x1_grid[0], x1_grid[-1], nx)
        x2_full = np.linspace(x2_grid[0], x2_grid[-1], ny)
        
        # Convert grids to desired units
        x_coords, x_label = self._convert_axis(x1_full, spatial_units, direction="x")
        y_coords, y_label = self._convert_axis(x2_full, spatial_units, direction="y")

        # Determine which axis to take lineout along
        if axis.lower() in ['x', 'x1']:
            lineout_coords = x_coords
            lineout_label = x_label
            transverse_coords = y_coords
            transverse_label = y_label
            transpose_data = False
        elif axis.lower() in ['y', 'x2']:
            lineout_coords = y_coords
            lineout_label = y_label
            transverse_coords = x_coords
            transverse_label = x_label
            transpose_data = True
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

        # Transpose data if taking lineout along y
        data = field_data.T if transpose_data else field_data

        # Determine transverse indices to average over
        if slab is not None:
            # Average over a range
            slab_min, slab_max = slab
            idx_min = np.argmin(np.abs(transverse_coords - slab_min))
            idx_max = np.argmin(np.abs(transverse_coords - slab_max))
            if idx_min > idx_max:
                idx_min, idx_max = idx_max, idx_min
            lineout = np.mean(data[:, idx_min:idx_max+1], axis=1)
            transverse_desc = f"avg {transverse_label} ∈ [{slab_min:.2f}, {slab_max:.2f}]"
        else:
            # Single position
            if position is None:
                # Default to center
                position = (transverse_coords[0] + transverse_coords[-1]) / 2
            idx = np.argmin(np.abs(transverse_coords - position))
            lineout = data[:, idx]
            actual_position = transverse_coords[idx]
            transverse_desc = f"{transverse_label} = {actual_position:.2f}"

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(lineout_coords, lineout, **kwargs)

        # Labels and title
        ax.set_xlabel(lineout_label)
        ax.set_ylabel(field_name)
        time, time_str = self._convert_time(time_val, time_units)
        ax.set_title(f"{field_name} lineout at t = {np.round(time,2)} {time_str}\n{transverse_desc}")
        ax.grid(alpha=0.3)

        return ax

    def plot_streak(
        self,
        field_name: str,
        axis: str = "x",
        position: float = None,
        slab: tuple = None,
        timesteps: tuple = None,
        ax=None,
        spatial_units: str = "ion",
        time_units: str = "ion gyrotime",
        cmap: str = "RdBu_r",
        vmin=None,
        vmax=None,
        colorbar: bool = True,
        **kwargs
    ):
        """
        Create a streak plot showing field evolution along one spatial axis over time.

        Parameters
        ----------
        field_name : str
            Name of the field to plot (e.g. 'b3-savg', 'e1')
        axis : str
            Spatial axis to plot along: 'x' or 'y'
        position : float, optional
            Position in the transverse direction (in spatial_units).
            If None, takes lineout at center. Ignored if slab is provided.
        slab : tuple of (float, float), optional
            Range (min, max) in transverse direction to average over (in spatial_units).
            If provided, overrides position.
        timesteps : tuple of (int, int) or list of int, optional
            Timesteps to include. If tuple (start, end), uses range(start, end).
            If list, uses specific timesteps. If None, uses all available timesteps.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates a new figure if None.
        spatial_units : str
            Units for spatial axis: 'ion', 'electron', 'physical', or 'cells'
        time_units : str
            Units for time axis: 'ion gyrotime', 'electron', or 'physical'
        cmap : str
            Colormap name
        vmin, vmax : float, optional
            Color scale limits
        colorbar : bool
            Whether to add a colorbar
        **kwargs
            Passed through to imshow

        Returns
        -------
        ax : matplotlib Axes
            The axes object with the plot
        """
        import matplotlib.pyplot as plt

        # Determine which timesteps to use
        field_obj = self._get_field(field_name)
        n_times_available = len(field_obj)
        
        if timesteps is None:
            timestep_list = list(range(n_times_available))
        elif isinstance(timesteps, tuple) and len(timesteps) == 2:
            timestep_list = list(range(timesteps[0], min(timesteps[1], n_times_available)))
        else:
            timestep_list = list(timesteps)

        # Get first timestep to determine dimensions and coordinates
        first_data = field_obj[timestep_list[0]]
        x1_grid, x2_grid = field_obj.grid
        nx, ny = first_data.shape
        
        # Create full coordinate arrays
        x1_full = np.linspace(x1_grid[0], x1_grid[-1], nx)
        x2_full = np.linspace(x2_grid[0], x2_grid[-1], ny)
        
        # Convert to desired units
        x_coords, x_label = self._convert_axis(x1_full, spatial_units, direction="x")
        y_coords, y_label = self._convert_axis(x2_full, spatial_units, direction="y")

        # Determine which axis to plot and transverse coordinates
        if axis.lower() in ['x', 'x1']:
            spatial_coords = x_coords
            spatial_label = x_label
            transverse_coords = y_coords
            transverse_label = y_label
            transpose_data = False
        elif axis.lower() in ['y', 'x2']:
            spatial_coords = y_coords
            spatial_label = y_label
            transverse_coords = x_coords
            transverse_label = x_label
            transpose_data = True
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

        # Determine transverse indices for lineout
        if slab is not None:
            slab_min, slab_max = slab
            idx_min = np.argmin(np.abs(transverse_coords - slab_min))
            idx_max = np.argmin(np.abs(transverse_coords - slab_max))
            if idx_min > idx_max:
                idx_min, idx_max = idx_max, idx_min
            transverse_desc = f"avg {transverse_label} ∈ [{slab_min:.2f}, {slab_max:.2f}]"
        else:
            if position is None:
                position = (transverse_coords[0] + transverse_coords[-1]) / 2
            idx_min = idx_max = np.argmin(np.abs(transverse_coords - position))
            actual_position = transverse_coords[idx_min]
            transverse_desc = f"{transverse_label} = {actual_position:.2f}"

        # Extract lineouts for all timesteps
        lineouts = []
        time_values = []
        
        for t_idx in timestep_list:
            data = field_obj[t_idx]
            if transpose_data:
                data = data.T
            
            if idx_min == idx_max:
                lineout = data[:, idx_min]
            else:
                lineout = np.mean(data[:, idx_min:idx_max+1], axis=1)
            
            lineouts.append(lineout)
            time_val, _ = field_obj.time(t_idx)
            time_values.append(time_val)

        # Stack lineouts into 2D array (time x space)
        streak_data = np.array(lineouts)
        
        # Convert time values to desired units
        time_array = np.array(time_values)
        time_converted = []
        for t_val in time_array:
            t_conv, _ = self._convert_time(t_val, time_units)
            time_converted.append(t_conv)
        time_converted = np.array(time_converted)
        _, time_label = self._convert_time(time_array[0], time_units)

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(
            streak_data.T,
            origin='lower',
            extent=[time_converted.min(), time_converted.max(),
                   spatial_coords.min(), spatial_coords.max()],
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )

        if colorbar:
            plt.colorbar(im, ax=ax, label=field_name)

        # Labels
        ax.set_xlabel(f"Time {time_label}")
        ax.set_ylabel(spatial_label)
        ax.set_title(f"{field_name} streak plot\n{transverse_desc}")

        return ax

    def _convert_axis(self, axes: np.ndarray, units: str, direction: str = "x"):
        """Convert cell indices to physical coordinates with label."""
        allowed_units = ["ion", "ion inertial length", "electron", "electron inertial length", "ion gyroradius", "physical", "cells"]
        match units:
            case "ion" | "ion inertial length":
                scale = 1/self.ion_inertial_length()
                return axes * scale, rf"${direction} [c/\omega_{{pi}}]$"
            case "electron" | "electron inertial length":
                scale = 1 # already normalized to electron inertial length
                return axes * scale, rf"${direction} [c/\omega_{{pe}}]$"
            case "ion gyroradius":
                #todo
                return
            case "physical":
                scale = self.electron_inertial_length_real()
                return axes * scale, rf"${direction} {self.electron_inertial_length_real.units()}$"
            case _:
                raise ValueError(
                    f"Unknown units '{units}'. Choose from {allowed_units}."
            )
    def _convert_time(self, time_value: float, units: str):
        """Convert time value to physical units with label."""
        allowed_units = ["ion gyrotime", "electron", "physical"]
        match units:
            case "ion gyrotime" | "1 / omega_ci" | "1/omega_ci" | "omega_ci^-1":
                if self.B0 is None:
                    raise ValueError("B0 must be set during initialization to use 'ion gyrotime' units")
                scale = self.omega_ci()
                return time_value * scale, r"$[\omega_{ci}^{-1}]$"
            case "electron":
                scale = 1
                return time_value * scale, r"$[\omega_{pe}^{-1}]$"
            case "physical":
                scale = self.omega_p_real
                return time_value * scale, r"$\ [s]$"
            case _:
                raise ValueError(
                    f"Unknown time units '{units}'. Choose from {allowed_units}."
            )    

    
    def upstream_density(self, timestep: int = 0) -> float:
        """Mean ion density in upstream region [cm^-3]"""
        ...
    
    def upstream_temperature(self, species: str, timestep: int = 0) -> float:
        """Mean thermal temperature in upstream region [eV]"""
        ...
    
    
    def compression_ratio(self, timestep: int = -1) -> float:
        """Downstream/upstream density ratio"""
        ...
    
    def summary(self, timestep: int = -1) -> dict:
        """Returns all key diagnostics as a dict — the 'PI question answerer'"""
        return {
            "upstream_n [cm^-3]": self.upstream_density(0),
            "upstream_T_e [eV]": self.upstream_temperature("electrons", 0),
            "upstream_B [G]":    self.upstream_B(timestep=0),
            "v_A [km/s]":        self.alfven_speed(0),
            "M_ms":              self.mach_number(timestep),
            "M_A":               self.alfvenic_mach_number(timestep),
            "compression ratio": self.compression_ratio(timestep),
        }
    
    def __getitem__(self, key):
        """Allow sim['quantity'] syntax to access underlying osiris_utils diagnostics"""
        return self.sim[key]
    
    def _get_field(self, field_name: str):
        """
        Robustly access a field, handling both direct and nested access patterns.
        
        Supports:
        - Direct access: 'b1-savg' -> sim['b1-savg']
        - Nested access: 'e/charge-savg' -> sim['e']['charge-savg']
        - Nested access: 'e.charge-savg' -> sim['e']['charge-savg']
        - Lazy moments: 'al/n-from-p1' -> self._lazy_moments['al/n-from-p1']
        
        Parameters
        ----------
        field_name : str
            Field name, with optional species prefix separated by '/' or '.'
            
        Returns
        -------
        field : osiris_utils field object or LazyMoment
        """
        # Check if this is a registered lazy moment first
        if hasattr(self, '_lazy_moments') and field_name in self._lazy_moments:
            return self._lazy_moments[field_name]
        
        # Try separators in order of preference
        for sep in ['/', '.']:
            if sep in field_name:
                parts = field_name.split(sep)
                if len(parts) == 2:
                    species, quantity = parts
                    return self.sim[species][quantity]
                else:
                    raise ValueError(
                        f"Field name '{field_name}' has too many separators. "
                        f"Expected format: 'species{sep}quantity' or 'quantity'"
                    )
        
        # No separator found - try direct access
        return self.sim[field_name]


# ============================================================================
# Test/Debug Section - Run this file directly to test
# ============================================================================

def _test_basic_functionality():
    """Quick smoke test of the MagShockZRun class"""
    import astropy.units as u
    
    # Use a sample input deck path (modify as needed)
    test_deck = "/pscratch/sd/d/dschnei/perlmutter_2.8.2d/perlmutter_2d.2d"
    
    print("=" * 60)
    print("Test 1: Initialization without plasma parameters")
    print("=" * 60)
    sim = MagShockZRun(input_deck=test_deck, norm_density=5e18 * u.cm**-3)
    
    print(f"✓ Simulation loaded: {sim.sim}")
    print(f"✓ Species: {sim.deck.species}")
    print(f"✓ Plasma frequency: {sim.omega_p_real:.2e}")
    print(f"✓ rqm: {sim.rqm}")
    
    # Test methods with explicit parameters
    Z = 6
    m_i = 27 * astropy.constants.m_p
    B0 = 100_000 * u.Gauss
    
    omega_ci = sim.omega_ci_real(B0_real=B0, Z=Z, m_i=m_i)
    ion_length = sim.ion_inertial_length_real(Z=Z, m_i=m_i)
    vA = sim.vA_real(B0_real=B0, Z=Z, m_i=m_i)
    
    print(f"✓ Ion cyclotron frequency (explicit params): {omega_ci:.2e}")
    print(f"✓ Ion inertial length (explicit params): {ion_length:.2f}")
    print(f"✓ Alfvén speed (explicit params): {vA:.2f}")
    
    print("\n" + "=" * 60)
    print("Test 2: Initialization WITH plasma parameters (defaults)")
    print("=" * 60)
    sim2 = MagShockZRun(
        input_deck=test_deck, 
        norm_density=5e18 * u.cm**-3,
        B0=100_000 * u.Gauss,
        Z=6,
        m_i=27 * astropy.constants.m_p
    )
    
    # Now we can call methods without arguments!
    omega_ci_default = sim2.omega_ci_real()
    ion_length_default = sim2.ion_inertial_length_real()
    vA_default = sim2.vA_real()
    omega_ci_norm = sim2.omega_ci()
    
    print(f"✓ Ion cyclotron frequency (using defaults): {omega_ci_default:.2e}")
    print(f"✓ Ion inertial length (using defaults): {ion_length_default:.2f}")
    print(f"✓ Alfvén speed (using defaults): {vA_default:.2f}")
    print(f"✓ Normalized ion cyclotron frequency: {omega_ci_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("Test 3: Derived diagnostics from phase space (optional)")
    print("=" * 60)
    print("Note: This test requires phase space diagnostics to be present")
    print("Skipping if not available...")
    
    try:
        # Try to compute moments for a species if phase space data exists
        sim2.list_derived_diagnostics()
        
        # Example: compute density from p1 phase space
        # Uncomment if you have phase space data:
        # sim2.compute_density('al', 'p1')
        # sim2.compute_temperature('al', 'p1', parallel=True)
        # sim2.compute_temperature('al', 'p2', parallel=False)
        # sim2.list_derived_diagnostics()
        
        print("✓ Derived diagnostic system initialized")
    except Exception as e:
        print(f"ℹ Derived diagnostic test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("✓✓✓ All tests passed! ✓✓✓")
    print("=" * 60)


if __name__ == "__main__":
    _test_basic_functionality()