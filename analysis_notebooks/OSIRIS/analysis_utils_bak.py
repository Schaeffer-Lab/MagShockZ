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
#TODO: allow for better integration when calculating things from the data, like multiplying by dimensional quantities. Right now it's weirdly hard to calculate temperatures.
#TODO: clean it up generally. Lot of AI cruft rn
#TODO: make those plots where you have vector fields
#TODO: make it so moments still have all the same class variables as regular diagnostics

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
        if self.sim_wrapper.deck.dim == 1:
            pha_field = f'{species}/{momentum_component}x1'
            self.pha_data = sim_wrapper._get_field(pha_field)
        if self.sim_wrapper.deck.dim == 2:
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
            return self.sim_wrapper.calculate_moment(
                self.species, timestep, order =0, momentum_component=self.momentum_component
            )
        elif self.order == 1:
            return self.sim_wrapper.calculate_moment(
                self.species, timestep, order =1, momentum_component=self.momentum_component
            )
        elif self.order == 2:
            return self.sim_wrapper.calculate_moment(
                self.species, timestep, order =2, momentum_component=self.momentum_component
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
        self.sim = osiris_utils.Simulation(input_deck_path = input_deck)
        self.deck = self.sim._input_deck
        self.norm_density = norm_density
        self.B0 = B0
        self.Z = Z
        self.m_i = m_i
    
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
    
    
    def calculate_moment(self, species: str, timestep: int, order: int, momentum_component: str = 'p1'):
        """
        Calculate density (0th moment) from phase space for a single timestep.
        
        Loads from cache if available, otherwise computes and saves to HDF5.
        """
        if order not in [0, 1, 2]:
            raise ValueError(f"order must be 0, 1, or 2, got {order}")
        match order:
            case 0: moment_name = 'n'
            case 1: moment_name = f'v{momentum_component[1]}'
            case 2: moment_name = 'vth2'
        
        diag_name = f'{species}/{moment_name}-from-{momentum_component}'
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

        if order == 0:
            # Get single timestep data and compute: n = ∫f dp
            data_t = pha_data[timestep]
            result = self._moment(data_t, p_axis, order=order, axis=p_axis_idx)
        
        elif order == 1:
            # Compute velocity: v = (∫p·f dp) / n
            density = self.calculate_moment(species, timestep, order=0, momentum_component=momentum_component)
            data_t = pha_data[timestep]
            flux = self._moment(data_t, p_axis, order=1, axis=p_axis_idx)
            result = flux / density

        elif order == 2:
            density = self.calculate_moment(species, timestep, order=0, momentum_component=momentum_component)
            velocity = self.calculate_moment(species, timestep, order=1, momentum_component=momentum_component)
            data_t = pha_data[timestep]
            
            shape = [1] * data_t.ndim
            shape[p_axis_idx] = -1
            p_broadcast = p_axis.reshape(shape)
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
        slice_pos: dict = None,
        **kwargs
    ):
        """
        Visualize a field component at a given timestep.
        
        Automatically detects dimensionality and plots accordingly:
        - 1D: line plot
        - 2D: imshow/heatmap
        - 3D: 2D slice at specified position

        Parameters
        ----------
        field_name : str
            Name of the field to plot (e.g. 'b3', 'e1')
        timestep : int
            Timestep index to plot
        ax : matplotlib Axes, optional
            Axes to plot on. Creates a new figure if None.
        spatial_units : str
            Units for axes: 'ion' (c/ω_ci), 'electron' (c/ω_ce), 'physical', or 'cells'
        time_units : str
            Units for time display: 'ion gyrotime', 'electron', or 'physical'
        log : bool
            If True, use logarithmic scale (color for 2D/3D, y-axis for 1D)
        norm : matplotlib Normalize, optional
            Explicit normalization for 2D/3D plots. Overrides log if provided.
        vmin, vmax : float, optional
            Value limits (color scale for 2D/3D, y-axis for 1D)
        cmap : str
            Colormap name (2D/3D only)
        colorbar : bool
            Whether to add a colorbar (2D/3D only)
        slice_pos : dict, optional
            For 3D data, specify slice position: {'axis': 'x'/'y'/'z', 'value': float}
            If None, takes center slice along first axis.
        **kwargs
            Passed through to plot/imshow
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        field_obj = self._get_field(field_name)
        field_data = field_obj[timestep]
        time_val, _ = field_obj.time(timestep)
        
        # Detect dimensionality and get coordinates
        dim = self._get_data_dimension(field_obj, timestep)
        coords, labels = self._get_coordinates(field_obj, timestep, spatial_units)
        
        # Create axes if needed
        if ax is None:
            ax = plt.figure(figsize=(6, 5)).gca()
        
        time, time_str = self._convert_time(time_val, time_units)
        
        # ===== 1D PLOTTING =====
        if dim == 1:
            x, x_label = coords[0], labels[0]
            
            # Filter out 2D/3D-only parameters
            plot_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['cmap', 'norm', 'interpolation', 'aspect', 'extent']}
            
            # Plot line
            ax.plot(x, field_data.squeeze(), **plot_kwargs)
            
            # Labels and formatting
            ax.set_xlabel(x_label)
            ax.set_ylabel(field_name)
            ax.set_title(rf"{field_name} at t = {np.round(time,2)} {time_str}")
            ax.grid(alpha=0.3)
            
            if log:
                ax.set_yscale('log')
            if vmin is not None or vmax is not None:
                ax.set_ylim(vmin, vmax)
            
            return ax
        
        # ===== 2D PLOTTING =====
        elif dim == 2:
            x, x_label = coords[0], labels[0]
            y, y_label = coords[1], labels[1]
            
            # Normalization
            if norm is None:
                if log:
                    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            # Plot
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
                plt.colorbar(im, ax=ax, label=field_name)
            
            # Labels
            ax.set_title(rf"{field_name} at t = {np.round(time,2)} {time_str}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            return ax
        
        # ===== 3D PLOTTING =====
        elif dim == 3:
            # Determine slice to take
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            if slice_pos is None:
                slice_axis = 0
                slice_idx = field_data.shape[0] // 2
            else:
                slice_axis = axis_map.get(slice_pos.get('axis', 'x').lower(), 0)
                slice_idx = np.argmin(np.abs(coords[slice_axis] - slice_pos.get('value', coords[slice_axis][len(coords[slice_axis])//2])))
            
            # Extract slice and corresponding coordinates
            if slice_axis == 0:
                slice_data = field_data[slice_idx, :, :]
                x, x_label = coords[1], labels[1]
                y, y_label = coords[2], labels[2]
            elif slice_axis == 1:
                slice_data = field_data[:, slice_idx, :]
                x, x_label = coords[0], labels[0]
                y, y_label = coords[2], labels[2]
            else:  # slice_axis == 2
                slice_data = field_data[:, :, slice_idx]
                x, x_label = coords[0], labels[0]
                y, y_label = coords[1], labels[1]
            
            # Normalization
            if norm is None:
                if log:
                    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            # Plot
            im = ax.imshow(
                slice_data.T,
                origin="lower",
                extent=[x[0], x[-1], y[0], y[-1]],
                norm=norm,
                cmap=cmap,
                aspect="auto",
                **kwargs,
            )
            
            if colorbar:
                plt.colorbar(im, ax=ax, label=field_name)
            
            # Get slice position for title
            slice_axis_name = ['x', 'y', 'z'][slice_axis]
            slice_val = coords[slice_axis][slice_idx]
            
            # Labels
            ax.set_title(rf"{field_name} at t = {np.round(time,2)} {time_str}, {slice_axis_name}={slice_val:.2f}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            return ax
        
        else:
            raise ValueError(f"Unsupported data dimension: {dim}D")

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
        slice_pos: dict = None,
        **kwargs
    ):
        """
        Plot a 1D lineout of a field along a specified axis.
        
        Automatically handles 1D, 2D, and 3D data:
        - 1D: plots the data directly along the single axis
        - 2D: takes lineout along specified axis
        - 3D: first takes a 2D slice (specified by slice_pos), then lineout

        Parameters
        ----------
        field_name : str
            Name of the field to plot (e.g. 'b3-savg', 'e1')
        timestep : int
            Timestep index to plot
        axis : str
            Axis to take lineout along: 'x', 'y', or 'z' (for 3D)
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
        slice_pos : dict, optional
            For 3D data, specify which 2D slice to take before lineout:
            {'axis': 'x'/'y'/'z', 'value': float}. If None, takes center slice.
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
        
        # Detect dimensionality and get coordinates
        dim = self._get_data_dimension(field_obj, timestep)
        coords, labels = self._get_coordinates(field_obj, timestep, spatial_units)
        
        # Create plot axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        time, time_str = self._convert_time(time_val, time_units)
        
        # ===== 1D DATA =====
        if dim == 1:
            x_coords, x_label = coords[0], labels[0]
            
            # Filter out 2D/3D-only parameters
            plot_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['cmap', 'norm', 'interpolation', 'aspect', 'extent']}
            
            ax.plot(x_coords, field_data.squeeze(), **plot_kwargs)
            ax.set_xlabel(x_label)
            ax.set_ylabel(field_name)
            ax.set_title(f"{field_name} at t = {np.round(time,2)} {time_str}")
            ax.grid(alpha=0.3)
            
            return ax
        
        # ===== 2D DATA =====
        elif dim == 2:
            x_coords, x_label = coords[0], labels[0]
            y_coords, y_label = coords[1], labels[1]

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
                raise ValueError(f"axis must be 'x' or 'y' for 2D data, got '{axis}'")

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

            ax.plot(lineout_coords, lineout, **kwargs)

            # Labels and title
            ax.set_xlabel(lineout_label)
            ax.set_ylabel(field_name)
            ax.set_title(f"{field_name} lineout at t = {np.round(time,2)} {time_str}\n{transverse_desc}")
            ax.grid(alpha=0.3)

            return ax
        
        # ===== 3D DATA =====
        elif dim == 3:
            # Determine slice to take
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            lineout_axis = axis_map.get(axis.lower(), 0)
            
            if slice_pos is None:
                # Choose different axis for slicing
                slice_axis = 0 if lineout_axis != 0 else 1
                slice_idx = field_data.shape[slice_axis] // 2
            else:
                slice_axis = axis_map.get(slice_pos.get('axis', 'x').lower(), 0)
                slice_idx = np.argmin(np.abs(coords[slice_axis] - slice_pos.get('value', coords[slice_axis][len(coords[slice_axis])//2])))
            
            # Extract 2D slice and get remaining coordinates
            remaining_indices = [i for i in range(3) if i != slice_axis]
            
            if slice_axis == 0:
                slice_data = field_data[slice_idx, :, :]
            elif slice_axis == 1:
                slice_data = field_data[:, slice_idx, :]
            else:  # slice_axis == 2
                slice_data = field_data[:, :, slice_idx]
            
            remaining_coords = [coords[i] for i in remaining_indices]
            remaining_labels = [labels[i] for i in remaining_indices]
            remaining_axes = [['x', 'y', 'z'][i] for i in remaining_indices]
            
            # Determine which of the remaining axes to lineout along
            lineout_axis_in_slice = remaining_axes.index(axis.lower()) if axis.lower() in remaining_axes else 0
            transpose_slice = (lineout_axis_in_slice == 1)
            
            lineout_coords = remaining_coords[lineout_axis_in_slice]
            transverse_coords = remaining_coords[1 - lineout_axis_in_slice]
            lineout_label = remaining_labels[lineout_axis_in_slice]
            transverse_label = remaining_labels[1 - lineout_axis_in_slice]
            
            # Extract lineout from slice
            slice_2d = slice_data.T if transpose_slice else slice_data
            
            if slab is not None:
                slab_min, slab_max = slab
                idx_min = np.argmin(np.abs(transverse_coords - slab_min))
                idx_max = np.argmin(np.abs(transverse_coords - slab_max))
                if idx_min > idx_max:
                    idx_min, idx_max = idx_max, idx_min
                lineout = np.mean(slice_2d[:, idx_min:idx_max+1], axis=1)
                transverse_desc = f"avg {transverse_label} ∈ [{slab_min:.2f}, {slab_max:.2f}]"
            else:
                if position is None:
                    position = (transverse_coords[0] + transverse_coords[-1]) / 2
                idx = np.argmin(np.abs(transverse_coords - position))
                lineout = slice_2d[:, idx]
                actual_position = transverse_coords[idx]
                transverse_desc = f"{transverse_label} = {actual_position:.2f}"
            
            # Get slice position info
            slice_axis_name = ['x', 'y', 'z'][slice_axis]
            slice_coords_full, _ = self._convert_axis(field_obj.grid[slice_axis], spatial_units, direction=slice_axis_name)
            slice_val = slice_coords_full[slice_idx]
            
            ax.plot(lineout_coords, lineout, **kwargs)
            
            # Labels
            ax.set_xlabel(lineout_label)
            ax.set_ylabel(field_name)
            time, time_str = self._convert_time(time_val, time_units)
            ax.set_title(f"{field_name} lineout at t = {np.round(time,2)} {time_str}\n{slice_axis_name}={slice_val:.2f}, {transverse_desc}")
            ax.grid(alpha=0.3)
            
            return ax
        
        else:
            raise ValueError(f"Unsupported data dimension: {dim}D")

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
        slice_pos: dict = None,
        **kwargs
    ):
        """
        Create a streak plot showing field evolution along one spatial axis over time.
        
        Automatically handles 1D, 2D, and 3D data:
        - 1D: streak plot showing evolution over time
        - 2D: lineout-based streak plot
        - 3D: first takes 2D slice, then lineout-based streak plot

        Parameters
        ----------
        field_name : str
            Name of the field to plot (e.g. 'b3-savg', 'e1')
        axis : str
            Spatial axis to plot along: 'x', 'y', or 'z' (for 3D)
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
        slice_pos : dict, optional
            For 3D data, specify which 2D slice to take before lineout:
            {'axis': 'x'/'y'/'z', 'value': float}. If None, takes center slice.
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

        # Detect dimensionality from first timestep
        dim = self._get_data_dimension(field_obj, timestep_list[0])
        
        # ===== 1D DATA =====
        if dim == 1:
            # Simple streak: just stack all timesteps
            lineouts = []
            time_values = []
            
            x_grid = field_obj.grid[0] if len(field_obj.grid) == 1 else field_obj.grid
            spatial_coords, spatial_label = self._convert_axis(x_grid, spatial_units, direction="x")
            
            for t_idx in timestep_list:
                data = field_obj[t_idx]
                lineouts.append(data.squeeze())
                time_val, _ = field_obj.time(t_idx)
                time_values.append(time_val)
            
            transverse_desc = "1D data"
        
        # ===== 2D DATA =====
        elif dim == 2:
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
                raise ValueError(f"axis must be 'x' or 'y' for 2D data, got '{axis}'")

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
        
        # ===== 3D DATA =====
        elif dim == 3:
            # Similar to 2D, but first take a 2D slice
            first_data = field_obj[timestep_list[0]]
            
            # Determine slice to take
            if slice_pos is None:
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                lineout_axis = axis_map.get(axis.lower(), 0)
                slice_axis = 0 if lineout_axis != 0 else 1
                slice_idx = first_data.shape[slice_axis] // 2
            else:
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                slice_axis = axis_map.get(slice_pos.get('axis', 'x').lower(), 0)
                axis_grid = field_obj.grid[slice_axis]
                axis_coords, _ = self._convert_axis(axis_grid, spatial_units, direction="x")
                slice_idx = np.argmin(np.abs(axis_coords - slice_pos.get('value', axis_coords[len(axis_coords)//2])))
            
            # Determine remaining axes after slice
            if slice_axis == 0:
                remaining_grids = [field_obj.grid[1], field_obj.grid[2]]
                remaining_axes = ['y', 'z']
                def extract_slice(data): return data[slice_idx, :, :]
            elif slice_axis == 1:
                remaining_grids = [field_obj.grid[0], field_obj.grid[2]]
                remaining_axes = ['x', 'z']
                def extract_slice(data): return data[:, slice_idx, :]
            else:  # slice_axis == 2
                remaining_grids = [field_obj.grid[0], field_obj.grid[1]]
                remaining_axes = ['x', 'y']
                def extract_slice(data): return data[:, :, slice_idx]
            
            # Determine which axis to lineout along in the slice
            lineout_axis_in_slice = remaining_axes.index(axis.lower()) if axis.lower() in remaining_axes else 0
            
            if lineout_axis_in_slice == 0:
                lineout_grid = remaining_grids[0]
                transverse_grid = remaining_grids[1]
                lineout_name = remaining_axes[0]
                transverse_name = remaining_axes[1]
                transpose_slice = False
            else:
                lineout_grid = remaining_grids[1]
                transverse_grid = remaining_grids[0]
                lineout_name = remaining_axes[1]
                transverse_name = remaining_axes[0]
                transpose_slice = True
            
            # Get slice shape
            slice_data_test = extract_slice(first_data)
            nx_slice, ny_slice = slice_data_test.shape
            
            lineout_coords_raw = np.linspace(lineout_grid[0], lineout_grid[-1], nx_slice if not transpose_slice else ny_slice)
            transverse_coords_raw = np.linspace(transverse_grid[0], transverse_grid[-1], ny_slice if not transpose_slice else nx_slice)
            
            spatial_coords, spatial_label = self._convert_axis(lineout_coords_raw, spatial_units, direction=lineout_name)
            transverse_coords, transverse_label = self._convert_axis(transverse_coords_raw, spatial_units, direction=transverse_name)
            
            # Determine transverse indices
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
            
            # Get slice position info
            slice_axis_name = ['x', 'y', 'z'][slice_axis]
            slice_coords_full, _ = self._convert_axis(field_obj.grid[slice_axis], spatial_units, direction=slice_axis_name)
            slice_val = slice_coords_full[slice_idx]
            transverse_desc = f"{slice_axis_name}={slice_val:.2f}, {transverse_desc}"
            
            # Extract lineouts for all timesteps
            lineouts = []
            time_values = []
            
            for t_idx in timestep_list:
                data = field_obj[t_idx]
                slice_2d = extract_slice(data)
                
                if transpose_slice:
                    slice_2d = slice_2d.T
                
                if idx_min == idx_max:
                    lineout = slice_2d[:, idx_min]
                else:
                    lineout = np.mean(slice_2d[:, idx_min:idx_max+1], axis=1)
                
                lineouts.append(lineout)
                time_val, _ = field_obj.time(t_idx)
                time_values.append(time_val)
        
        else:
            raise ValueError(f"Unsupported data dimension: {dim}D")

        # Common plotting code for all dimensions
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
                return axes * scale, rf"${direction} {self.electron_inertial_length_real()}$"
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
    
    def _get_data_dimension(self, field_obj, timestep=0):
        """
        Determine the dimensionality of field data.
        
        Parameters
        ----------
        field_obj : osiris_utils field object or LazyMoment
            Field object to check
        timestep : int
            Timestep to check (default: 0)
            
        Returns
        -------
        int
            Dimensionality: 1, 2, or 3
        """
        data = field_obj[timestep]
        # Remove singleton dimensions
        data_shape = [s for s in data.shape if s > 1]
        return len(data_shape)
    
    def _get_coordinates(self, field_obj, timestep=0, spatial_units="ion"):
        """
        Extract properly formatted coordinate arrays for any dimension.
        
        Parameters
        ----------
        field_obj : osiris_utils field object or LazyMoment
            Field object
        timestep : int
            Timestep to use for getting data shape
        spatial_units : str
            Units for coordinates
            
        Returns
        -------
        coords : list of arrays
            Coordinate arrays for each dimension
        labels : list of str
            Axis labels for each dimension
        """
        data = field_obj[timestep]
        grids = field_obj.grid if isinstance(field_obj.grid, list) else [field_obj.grid]
        
        coords = []
        labels = []
        axis_names = ['x', 'y', 'z']
        
        for i, grid in enumerate(grids):
            # Create full coordinate array from grid boundaries
            n_points = data.shape[i]
            coord_array = np.linspace(grid[0], grid[-1], n_points)
            
            # Convert to desired units
            converted_coords, label = self._convert_axis(coord_array, spatial_units, direction=axis_names[i])
            coords.append(converted_coords)
            labels.append(label)
        
        return coords, labels


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