"""
FLASH-OSIRIS Interface

This package provides tools to process 3D FLASH simulation data, apply user-defined edits and normalization, 
and generate input files for OSIRIS simulations.

It needs my little yt plugin in order to work... If you are a person who isn't david and you are reading this,
you probably don't have it. Email me at dschneidinger@g.ucla.edu
"""

import yt
from pathlib import Path
import numpy as np
"""
Args we need so far
- FLASH_data: str -> The name of the FLASH file to load.
- inputfile_name: str -> The name of the OSIRIS input file to generate.
- normalizations: dict -> A dictionary containing how you want each field to be normalized.
- reference_density: float -> The reference density in cm^-3.

OPTARGS
- ion_2: The second ion species to be used in the simulation. Default is 'Si'.
- B_background: Background magnetic field strength in x direction. Default is 0.
- rqm: Mass ratio of ions to electrons. Default is 100.
"""

class FLASH_OSIRIS:
    def __init__(self, FLASH_data, inputfile_name, reference_density, 
                 ion_2 = "Si", B_background = 0, rqm = 100, osiris_dims = 2, ppc = 80, start_point = [0, 240], theta = np.pi/2, xmax = 7100):
        import sys
        sys.path.append("../src")
        yt.enable_plugins()
        self.FLASH_data: str = Path(FLASH_data)
        if not self.FLASH_data.exists():
            raise ValueError(f"{self.FLASH_data} does not exist")
        self.inputfile_name: str = inputfile_name
        self.ion_1 = "Al"
        self.ion_2: str = ion_2
        self.B0: float = B_background
        self.rqm: int = rqm
        self.n0: float = reference_density
        self.osiris_dims: int = osiris_dims
        self.ppc: int = ppc
        self.xmax = xmax # osiris units... DEFINITELY need to change this later so it's not hard coded
        self.start_point = start_point # start point in osiris units. Need to change this later so it's not hard coded
        self.theta = theta # angle that ray makes with the x axis [radians]. Also need to change this later so it's not hard coded

        self.e = 4.80320425e-10 # [statC] = [cm^3/2⋅g^1/2⋅s^−1]
        self.m_e = 9.1093837139e-28 # [g]
        self.c = 2.99792458e10 # [cm/s]

        self.omega_pe = np.sqrt(4 * np.pi * self.n0 * self.e**2 / self.m_e) # in rad/s

        self.ds = yt.load_for_osiris(self.FLASH_data, rqm = self.rqm, B_background=self.B0, ion_2 = self.ion_2)  # Load the FLASH data using yt, this handles field derivation as well
        
        level = 0
        self.dims = self.ds.domain_dimensions * self.ds.refine_by**level

        self.all_data = self.ds.covering_grid(
            level, #level of refinement
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1,
        )
        self.proj_dir = Path(f"/home/dschneidinger/MagShockZ")
        self.output_dir = Path(self.proj_dir / "input_files" / self.inputfile_name)

        print('*'*10 + " FLASH-OSIRIS interface initialized " + '*'*10)
        print(f"FLASH data: {self.FLASH_data}")
        print(f"Input file name: {self.inputfile_name}")
        print(f"Reference density: {self.n0} cm^-3")
        print(f"Second ion species: {self.ion_2}")
        print(f"External background magnetic field: {self.B0} Gauss")
        print(f"OSIRIS mass ratio: {self.rqm}")
        print(f"OSIRIS dimensions: {self.osiris_dims}")
        print(f"Particles per cell: {self.ppc}")
        print(f"Start point: {self.start_point} [c / wpe]")
        print(f"Angle: {self.theta} (only used in 1D)")
        print(f"Xmax: {self.xmax} (only used in 1D)")
        print(f"Output directory: {self.output_dir}")
        print('*'*10 + " FLASH-OSIRIS interface initialized " + '*'*10)


    def get_filename(self):
        return self.FLASH_data
    def get_inputfile_name(self):
        return self.inputfile_name
    def get_normalizations(self):
        return self.normalizations
    def get_reference_density(self):
        return self.n0
    def get_ion_2(self):
        return self.ion_2
    def get_B_background(self):
        return self.B0
    def get_rqm(self):
        return self.rqm
    def get_osiris_dims(self):
        return self.osiris_dims
    def get_ds(self):
        return self.ds
    def get_all_data(self):
        return self.all_data
    def get_output_dir(self):
        return self.output_dir
    def get_proj_dir(self):
        return self.proj_dir
    
    def set_inputfile_name(self, inputfile_name):
        self.inputfile_name = inputfile_name
    def set_normalizations(self, normalizations):
        self.normalizations = normalizations
    def set_reference_density(self, reference_density):
        self.n0 = reference_density
    def set_ion_2(self, ion_2):
        self.ion_2 = ion_2
    def set_B_background(self, B_background):
        self.B0 = B_background
    def set_rqm(self, rqm):
        self.rqm = rqm
    def set_osiris_dims(self, osiris_dims):
        self.osiris_dims = osiris_dims

    def calculate_numbers(self, n_gyroperiods = 10, n_debye = 7.14):
        self.x = self.all_data['flash', 'x'][:, 0, 0] * self.omega_pe / self.c # x axis in osiris units
        self.y = self.all_data['flash', 'y'][0, :, 0] * self.omega_pe / self.c # y axis in osiris units
        self.z = self.all_data['flash', 'z'][0, 0, :] * self.omega_pe / self.c # z axis in osiris units


        
        self.debye = 7.43e-2 * np.sqrt(self.all_data['flash', 'tele'][-1, -1, 0] / self.n0) # in cm, from NRL formulary
        print(f"background temp in eV: {self.all_data['flash', 'tele'][-1, -1, 0] * 8.617333262e-5}")

        self.dx = self.debye / (self.c / self.omega_pe) * n_debye # now in osiris units
        self.dt = self.dx * 0.98

        Al_charge_state = 13 ### DEFINITELY need to change this later...
        # from Wikipedia
        aluminum_molecular_weight = 26.981 
        m_p = 1.6726219e-24 # [g]
        rqm_real = int(aluminum_molecular_weight*m_p/(Al_charge_state*self.m_e))

        self.normalizations = {
            'edens':self.n0,
            f'{str.lower(self.ion_1)}dens':self.n0,
            f'{str.lower(self.ion_2)}dens':self.n0,
            'Bx_int':(self.omega_pe*self.m_e*self.c)/self.e,
            'By_int':(self.omega_pe*self.m_e*self.c)/self.e,
            'Bz_int':(self.omega_pe*self.m_e*self.c)/self.e,
            'magx':(self.omega_pe*self.m_e*self.c)/self.e,
            'magy':(self.omega_pe*self.m_e*self.c)/self.e,
            'magz':(self.omega_pe*self.m_e*self.c)/self.e,
            'Ex':(self.omega_pe*self.m_e*self.c**2)/self.e, 
            'Ey':(self.omega_pe*self.m_e*self.c**2)/self.e, 
            'Ez':(self.omega_pe*self.m_e*self.c**2)/self.e,
            'v_ix': self.c/np.sqrt(rqm_real/self.rqm),
            'v_iy': self.c/np.sqrt(rqm_real/self.rqm),
            'v_iz': self.c/np.sqrt(rqm_real/self.rqm),
            'v_ex': self.c/np.sqrt(rqm_real/self.rqm),
            'v_ey': self.c/np.sqrt(rqm_real/self.rqm),
            'v_ez': self.c/np.sqrt(rqm_real/self.rqm),
            'vthele':self.c, # already normalized
            f'vth{str.lower(self.ion_1)}':self.c, # already normalized
            f'vth{str.lower(self.ion_2)}':self.c, # already normalized
        }
        if self.B0 != 0:
            self.gyrotime = self.rqm / (self.B0 / self.normalizations['Bx_int'])

        else:
            self.gyrotime = self.rqm / (self.all_data['flash', 'magx'][-1, -1, 0] / self.normalizations['magx']) # in osiris units
        self.tmax = self.gyrotime * n_gyroperiods  

    def save_slices(self, normal_axis = "z"):
        '''
        args: normalizations: dict, target_index: int
        normalizations: dict
            key: str, field name
            value: float, normalization factor. Note: function automatically divides by this factor


        Note: Density data will be output as a numpy array because OSIRIS uses its own interpolator for density data
        '''
        import pickle
        from scipy.interpolate import RegularGridInterpolator
        
        interp_dir = self.output_dir / "interp"
        if not (interp_dir).exists():
            (interp_dir).mkdir(parents=True)
        if normal_axis not in ["x", "y", "z"]:
            raise ValueError("normal_axis must be one of 'x', 'y', or 'z'")
        match normal_axis:
            case "x":
                normal = 0
            case "y":
                normal = 1
            case "z":
                normal = 2
        middle_index=self.dims[normal]//2

        chunk_size = 128  # Adjust this based on your memory constraints

        # Not implemented for x or y yet
        for (field, normalization) in self.normalizations.items():
            print(f"Processing {field} with normalization {normalization}")
            print(self.ds.field_list)
            if not isinstance(normalization, (int, float)):
                raise ValueError(f"{normalization} is not a valid normalization factor")
            
            field_data = np.zeros(self.all_data['flash', field][:, :, middle_index].shape)
            for i in range(0, self.all_data['flash', field].shape[0], chunk_size):
                end = min(i + chunk_size, self.all_data['flash', field].shape[0])
                field_data_chunk = np.array(self.all_data['flash', field][i:end, :, middle_index]) / normalization
                field_data[i:end, :] = field_data_chunk

            if field.endswith('dens'):
                lower_bound = 0.01
                field_data[field_data < lower_bound] = 0
                np.save(f"{interp_dir}/{field}.npy", field_data)
            else:
                x = self.all_data['flash', 'x'][:, 0, 0] * self.omega_pe / self.c
                y = self.all_data['flash', 'y'][0, :, 0] * self.omega_pe / self.c
                interp1 = RegularGridInterpolator((y, x), field_data.T, method='linear', bounds_error=True, fill_value=0)
                with open(f"{interp_dir}/{field}.pkl", "wb") as f:
                    pickle.dump(interp1, f)


    def write_input_file(self):
        import pickle

        end_point = [self.xmax * np.cos(self.theta), self.xmax * np.sin(self.theta)] + self.start_point

        with open(self.output_dir / "interp/vthele.pkl", "wb") as f:
            vthele = pickle.load(f)
        with open(self.output_dir / "interp/vthal.pkl", "wb") as f:
            vthal = pickle.load(f)
        with open(self.output_dir / "interp/vthsi.pkl", "wb") as f:
            vthsi = pickle.load(f)
        # Write the input file to generate the OSIRIS input file
        
        with open(f'{self.output_dir / self.inputfile_name}.{self.osiris_dims}d', "w") as f:
            f.write(f'''!----------------- Input deck illustrating the Python-Fortran interface ------------------
! To run this input deck as is, first put the input deck, OSIRIS executable, and the
! py-script-{self.osiris_dims}d.py file all in the same directory.  Next, do `export PYTHONPATH=.` to set the Python
! path to the directory that contains the py-script-{self.osiris_dims}d.py file (current directory). Finally,
! execute `./osiris-{self.osiris_dims}D.e {self.inputfile_name}` to run the simulation, which will use the
! py-script-{self.osiris_dims}d.py and interp.npy files to set various field and particle data.
!-----------------------------------------------------------------------------------------

!----------global simulation parameters----------
simulation 
\u007b
 parallel_io = "mpi",
\u007d

!--------the node configuration for this simulation--------
node_conf 
\u007b
 node_number = 16, ! edit this to the number of nodes you are using
 n_threads=2,
\u007d


!----------spatial grid----------
grid
\u007b
 nx_p = {int(self.xmax/self.dx)}, ! fix this later
\u007d

!----------time step and global data dump timestep number----------
time_step
\u007b
 dt     =   {self.dt},
 ndump  =   {self.tmax / (400* self.dt)}, ! 400 dumps total
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
\u007b ! This is euclidean distance, not the span in y direction
 ! Start point in 2D plane is specified in py-script-1d
 xmin =  0, ! This should always be == 0
 xmax =  {self.xmax}, ! fix this later
\u007d

!----------time limits ----------
time
\u007b
 tmin = 0.0,
 tmax  = {self.tmax},
\u007d

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
 
 ext_fld = "static",
 type_ext_b(1:3) = "uniform", "uniform", "uniform",
 ext_b0(1:3) = {self.B0 / self.normalizations['Bx_int']}, 0.0, 0.0,
\u007d

!----------boundary conditions for em-fields ----------
emf_bound
\u007b
 type(1:2,1) =   "open", "open", ! modify this as needed
\u007d

!----------- electo-magnetic field diagnostics ---------
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
 ! ndump_fac_lineout = 1,            ! do lineouts/slices at every timestep
 ndump_fac_ene_int = 1,
 n_ave(1:1) = 4,                     ! average/envelope 8 cells (2x2x2)
 !n_tavg = 5,                        ! average 5 iterations for time averaged diagnostics 
\u007d

!----------number of particle species----------
particles
\u007b
 interpolation = "quadratic",
 num_species = 3,
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
 uth_bnd(1:3,1,1) = {vthele((self.start_point[1], self.start_point[0]))}, {vthele((self.start_point[1], self.start_point[0]))}, {vthele((self.start_point[1], self.start_point[0]))}, 
 uth_bnd(1:3,2,1) = {vthele((end_point[1], end_point[0]))}, {vthele((end_point[1], end_point[0]))}, {vthele((end_point[1], end_point[0]))},
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

!----------information for {self.ion_1} ions----------
species
\u007b
 name = "{self.ion_1}",
 rqm = {self.rqm},
 num_par_x = {self.ppc},
 init_type = "python",
\u007d

!----------inital proper velocities - {self.ion_1}-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 uth_py_func = "set_uth_{self.ion_1}", ! Name of function in the Python file to call
 
 ! use_spatial_ufl = .true.,
 ufl_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 ufl_py_func = "set_ufl_i", ! Name of function in the Python file to call
\u007d

!----------density profile for Aluminum----------
profile
\u007b
 py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 py_func = "set_density_{self.ion_1}", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for {self.ion_1}----------
spe_bound ! hard coded for now... shouldn't be too hard to generalize
\u007b
 type(1:2,1) =   "thermal","thermal",
 uth_bnd(1:3,1,1) = {vthal((self.start_point[1], self.start_point[0]))}, {vthal((self.start_point[1], self.start_point[0]))}, {vthal((self.start_point[1], self.start_point[0]))}, 
 uth_bnd(1:3,2,1) = {vthal((end_point[1], end_point[0]))}, {vthal((end_point[1], end_point[0]))}, {vthal((end_point[1], end_point[0]))},
\u007d

!----------diagnostic for Aluminum----------
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
\u007d


!----------information for Silicon----------
species
\u007b
 name = "{self.ion_2}",
 rqm={self.rqm * 28.08/26.982}, ! This is extremely hard coded, but this is assuming a silicon piston w aluminum background
 num_par_x = {self.ppc},
 init_type = "python",
\u007d

!----------inital proper velocities - Silicon-----------------
udist
\u007b
 use_spatial_uth = .true.,
 uth_py_mod = "py-script-1d", ! Name of Python file
 uth_py_func = "set_uth_{self.ion_2}", ! Name of function in the Python file to call
 
 ufl_py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 ufl_py_func = "set_ufl_i", ! Name of function in the Python file to call
\u007d

!----------density profile for {self.ion_2}----------
profile
\u007b
 py_mod = "py-script-{self.osiris_dims}d", ! Name of Python file
 py_func = "set_density_{self.ion_2}", ! Name of function in the Python file to call
\u007d

!----------boundary conditions for {self.ion_2}----------
spe_bound ! again, this is hard coded, but it shouldn't be too hard to generalize
\u007b
 type(1:2,1) =   "thermal","thermal",
 uth_bnd(1:3,1,1) = {vthsi((self.start_point[1], self.start_point[0]))}, {vthsi((self.start_point[1], self.start_point[0]))}, {vthsi((self.start_point[1], self.start_point[0]))}, 
 uth_bnd(1:3,2,1) = {vthsi((end_point[1], end_point[0]))}, {vthsi((end_point[1], end_point[0]))}, {vthsi((end_point[1], end_point[0]))},
\u007d


!----------diagnostic for this {self.ion_2}----------
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
 ps_xmax(1:1) = {self.xmax}
 ps_np = 4096,
 ps_nx = 4096,
 !if_ps_p_auto(1:3) = .true., .true., .true.,
 phasespaces = "p1x1", "p2x1","p3x1",
\u007d

!----------diagnostic for currents----------
diag_current
\u007b
 !ndump_fac = 1,
 !reports = "j1", "j2", "j3" , 
\u007d

!---------- end of osiris input file -------------")''')      
    def write_python_script(self):

    
        # Write the python script to generate the input file
        with open(self.output_dir / "input.py", "w") as f:
            f.write(f'''import numpy as np
import pickle

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------

# Define the start point for the ray in OSIRIS units
start_point = {self.start_point} # start point in OSIRIS units
theta = {self.theta} # angle that ray makes with the x axis [radians]

# Parameters of FLASH simualation
ions_1 = "{self.ion_1}" 
ions_2 = "{self.ion_2}"

box_bounds = \u007b
    "xmin": {self.x[0]},
    "xmax": {self.x[-1]},
    "ymin": {self.y[0]},
    "ymax": {self.y[-1]},
\u007d

def set_fld_int( STATE ):
    """
    Function to set the field data in th, STATE dictionary based on the field component.
    
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
        "e1": ("interp/Ex_int.pkl", "interp/Ey_int.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex_int.pkl", "interp/Ey_int.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez_int.pkl", None, lambda Ez, _: Ez),
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
#-----------------------------------------------------------------------------------------
def set_uth_{self.ion_1}( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """
    if f"vth{self.ion_1}" not in STATE.keys():
        with open(f'interp/vth{self.ion_1}.pkl', "rb") as f:
            STATE[f'vth{self.ion_1}'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE[f'vth{self.ion_1}']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE[f'vth{self.ion_1}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{self.ion_1}']((y_positions, x_positions))
    return
#-----------------------------------------------------------------------------------------

def set_uth_{self.ion_2}( STATE ):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """
    # print("calling set_uth_e...")
    if f"vth{self.ion_2}" not in STATE.keys():
        with open(f'interp/vth{self.ion_2}.pkl', "rb") as f:
            STATE[f'vth{self.ion_2}'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE[f'vth{self.ion_2}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{self.ion_2}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{self.ion_2}']((y_positions, x_positions))

    return
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
    if filename == "interp/edens.npy":
        density_grid = np.load(f"interp/{self.ion_1}dens.npy") + np.load(f"interp/{self.ion_2}dens.npy")
    else:
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

#-----------------------------------------------------------------------------------------
def set_density_{self.ion_1}( STATE ):
    """
    Set the aluminum density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting {str.upper(self.ion_1)} DENSITY...")
    load_and_interpolate_density(STATE, f"interp/{self.ion_1}dens.npy")

#-----------------------------------------------------------------------------------------
def set_density_{self.ion_2}( STATE ):
    """
    Set the silicon density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting {str.upper(self.ion_2)} DENSITY...")
    load_and_interpolate_density(STATE, f"interp/{self.ion_2}dens.npy")
                    
''')
            
    def write_everything(self):
        # Main function to run the interface
        self.calculate_numbers()
        self.save_slices()
        self.write_input_file()
        self.write_python_script()

        print(f"Input file {self.inputfile_name} and python script input.py have been generated in {self.output_dir}")

if __name__ == "__main__":
    # Example usage
    interface = FLASH_OSIRIS(
        FLASH_data="/home/dschneidinger/shared/data/VAC_DEREK3D_20um/MagShockZ_hdf5_chk_0006",
        inputfile_name="magshockz-v3.2.1d",
        osiris_dims=1,
        reference_density=5e18,
        ion_2="Si",
        ppc=100,
        start_point=[0, 240],
        theta=np.pi / 2,
        # normalizations='default',
        B_background=75000, # [G]
    )
    interface.write_everything()