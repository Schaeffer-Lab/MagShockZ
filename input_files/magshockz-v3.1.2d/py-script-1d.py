import numpy as np
import pickle

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------

# Define the start point for the ray in OSIRIS units
start_point = [0, 240]
theta = np.pi/2 # angle that ray makes with the x axis [radians]

# Parameters of FLASH simualation
ions_1 = 'al'
ions_2 = 'si'

box_bounds = {
    "xmin": -3357.0,
    "xmax": 3357.0,
    "ymin": -307.0,
    "ymax": 8407.0,
}

def set_fld_int( STATE ):
    """
    Function to set the field data in th, STATE dictionary based on the field component.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including field component and positional boundary data.
    """
    # print("calling set_fld...")
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    print(f'x_bnd: {x_bnd}')

    # Shape of the data array
    nx = STATE["data"].shape

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace(x_bnd[0][0] * np.cos(theta), x_bnd[0][1] * np.cos(theta), nx[0], endpoint=True) + start_point[0]
    y = np.linspace(x_bnd[0][0] * np.sin(theta), x_bnd[0][1] * np.sin(theta), nx[0], endpoint=True) + start_point[1]

    # Dictionary to map field components to their respective filenames and operations
    field_map = {
        "e1": ("interp/Ex_int.pkl", "interp/Ey_int.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex_int.pkl", "interp/Ey_int.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez_int.pkl", None, lambda Ez, _: Ez),
        "b1": ("interp/Bx_int.pkl", "interp/By_int.pkl", lambda Bx, By: np.cos(theta) * Bx + np.sin(theta) * By),
        "b2": ("interp/Bx_int.pkl", "interp/By_int.pkl", lambda Bx, By: -np.sin(theta) * Bx + np.cos(theta) * By),
        "b3": ("interp/Bz_int.pkl", None, lambda Bz, _: Bz)
    }

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
    print(f'x_bnd: {x_bnd}')

    # Shape of the data array
    nx = STATE["data"].shape

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace(x_bnd[0][0] * np.cos(theta), x_bnd[0][1] * np.cos(theta), nx[0], endpoint=True) + start_point[0]
    y = np.linspace(x_bnd[0][0] * np.sin(theta), x_bnd[0][1] * np.sin(theta), nx[0], endpoint=True) + start_point[1]

    # Dictionary to map field components to their respective filenames and operations
    field_map = {
        "e1": ("interp/Ex_ext.pkl", "interp/Ey_ext.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex_ext.pkl", "interp/Ey_ext.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez_ext.pkl", None, lambda Ez, _: Ez),
        "b1": ("interp/Bx_ext.pkl", "interp/By_ext.pkl", lambda Bx, By: np.cos(theta) * Bx + np.sin(theta) * By),
        "b2": ("interp/Bx_ext.pkl", "interp/By_ext.pkl", lambda Bx, By: -np.sin(theta) * Bx + np.cos(theta) * By),
        "b3": ("interp/Bz_ext.pkl", None, lambda Bz, _: Bz)
    }

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
def set_uth_al( STATE ):
    '''
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    '''
    # print("calling set_uth_e...")
    if f"vth{ions_1}" not in STATE.keys():
        with open(f'interp/vth{ions_1}.pkl', "rb") as f:
            STATE[f'vth{ions_1}'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE[f'vth{ions_1}']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE[f'vth{ions_1}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{ions_1}']((y_positions, x_positions))
    return
#-----------------------------------------------------------------------------------------

def set_uth_si( STATE ):
    '''
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    '''
    # print("calling set_uth_e...")
    if f"vth{ions_2}" not in STATE.keys():
        with open(f'interp/vth{ions_2}.pkl', "rb") as f:
            STATE[f'vth{ions_2}'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE[f'vth{ions_2}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{ions_2}']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE[f'vth{ions_2}']((y_positions, x_positions))

    return
#-----------------------------------------------------------------------------------------
def set_ufl_e( STATE ):
    # print("calling set_ufl...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))
    # print(f'shape of x array: {STATE["x"].shape}')

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
    # print(f'shape of x array: {STATE["x"].shape}')

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
        density_grid = np.load(f"interp/{ions_1}dens.npy") + np.load(f"interp/{ions_2}dens.npy")
    else:
        density_grid = np.load(filename)

    STATE["nx"] = np.array([4096])
    STATE["xmin"] = np.array([0.0])
    STATE["xmax"] = np.array([7150]) # a little more than the final distance specified in input file

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
def set_density_Al( STATE ):
    """
    Set the aluminum density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting {str.upper(ions_1)} DENSITY...")
    load_and_interpolate_density(STATE, f"interp/{ions_1}dens.npy")

#-----------------------------------------------------------------------------------------
def set_density_Si(STATE):
    """
    Set the silicon density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting {str.upper(ions_2)} DENSITY...")
    load_and_interpolate_density(STATE, f"interp/{ions_2}dens.npy")