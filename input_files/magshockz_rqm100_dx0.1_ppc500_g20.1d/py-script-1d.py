
'''
This template needs parameters:
- dims: int (1 or 2)
- start_point: list of float (length 2)
- theta: float (radians)
- distance: float
- xmin, xmax, ymin, ymax: float
- species_list: list of dict
'''



import numpy as np
import pickle

debug = False

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------


# Define the start point for the ray in OSIRIS units
start_point = [np.float64(0.0), np.float64(294.5465025173213)] # start point in OSIRIS units
theta = 1.5707963267948966 # angle that ray makes with the x axis [radians]


# Parameters of FLASH simulation
box_bounds = {
    "xmin": -3569,
    "xmax": 3569,
    "ymin": -413,
    "ymax": 6725,
}

def set_fld_int( STATE ):
    """
    Function to set the field data in the STATE dictionary based on the field component.
    """
    if debug:
        print("Setting field:", STATE['fld'])
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    if debug:
        print("x_bnd:", x_bnd)

    # Shape of the data array
    nx = STATE["data"].shape
    if debug:
        print("nx:", nx)

    
    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace(x_bnd[0][0] * np.cos(theta), x_bnd[0][1] * np.cos(theta), nx[0], endpoint=True) + start_point[0]
    y = np.linspace(x_bnd[0][0] * np.sin(theta), x_bnd[0][1] * np.sin(theta), nx[0], endpoint=True) + start_point[1]

    # Dictionary to map field components to their respective filenames and operations
    field_map = { 
        "e1": ("interp/Ex.pkl", "interp/Ey.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex.pkl", "interp/Ey.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez.pkl", None, lambda Ez, _: Ez),
        "b1": ("interp/magx.pkl", "interp/magy.pkl", lambda Bx, By: np.cos(theta) * Bx + np.sin(theta) * By),
        "b2": ("interp/magx.pkl", "interp/magy.pkl", lambda Bx, By: -np.sin(theta) * Bx + np.cos(theta) * By),
        "b3": ("interp/magz.pkl", None, lambda Bz, _: Bz)
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
    
    x_bnd = STATE["x_bnd"]

    # Shape of the data array
    nx = STATE["data"].shape

    
    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace(x_bnd[0,0] * np.cos(theta), x_bnd[0,1] * np.cos(theta), nx[0], endpoint=True) + start_point[0]
    y = np.linspace(x_bnd[0,0] * np.sin(theta), x_bnd[0,1] * np.sin(theta), nx[0], endpoint=True) + start_point[1]

    # Dictionary to map field components to their respective filenames and operations
    field_map = { 
        "e1": ("interp/Ex_ext.pkl", "interp/Ey_ext.pkl", lambda Ex, Ey: np.cos(theta) * Ex + np.sin(theta) * Ey),
        "e2": ("interp/Ex_ext.pkl", "interp/Ey_ext.pkl", lambda Ex, Ey: -np.sin(theta) * Ex + np.cos(theta) * Ey),
        "e3": ("interp/Ez_ext.pkl", None, lambda Ez, _: Ez),
        "b1": ("interp/magx.pkl", "interp/magy.pkl", lambda Bx, By: np.cos(theta) * Bx + np.sin(theta) * By),
        "b2": ("interp/magx.pkl", "interp/magy.pkl", lambda Bx, By: -np.sin(theta) * Bx + np.cos(theta) * By),
        "b3": ("interp/magz.pkl", None, lambda Bz, _: Bz)
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
        STATE["data"] = operation(field1((x, y)), field2((x, y)) if field2 else None)
    

    return



#-----------------------------------------------------------------------------------------

def set_uth_e(STATE):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(npart, p_x_dim)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(npart, 3)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """
    npart = STATE["x"].shape[0]
    if debug:
        print("x_dim", STATE["x"].shape[1])
    x_dim = STATE["x"].shape[1]

    if debug:
        print("xmax and xmin", STATE["x"].max(), STATE["x"].min())

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    if "vthele" not in STATE:
        with open('interp/vthele.pkl', "rb") as f:
            STATE['vthele'] = pickle.load(f)


    
    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    chunk_size = 1024
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE['vthele']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE['vthele']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE['vthele']((y_positions, x_positions))
    

    return
#-----------------------------------------------------------------------------------------

def set_uth_al( STATE ):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(npart, p_x_dim)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(npart, 3)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """
    npart = STATE["x"].shape[0]
    x_dim = STATE["x"].shape[1]

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    if "vthal" not in STATE:
        with open('interp/vthal.pkl', "rb") as f:
            STATE['vthal'] = pickle.load(f)

    
    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    chunk_size = 1024
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE['vthal']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE['vthal']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE['vthal']((y_positions, x_positions))
    
    return
#-----------------------------------------------------------------------------------------

def set_uth_si( STATE ):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(npart, p_x_dim)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(npart, 3)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """
    npart = STATE["x"].shape[0]
    x_dim = STATE["x"].shape[1]

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    if "vthsi" not in STATE:
        with open('interp/vthsi.pkl', "rb") as f:
            STATE['vthsi'] = pickle.load(f)

    
    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    chunk_size = 1024
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        x_positions = start_point[0] + np.cos(theta) * STATE["x"][start:end, 0]
        y_positions = start_point[1] + np.sin(theta) * STATE["x"][start:end, 0]

        STATE["u"][start:end, 0] = STATE['vthsi']((y_positions, x_positions))
        STATE["u"][start:end, 1] = STATE['vthsi']((y_positions, x_positions))
        STATE["u"][start:end, 2] = STATE['vthsi']((y_positions, x_positions))
    
    return
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
def set_ufl_e( STATE ):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(npart, p_x_dim)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(npart, 3)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """
    npart = STATE["x"].shape[0]
    x_dim = STATE["x"].shape[1]

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    if "v_ex" not in STATE:
        with open('interp/v_ex.pkl', "rb") as f:
            STATE['v_ex'] = pickle.load(f)

    if "v_ey" not in STATE:
        with open('interp/v_ey.pkl', "rb") as f:
            STATE['v_ey'] = pickle.load(f)

    if "v_ez" not in STATE:
        with open('interp/v_ez.pkl', "rb") as f:
            STATE['v_ez'] = pickle.load(f)

        
    
    # Calculate positions
    x_positions = start_point[0] + np.cos(theta) * STATE["x"][:, 0]
    y_positions = start_point[1] + np.sin(theta) * STATE["x"][:, 0]

    # Set ufl_x1
    STATE["u"][:, 0] = (
        np.cos(theta) * STATE['v_ex']((x_positions, y_positions)) +
        np.sin(theta) * STATE['v_ey']((x_positions, y_positions))
    )

    # Set ufl_x2
    STATE["u"][:, 1] = (
        -np.sin(theta) * STATE['v_ex']((x_positions, y_positions)) +
        np.cos(theta) * STATE['v_ey']((x_positions, y_positions))
    )
    # Set ufl_x3
    STATE["u"][:, 2] = STATE['v_ez']((x_positions, y_positions))
    

    return
#-----------------------------------------------------------------------------------------

def set_ufl_i( STATE ):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(npart, p_x_dim)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(npart, 3)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """
    npart = STATE["x"].shape[0]
    x_dim = STATE["x"].shape[1]

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    if "v_ix" not in STATE:
        with open('interp/v_ix.pkl', "rb") as f:
            STATE['v_ix'] = pickle.load(f)

    if "v_iy" not in STATE:
        with open('interp/v_iy.pkl', "rb") as f:
            STATE['v_iy'] = pickle.load(f)

    if "v_iz" not in STATE:
        with open('interp/v_iz.pkl', "rb") as f:
            STATE['v_iz'] = pickle.load(f)
     
    
    # Calculate positions
    x_positions = start_point[0] + np.cos(theta) * STATE["x"][:, 0]
    y_positions = start_point[1] + np.sin(theta) * STATE["x"][:, 0]

    # Set ufl_x1
    STATE["u"][:, 0] = (
        np.cos(theta) * STATE['v_ix']((x_positions, y_positions)) +
        np.sin(theta) * STATE['v_iy']((x_positions, y_positions))
    )

    # Set ufl_x2
    STATE["u"][:, 1] = (
        -np.sin(theta) * STATE['v_ix']((x_positions, y_positions)) +
        np.cos(theta) * STATE['v_iy']((x_positions, y_positions))
    )

    # Set ufl_x3
    STATE["u"][:, 2] = STATE['v_iz']((x_positions, y_positions))
    

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
    STATE["xmax"] = np.array([2655.918522655891]) # a little more than the final distance specified in input file

    from scipy.interpolate import RegularGridInterpolator
    loaded_interpolator = RegularGridInterpolator((np.linspace(box_bounds["xmin"], box_bounds['xmax'], density_grid.shape[0]), 
                                                   np.linspace(box_bounds['ymin'], box_bounds['ymax'], density_grid.shape[1])), 
                                                   density_grid, bounds_error=True, fill_value=None)

    x = np.linspace(STATE['xmin'][0]*np.cos(theta), STATE['xmax'][0]*np.cos(theta), STATE['nx'][0], endpoint=True ) + start_point[0]
    y = np.linspace(STATE['xmin'][0]*np.sin(theta), STATE['xmax'][0]*np.sin(theta), STATE['nx'][0], endpoint=True ) + start_point[1]
    
    STATE["data"] = loaded_interpolator((x, y)) # This one is reversed because it does not come pre-interpolated

    

    return

#-----------------------------------------------------------------------------------------
def set_density_e( STATE ):
    """
    Set the electron density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    if debug:
        print("setting ELECTRON DENSITY...")
    load_and_interpolate_density(STATE, "interp/edens.npy")

#-----------------------------------------------------------------------------------------


def set_density_al( STATE ):
    """
    Set the al density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    if debug:
        print(f"setting AL DENSITY...")
    load_and_interpolate_density(STATE, f"interp/aldens.npy")

#-----------------------------------------------------------------------------------------


def set_density_si( STATE ):
    """
    Set the si density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    if debug:
        print(f"setting SI DENSITY...")
    load_and_interpolate_density(STATE, f"interp/sidens.npy")

#-----------------------------------------------------------------------------------------
