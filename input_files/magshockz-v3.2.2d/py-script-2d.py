
import numpy as np
# Outside of the function definition, any included code only runs once at the
# initialization. You could, e.g., load an ML model from a file here
import numpy as np
import pickle


# Define bounds of box in osiris units, ensure that this is larger than the bounds specified in input file
box_bounds = {
    "xmin": -3357.479151559726, 
    "xmax": 3357.4791515597244,
    "ymin": -306.7838957759923,
    "ymax": 8406.81192721512,
}

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
    # print(f"x_bnd = { x_bnd }")

    # Shape of the data array
    nx = STATE["data"].shape
    # print(f"nx = { nx }")

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

#-----------------------------------------------------------------------------------------
def set_uth_channel( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """

    if "vthchannel" not in STATE.keys():
        with open(f'interp/vthchannel.pkl', "rb") as f:
            STATE[f'vthchannel'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE[f'vthchannel']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE[f'vthchannel']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE[f'vthchannel']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
    return
#-----------------------------------------------------------------------------------------

def set_uth_sheathe( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """

    if "vthsheathe" not in STATE.keys():
        with open(f'interp/vthsheathe.pkl', "rb") as f:
            STATE[f'vthsheathe'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE[f'vthsheathe']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE[f'vthsheathe']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE[f'vthsheathe']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
    return
#-----------------------------------------------------------------------------------------

def set_uth_background( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """

    if "vthbackground" not in STATE.keys():
        with open(f'interp/vthbackground.pkl', "rb") as f:
            STATE[f'vthbackground'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE[f'vthbackground']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE[f'vthbackground']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE[f'vthbackground']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
    return
#-----------------------------------------------------------------------------------------

def set_uth_si( STATE ):
    """ 
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    """

    if "vthsi" not in STATE.keys():
        with open(f'interp/vthsi.pkl', "rb") as f:
            STATE[f'vthsi'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE[f'vthsi']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE[f'vthsi']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE[f'vthsi']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
    return
#-----------------------------------------------------------------------------------------


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
    STATE["xmin"] = np.array([-3357.479151559726, -306.7838957759923])
    STATE["xmax"] = np.array([3357.4791515597244,8406.81192721512])
    STATE['data'] = density_grid[::2,::2].T

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

def set_density_channel( STATE ):
    """
    Set the channel density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting CHANNEL DENSITY...")
    load_and_interpolate_density(STATE, f"interp/channeldens.npy")

#-----------------------------------------------------------------------------------------'                   



def set_density_sheathe( STATE ):
    """
    Set the sheathe density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting SHEATHE DENSITY...")
    load_and_interpolate_density(STATE, f"interp/sheathedens.npy")

#-----------------------------------------------------------------------------------------'                   



def set_density_background( STATE ):
    """
    Set the background density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting BACKGROUND DENSITY...")
    load_and_interpolate_density(STATE, f"interp/backgrounddens.npy")

#-----------------------------------------------------------------------------------------'                   



def set_density_si( STATE ):
    """
    Set the si density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting SI DENSITY...")
    load_and_interpolate_density(STATE, f"interp/sidens.npy")

#-----------------------------------------------------------------------------------------'                   
