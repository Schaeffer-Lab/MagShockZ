# Outside of the function definition, any included code only runs once at the
# initialization. You could, e.g., load an ML model from a file here
# print("I am at the module scope.")
import numpy as np
import pickle

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------
import numpy as np
start_pt = [0, 0]
end_pt = [0, 0]

def set_fld( STATE ):
    """
    Function to set the field data in the STATE dictionary based on the field component.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including field component and positional boundary data.
    """
    # print("calling set_fld...")
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    print(f"x_bnd = {x_bnd}")

    # Shape of the data array
    nx = STATE["data"].shape
    print(f"nx = {nx}")

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x = np.linspace( x_bnd[0], x_bnd[1], nx[0], endpoint=True ).astype(np.float32)

    # Determine the filename based on the field component
    match STATE['fld']:
        case "e1":
            filename = "interp/Ex.pkl"
        case "e2":
            filename = "interp/Ey.pkl"
        case "e3":
            filename = "interp/Ez.pkl"
        case "b1":
            filename = "interp/magx.pkl"
        case "b2":
            filename = "interp/magy.pkl"
        case "b3":
            filename = "interp/magz.pkl"

    with open(filename, "rb") as f:
        loaded_interpolator = pickle.load(f)

    STATE["data"] = loaded_interpolator((r*np.cos(theta), r*np.sin(theta))).astype(np.float32)


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
    r = np.linspace( x_bnd[0], x_bnd[1], nx[0], endpoint=True ).astype(np.float32)
    # X1, X2 = np.meshgrid( x1, x2, indexing='xy' ) # Matches Fortran array indexing

    # # Perform some function to fill in the field values based on the coordinates
    # with open(filename, "rb") as f:
    #     loaded_interpolator = pickle.load(f)

    # STATE["data"] = loaded_interpolator((r*np.cos(theta), r)).astype(np.float32)



#-----------------------------------------------------------------------------------------

def set_uth_e( STATE ):
    '''
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    '''
    # print("calling set_uth_e...")
    if "vthele" not in STATE.keys():
        with open('interp/vthele.pkl', "rb") as f:
            STATE['vthele'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)
        STATE["u"][start:end, 1] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)
        STATE["u"][start:end, 2] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)

    return
#-----------------------------------------------------------------------------------------

def set_uth_al( STATE ):
    # print("calling set_uth_i...")

    if "vthal" not in STATE.keys():
        with open('interp/vthal.pkl', "rb") as f:
            STATE['vthal'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE['vthal']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)
        STATE["u"][start:end, 1] = STATE['vthal']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)
        STATE["u"][start:end, 2] = STATE['vthal']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)

    return

#-----------------------------------------------------------------------------------------

def set_uth_si( STATE ):
    # print("calling set_uth_i...")

    if "vthsi" not in STATE.keys():
        with open('interp/vthsi.pkl', "rb") as f:
            STATE['vthsi'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end, 0] = STATE['vthsi']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)
        STATE["u"][start:end, 1] = STATE['vthsi']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)
        STATE["u"][start:end, 2] = STATE['vthsi']((STATE["x"][start:end, 1], STATE["x"][start:end, 0])).astype(np.float32)

    return
#-----------------------------------------------------------------------------------------
def set_ufl( STATE ):
    # print("calling set_ufl...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    chunk_size = 1024  # Define a chunk size

        # Set ufl_x1
    with open("interp/velx.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,0] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0])).astype(np.float32)

    # Set ufl_x2
    with open("interp/vely.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,1] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0])).astype(np.float32)

        # Set ufl_x3
    with open("interp/velz.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
        for start in range(0, len(STATE["u"][:,0]), chunk_size):
            end = min(start + chunk_size, len(STATE["u"][:,0]))
            STATE["u"][start:end,2] = loaded_interpolator((STATE["x"][start:end,1], STATE["x"][start:end,0])).astype(np.float32)

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
    # print(STATE.keys())
    if "fld" in STATE.keys():
        del STATE["fld"]
    if filename == "interp/edens.npy":
        density_grid = np.load("interp/aldens.npy") + np.load("interp/sidens.npy")
    else:
        density_grid = np.load(filename)
    STATE["nx"] = np.array(density_grid.shape)//2
    STATE["xmin"] = np.array([-3362.0, 1889.0])
    STATE["xmax"] = np.array([3362.0, 8411.0])
    STATE['data'] = density_grid[::2,::2].T.astype(np.float32)  

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
    print("setting ALUMINUM DENSITY...")
    load_and_interpolate_density(STATE, "interp/aldens.npy")

#-----------------------------------------------------------------------------------------
def set_density_Si(STATE):
    """
    Set the magnesium density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print("setting MAGNESIUM DENSITY...")
    load_and_interpolate_density(STATE, "interp/sidens.npy")