'''
This template needs parameters:
- dims: int (1 or 2)
- start_point: list of float (length 2)
- theta: float (radians)
- distance: float
- xmin, xmax, ymin, ymax: float
- ion_species: list of str
- species_list: list of dict
'''

import numpy as np
import pickle

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------



# Parameters of FLASH simulation
box_bounds = {
    "xmin": -2003,
    "xmax": 2003,
    "ymin": 197,
    "ymax": 3003,
}

def set_fld_int( STATE ):
    """
    Function to set the field data in the STATE dictionary based on the field component.
    """
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]

    # Shape of the data array
    nx = STATE["data"].shape

    
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
    # x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    # x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    # X1, X2 = np.meshgrid( x1, x2, indexing='xy' ) # Matches Fortran array indexing

    # # Perform some function to fill in the field values based on the coordinates
    # with open(filename, "rb") as f:
    #     loaded_interpolator = pickle.load(f)

    # STATE["data"] = loaded_interpolator((X2, X1))
    

    return



#-----------------------------------------------------------------------------------------

def set_uth_e(STATE):
    """
    The `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`. This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles. **This quantity should be set to the desired momentum data.**
    """

    if "vthele" not in STATE:
        with open('interp/vthele.pkl', "rb") as f:
            STATE['vthele'] = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    # Define a chunk size for processing
    chunk_size = 1024

    # Assign velocities in chunks, this saves memory in 2D. In 1D the difference is negligible
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        
        STATE["u"][start:end, 0] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 1] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        STATE["u"][start:end, 2] = STATE['vthele']((STATE["x"][start:end, 1], STATE["x"][start:end, 0]))
        

    return
#-----------------------------------------------------------------------------------------


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
        
    
    chunk_size = 1024  # Define a chunk size

    # Set ufl_x1
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end,0] = velx((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    # Set ufl_x2
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end,1] = vely((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    # Set ufl_x3
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end,2] = velz((STATE["x"][start:end,1], STATE["x"][start:end,0]))
    

    return
#-----------------------------------------------------------------------------------------

def set_ufl_i( STATE ):
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    with open("interp/v_ix.pkl", "rb") as f:
        velx = pickle.load(f)
    with open("interp/v_iy.pkl", "rb") as f:
        vely = pickle.load(f)
    with open("interp/v_iz.pkl", "rb") as f:
        velz = pickle.load(f)
     
    
    chunk_size = 1024  # Define a chunk size

    # Set ufl_x1
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end,0] = velx((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    # Set ufl_x2
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end,1] = vely((STATE["x"][start:end,1], STATE["x"][start:end,0]))

    # Set ufl_x3
    for start in range(0, len(STATE["u"][:,0]), chunk_size):
        end = min(start + chunk_size, len(STATE["u"][:,0]))
        STATE["u"][start:end,2] = velz((STATE["x"][start:end,1], STATE["x"][start:end,0]))
    

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

    
    STATE["nx"] = np.array(density_grid.shape)//2
    STATE["xmin"] = np.array([-2002, 198]) # go a little beyond the bounds sepcified in the input file
    STATE["xmax"] = np.array([2002, 3002])
    STATE['data'] = density_grid[::2,::2].T # We have to down sample here or else it breaks
    

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


def set_density_al( STATE ):
    """
    Set the al density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting AL DENSITY...")
    load_and_interpolate_density(STATE, f"interp/aldens.npy")

#-----------------------------------------------------------------------------------------


def set_density_si( STATE ):
    """
    Set the si density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    print(f"setting SI DENSITY...")
    load_and_interpolate_density(STATE, f"interp/sidens.npy")

#-----------------------------------------------------------------------------------------
