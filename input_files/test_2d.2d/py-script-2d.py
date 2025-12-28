

'''
This template needs parameters:
- dims: int (1 or 2)
- box_bounds: dict
- xmin, xmax, ymin, ymax: float
- xmin, xmax, ymin, ymax: float
- species_list: list of dict
'''


import numpy as np
import pickle

debug = False

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------



# Parameters of FLASH simulation
box_bounds = {
    "xmin": -2520,
    "xmax": 2520,
    "ymin": -311,
    "ymax": 4203,
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
    x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' )
    if debug:
        print("X1 shape:", X1.shape)
        print("X2 shape:", X2.shape)

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

    STATE["data"] = loaded_interpolator((X1, X2))
    

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

    # STATE["data"] = loaded_interpolator((X1, X2))
    

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


    
    STATE["u"][:,0] = STATE['vthele']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:,1] = STATE['vthele']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:,2] = STATE['vthele']((STATE["x"][:,0], STATE["x"][:,1]))
    

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

    
    STATE["u"][:, 0] = STATE['vthal']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 1] = STATE['vthal']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 2] = STATE['vthal']((STATE["x"][:,0], STATE["x"][:,1]))
    
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

    
    STATE["u"][:, 0] = STATE['vthsi']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 1] = STATE['vthsi']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 2] = STATE['vthsi']((STATE["x"][:,0], STATE["x"][:,1]))
    
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

        
    
    STATE["u"][:, 0] = STATE['v_ex']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 1] = STATE['v_ey']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 2] = STATE['v_ez']((STATE["x"][:,0], STATE["x"][:,1]))
    

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
     
    
    STATE["u"][:, 0] = STATE['v_ix']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 1] = STATE['v_iy']((STATE["x"][:,0], STATE["x"][:,1]))
    STATE["u"][:, 2] = STATE['v_iz']((STATE["x"][:,0], STATE["x"][:,1]))
    

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

    
    # Downsample by factor of 2
    downsampled = density_grid.T 
    
    # You might need this I'm not sure
    # downsampled = np.maximum(downsampled, 0.0)
    
    STATE['data'] = downsampled
    
    STATE["nx"] = np.array([STATE['data'].shape[1], STATE['data'].shape[0]]) # This is correct holy shit
    STATE["xmin"] = np.array([box_bounds["xmin"], box_bounds["ymin"]])
    STATE["xmax"] = np.array([box_bounds["xmax"], box_bounds["ymax"]])
    

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
