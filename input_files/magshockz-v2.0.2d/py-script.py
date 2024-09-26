# Outside of the function definition, any included code only runs once at the
# initialization. You could, e.g., load an ML model from a file here
# print("I am at the module scope.")
import numpy as np
import pickle
#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------
def set_fld( STATE ):
    """
    Function to set the field data in the STATE dictionary based on the field component.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including field component and positional boundary data.
    """
    # print("calling set_fld...")
    
    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]
    # print(f"x_bnd = {x_bnd}")

    # Shape of the data array
    nx = STATE["data"].shape
    # print(f"nx = {nx}")

    # Create x arrays that indicate the position (remember indexing order is reversed)
    x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' )

    # Determine the filename based on the field component
    match STATE['fld']:
        case "e1":
            filename = "interp/Ex-interp.pkl"
        case "e2":
            filename = "interp/Ey-interp.pkl"
        case "e3":
            filename = "interp/Ez-interp.pkl"
        case "b1":
            filename = "interp/magx-interp.pkl"
        case "b2":
            filename = "interp/magy-interp.pkl"
        case "b3":
            filename = "interp/magz-interp.pkl"

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
    #         filename = "interp/Ex-interp.pkl"
    #     case "e2":
    #         filename = "interp/Ey-interp.pkl"
    #     case "e3":
    #         filename = "interp/Ez-interp.pkl"
    #     case "b1":
    #         filename = "interp/magx-interp.pkl"
    #     case "b2":
    #         filename = "interp/magy-interp.pkl"
    #     case "b3":
    #         filename = "interp/magz-interp.pkl"

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
    '''
    In each of the above cases, the `STATE` dictionary will be prepared with the following key:
    "x" - A real array of size `(p_x_dim, npart)` containing the positions of the particles.

    The desired momentum array can then be created and set based on the positions `"x"`.  This array should be passed to the `STATE` array with the following key:
    "u" - A real array of size `(3, npart)` containing either the thermal or fluid momenta of the particles.  **This quantity should be set to the desired momentum data.**
    '''
    # print("calling set_uth_e...")
    with open('interp/vthele-interp.pkl', "rb") as f:
        loaded_interpolator = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    # Set uth_x1
    STATE["u"][:,0] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    # # Set uth_x2
    STATE["u"][:,1] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    # # Set uth_x3
    STATE["u"][:,2] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    return STATE['u']
#-----------------------------------------------------------------------------------------

def set_uth_i( STATE ):
    # print("calling set_uth_i...")
    with open('interp/vthion-interp.pkl', "rb") as f:
        loaded_interpolator = pickle.load(f)

    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    # Set uth_x1
    STATE["u"][:,0] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    # # Set uth_x2
    STATE["u"][:,1] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    # # Set uth_x3
    STATE["u"][:,2] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    return STATE['u']
#-----------------------------------------------------------------------------------------
def set_ufl( STATE ):
    # print("calling set_ufl...")
    # Prepare velocity array
    STATE["u"] = np.zeros((STATE["x"].shape[0], 3))

    # Set uth_x1
    with open("interp/velx-interp.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
    STATE["u"][:,0] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    # # Set uth_x2
    with open("interp/vely-interp.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
    STATE["u"][:,1] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

    # # Set uth_x3
    with open("interp/velz-interp.pkl", "rb") as f:
        loaded_interpolator = pickle.load(f)
    STATE["u"][:,2] = loaded_interpolator((STATE["x"][:,1], STATE["x"][:,0]))

#-----------------------------------------------------------------------------------------
def load_and_interpolate_density(STATE, filename):
    """
    Helper function to load interpolator from a file and set the density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information, including positional boundary data.
    filename (str): Path to the file containing the interpolator.
    """
    # Number of points, xmin, and xmax in x and y, respectively
    STATE["nx"] = np.array(STATE["data"].shape)
    STATE["xmin"] = STATE['x_bnd'][:,0]
    STATE["xmax"] = STATE['x_bnd'][:,1]

    x1 = np.linspace(STATE["xmin"][0], STATE["xmax"][0], STATE["nx"][0], endpoint=True)
    x2 = np.linspace(STATE["xmin"][1], STATE["xmax"][1], STATE["nx"][1], endpoint=True)
    X1, X2 = np.meshgrid(x1, x2, indexing='xy')  # Matches Fortran array indexing

    with open(filename, "rb") as f:
        loaded_interpolator = pickle.load(f)

    # Perform some function to fill in the field values based on the coordinates
    return loaded_interpolator((X2, X1))

#-----------------------------------------------------------------------------------------
def set_density_e( STATE ):
    """
    Set the electron density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    # print("calling set_density_e...")
    STATE['data'] = load_and_interpolate_density(STATE, "interp/edens-interp.pkl")

#-----------------------------------------------------------------------------------------
def set_density_Al( STATE ):
    """
    Set the aluminum density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    # print("calling set_density_Al...")
    STATE['data'] = load_and_interpolate_density(STATE, "interp/aldens-interp.pkl")

#-----------------------------------------------------------------------------------------
def set_density_Mg(STATE):
    """
    Set the magnesium density data in the STATE dictionary.
    
    Parameters:
    STATE (dict): Dictionary containing the state information.
    """
    # print("calling set_density_Mg...")
    STATE['data'] = load_and_interpolate_density(STATE, "interp/mgdens-interp.pkl")