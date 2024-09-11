# Outside of the function definition, any included code only runs once at the
# initialization. You could, e.g., load an ML model from a file here
# print("I am at the module scope.")
import numpy as np
from scipy import interpolate

if __name__ == '__main__':
    # Compute some interpolation function and save it to disk
    # This can be done just once by one processor and before the OSIRIS run
    # E.g., "python py-script.py"

    # Make position arrays
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 200)

    # Compute complicated function on a grid
    # Here we keep the same indexing as the Fortran-communicated arrays, for no real reason
    val1 = 0.2*x[np.newaxis, :] + y[:, np.newaxis]
    val2 = x[np.newaxis, :] - y[:, np.newaxis]

    # Make an interpolation object for that function (here we set the value to 0 out of bounds)
    interp1 = interpolate.RegularGridInterpolator( (y, x), val1, method='cubic', bounds_error=False, fill_value=0 )
    interp2 = interpolate.RegularGridInterpolator( (y, x), val2, method='cubic', bounds_error=False, fill_value=0 )
    np.save( "interp", np.array([interp1, interp2]) )

#-----------------------------------------------------------------------------------------
# Functions callable by OSIRIS
#-----------------------------------------------------------------------------------------
def add_one( STATE ):
    a = STATE["VAR_NAME"]
    a = a + 1.0
    STATE["VAR_NAME"] = a


#-----------------------------------------------------------------------------------------
def set_fld( STATE ):
    # Field data is in STATE["data"]
    # The below statement would make a copy
    # data = STATE["data"]

    # Name of the field component
    fld = STATE["fld"]

    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]

    # Time (in case fields are dynamic)
    t = STATE["t"][0]

    # Could make decisions based on field component
    # if fld=="e1":
    #     # Do something here
    # elif fld=="b2":
    #     # Do something else

    # Create x arrays that indicate the position (remember indexing order is reversed)
    nx = STATE["data"].shape
    x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' ) # Matches Fortran array indexing

    # Perform some function to fill in the field values based on the coordinates
    STATE["data"] = 2*(X1-1) - (X2-2)

    # Could also do this with iteration
    # for i in range(nx[1]):
    #     for j in range(nx[0]):
    #         STATE["data"][j,i] = 2*(x1[i]-1) - (x2[j]-2)


#-----------------------------------------------------------------------------------------
def set_fld_ext( STATE ):
    # Field data is in STATE["data"]
    # The below statement would make a copy
    # data = STATE["data"]

    # Name of the field component
    fld = STATE["fld"]

    # Positional boundary data (makes a copy, but it's small)
    x_bnd = STATE["x_bnd"]

    # Time (in case fields are dynamic)
    t = STATE["t"][0]

    # Could make decisions based on field component
    # if fld=="e1":
    #     # Do something here
    # elif fld=="b2":
    #     # Do something else

    # Create x arrays that indicate the position (remember indexing order is reversed)
    nx = STATE["data"].shape
    x1 = np.linspace( x_bnd[0,0], x_bnd[0,1], nx[1], endpoint=True )
    x2 = np.linspace( x_bnd[1,0], x_bnd[1,1], nx[0], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' ) # Matches Fortran array indexing

    # Perform some function to fill in the field values based on the coordinates
    STATE["data"] = -0.2*(X1-2) + (X2-1) + t

    # Could also do this with iteration
    # for i in range(nx[1]):
    #     for j in range(nx[0]):
    #         STATE["data"][j,i] = -0.2*(x1[i]-2) + (x2[j]-1) + t


#-----------------------------------------------------------------------------------------
def set_uth( STATE ):
    # Particle positions are in STATE["x"]
    # Thermal velocities go in STATE["u"]
    # The below statement would make a copy
    # x = STATE["x"]

    npart = STATE["x"].shape[0]
    x_dim = STATE["x"].shape[1]

    # Load interpolation variable saved earlier
    interp1, interp2 = np.load("interp.npy", allow_pickle=True)

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    # Set uth_x1
    # STATE["u"][:,0] = np.abs( 0.2*STATE["x"][:,0] + 1.0*STATE["x"][:,1] )
    STATE["u"][:,0] = interp1((STATE["x"][:,1], STATE["x"][:,0]))
    # Set uth_x2
    # STATE["u"][:,1] = np.abs( 1.0*STATE["x"][:,0] - 1.0*STATE["x"][:,1] )
    STATE["u"][:,1] = interp2((STATE["x"][:,1], STATE["x"][:,0]))
    # Set uth_x3
    STATE["u"][:,2] = 0.1


#-----------------------------------------------------------------------------------------
def set_ufl( STATE ):
    # Particle positions are in STATE["x"]
    # Fluid velocities go in STATE["u"]
    # The below statement would make a copy
    # x = STATE["x"]

    npart = STATE["x"].shape[0]
    x_dim = STATE["x"].shape[1]

    # Prepare velocity array
    STATE["u"] = np.zeros((npart, 3))

    # Set ufl_x1
    STATE["u"][:,0] = -np.abs( 0.2*STATE["x"][:,0] + 1.0*STATE["x"][:,1] )
    # Set ufl_x2
    STATE["u"][:,1] = -np.abs( 1.0*STATE["x"][:,0] - 1.0*STATE["x"][:,1] )
    # Set ufl_x3
    STATE["u"][:,2] = -0.1


#-----------------------------------------------------------------------------------------
def set_density( STATE ):
    # Number of points, xmin, and xmax in x and y, respectively
    STATE["nx"] = np.array([20, 15])
    STATE["xmin"] = np.array([0.25, 0.5])
    STATE["xmax"] = np.array([2.75, 2.5])

    x1 = np.linspace( STATE["xmin"][0], STATE["xmax"][0], STATE["nx"][0], endpoint=True )
    x2 = np.linspace( STATE["xmin"][1], STATE["xmax"][1], STATE["nx"][1], endpoint=True )
    X1, X2 = np.meshgrid( x1, x2, indexing='xy' ) # Matches Fortran array indexing
    STATE["data"] = 0.5 * np.sin(X1 * X2 + 2 * X1) + 1.0
