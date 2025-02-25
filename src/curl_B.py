import numpy as np

def curl_B(Bx,By,Bz, dx=1.0):
    """
    Calculate the curl of a 3D magnetic field.
    
    Parameters:
    Bx : numpy array of shape (nx, ny, nz) representing x-component
    By : numpy array of shape (nx, ny, nz) representing y-component
    Bz : numpy array of shape (nx, ny, nz) representing z-component

    dx : float, grid spacing in x direction
    dy : float, grid spacing in y direction
    dz : float, grid spacing in z direction
    
    Returns:
    curl : numpy array of shape (3, nx, ny, nz) representing (curl_x, curl_y, curl_z)
    """
    
    # Get gradients using numpy's gradient function with specified spacing
    Bx_grad = np.gradient(Bx, dx)
    By_grad = np.gradient(By, dx)
    Bz_grad = np.gradient(Bz, dx)
    # Extract individual components
    
    c = 2.99792458e10 # Speed of light in cgs units

    # Calculate curl components
    J_x = c/(4*np.pi)*(Bz_grad[1] - By_grad[2])
    J_y = c/(4*np.pi)*(Bx_grad[2] - Bz_grad[0])
    J_z = c/(4*np.pi)*(By_grad[0] - Bx_grad[1])
    
    # Stack the components into a single array
    
    return [J_x, J_y, J_z]

def add_vi_and_ve(data, J):
    """
    Add ion and electron velocities to the dataset.

    Parameters:
    data : YtCoveringGrid
    J : list of numpy arrays of shape (nx, ny, nz) representing current density
    """
    
    e = 4.80320425e-10 # Elementary charge in statcoulombs
    m_e = 9.10938356e-28 # Electron mass in grams

    data['v_iy'] = m_e*J[1].value/(data['dens'].value*e) + data['vely'].value
    data['v_iz'] = m_e*J[2].value/(data['dens'].value*e) + data['velz'].value
    data['v_ix'] = m_e*J[0].value/(data['dens'].value*e) + data['velx'].value

    data['v_ex'] = m_e*J[0].value/(data['dens'].value*e) - J[0].value/(data['edens'].value*e) + data['velx'].value
    data['v_ey'] = m_e*J[1].value/(data['dens'].value*e) - J[1].value/(data['edens'].value*e) + data['vely'].value
    data['v_ez'] = m_e*J[2].value/(data['dens'].value*e) - J[2].value/(data['edens'].value*e) + data['velz'].value