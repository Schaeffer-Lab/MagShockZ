import numpy as np

def find_shock_front( data ):
    '''
    data: recommend that you input a 2D numpy array with the first column being time and the second column being magnitude of B field.
    '''
    grad = np.gradient(data[:,1], data[:,0])