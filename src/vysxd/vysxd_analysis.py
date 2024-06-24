import numpy as np
import matplotlib.pyplot as plt
import os
from vysxd_define import *
import h5py
import numba

def remove_repeated_elements(my_list):
    a = []
    for elem in my_list:
        if elem not in a:
            a.append(elem)
    return a

def get_osiris_quantity_1d(quant_path, n_start = 0, n_end = -1, n_skip = 1):
    quant_list = np.sort(np.array(os.listdir(quant_path)))
    quant_list = quant_list[np.logical_not(quant_list == '.DS_Store')]
    quant_list = remove_repeated_elements(quant_list)
    quant_list = quant_list[n_start:n_end:n_skip]
    
    n_t = len(quant_list)
    quant = vysxd_get_data(quant_path + quant_list[0])
    n_x = quant.NX
    dx = quant.X[1]-quant.X[0]
    
    Q = np.zeros((n_t, n_x), dtype = np.float32)
    t = np.zeros(n_t)
    Q[0,:] = quant.DATA
    t[0]   = quant.TIME[0]
    for i in range(1, n_t):
        quant = vysxd_get_data(quant_path + quant_list[i])
        Q[i,:] = quant.DATA
        t[i] = quant.TIME[0]
    
    dt = t[1]-t[0]
    
    X = quant.X
    
    return [Q, dt, dx, t, X]

def get_osiris_quantity_2d(quant_path, n_start = 0, n_end = -1, n_skip = 1):
    quant_list = np.sort(np.array(os.listdir(quant_path)))
    quant_list = quant_list[np.logical_not(quant_list == '.DS_Store')]
    quant_list = remove_repeated_elements(quant_list)
    quant_list = quant_list[n_start:n_end:n_skip]
    
    n_t = len(quant_list)
    quant = vysxd_get_data(quant_path + quant_list[0]) # Read hdf5 file format
    n_x = quant.NX
    n_y = quant.NY
    dx = quant.X[1]-quant.X[0]
    dy = quant.Y[1]-quant.Y[0]
    dt = quant.TIME[0]
    
    Q = np.zeros((n_t, n_y, n_x), dtype = np.float32)
    t = np.zeros(n_t)
    Q[0,:,:] = quant.DATA
    t[0]     = quant.TIME[0]
    for i in range(1, n_t):
        quant = vysxd_get_data(quant_path + quant_list[i])
        Q[i,:,:] = quant.DATA
        t[i] = quant.TIME[0]
    
    dt = t[1]-t[0]
    
    X = quant.X
    Y = quant.Y
    
    return [Q, dt, dx, dy, t, X, Y]

def ene_analysis(hist_path, osirisv = 'osiris3'):
  hist_file_list = os.listdir(hist_path)
  particle_hist_files = []
  for hist_file in hist_file_list:
    if ('.DS_Store' in hist_file):
      hist_file_list.remove(hist_file)
    if 'par' in hist_file:
      if 'par_' not in hist_file:
        particle_hist_files.append(hist_file)

  n_species = len(particle_hist_files)
  if osirisv == 'osiris3':
    n_points = len(np.loadtxt(hist_path + particle_hist_files[0], skiprows=1, usecols=[1]))
  elif osirisv == 'osiris4':
    n_points = len(np.loadtxt(hist_path + particle_hist_files[0], skiprows=2, usecols=[1]))


  particle_energy_evol = np.zeros((n_points, n_species)) # [E_par01, E_par02]
  fld_energy_evol = np.zeros((n_points, 6)) # [B1^2, B2^2, B3^2, E1^2, E2^2, E3^2]
  time = np.zeros(n_points) # [B1^2, B2^2, B3^2, E1^2, E2^2, E3^2]

  # load data fld_data
  if osirisv == 'osiris3':
    fld_ene = np.loadtxt(hist_path + 'fld_ene', skiprows=1, usecols=[1,2,3,4,5,6,7])
  if osirisv == 'osiris4':
    fld_ene = np.loadtxt(hist_path + 'fld_ene', skiprows=2, usecols=[1,2,3,4,5,6,7])
  time =  fld_ene[:,0]
  for i in range(6):
    fld_energy_evol[:len(fld_ene[:,i+1]),i] = fld_ene[:,i+1]

  for spec in range(n_species):
    if osirisv == 'osiris3':
      par_ene = np.loadtxt(hist_path + particle_hist_files[spec], skiprows=1, usecols=[1,6])
    elif osirisv == 'osiris4':
      par_ene = np.loadtxt(hist_path + particle_hist_files[spec], skiprows=2, usecols=[1,3])
      
    time =  par_ene[:,0]
    particle_energy_evol[:,spec] = par_ene[:,-1]

          
  total_particle_energy = np.sum(particle_energy_evol,axis = 1)
  total_field_energy = np.sum(fld_energy_evol,axis = 1)


  energy_evol = {'time': time,
                 'Ek': total_particle_energy,
                 'Eemf': total_field_energy,
                 'par_ene': particle_energy_evol, 
                 'B_ene': np.sum(fld_energy_evol[:,0:3],axis = 1),
                 'E_ene': np.sum(fld_energy_evol[:,3:],axis = 1),
                 'B1_ene': fld_energy_evol[:,0],
                 'B2_ene': fld_energy_evol[:,1],
                 'B3_ene': fld_energy_evol[:,2],
                 'E1_ene': fld_energy_evol[:,3],
                 'E2_ene': fld_energy_evol[:,4],
                 'E3_ene': fld_energy_evol[:,5],
                 'ene_conserv': (total_field_energy+total_particle_energy)/(total_field_energy[0]+total_particle_energy[0])-1.
                }

  return energy_evol


# Ryan: I wrote a version using regular python. Then I rewrote it using Numba which is much faster. If you don't want
# to use Numba, just set the last line of getTrackData to return labels[1:], reorderTable(data, itermap, ntracks)

def getTrackData(fname): 
    # Read chunked hdf5 track data and order it correctly.
    # This works because of the 'itermap' member. Every row of itermap contains information about the next
    # several rows of the 'data' attribute. Each row of itermap is (itrack, chunklen, isimulation) where
    # itrack is the particle track id for the next chunklen rows of the 'data' member, and begins at sim step
    # isimulation. The algorithm works by iterating through itermap and pasting each chunklen rows of 'data' into
    # the output slot for that particle. It builds up the output for each particle until returning at the end.
    #
    # If particles are gained/lost during the simulation, they will have unequal lengths and so the remaining
    # space in the output is padded with zeros.
    #
    # Input: Filename of the track data
    # Output: a 2-tuple containing:
    # 1 - the labels of the quants
    # 2 - 3D array of data where the indices are (i_quant, i_particle, i_timestep)
    
    tracks = h5py.File(fname, 'r')
    labels = [s.decode() for s in tracks.attrs['QUANTS']] # Get quant labels as regular (non-binary) strings
    data = tracks['data'][:] # Get the data as a numeric array
    itermap = tracks['itermap'][:] # Get itermap as a numeric array
    ntracks = tracks.attrs['NTRACKS'][0]
    tracks.close()
    
#     return labels[1:], reorderTable(data, itermap, ntracks)
    return labels[1:], reorderTableNumba(data, itermap, ntracks)

def reorderTable(data, itermap, ntracks): # Perform the algorithm described in getOrderedTable
    ntracks_times_nsteps, nquants = data.shape
    nsteps = ntracks_times_nsteps//ntracks
    output = np.zeros((nquants,ntracks,nsteps)) # Allocate output array assuming equal track lengths
    
    ioutput_all = np.zeros((nquants,ntracks),dtype=int) # Keep a cursor for writing each output column
    for iquant in range(nquants): # For each reported variable
        idata = 0 # Which row of the data we are processing
        for itrack, chunklen, isimulation in itermap: # istart is i(t0) for this chunk and particle
            ioutput = ioutput_all[iquant, itrack-1] # Output data cursor location for this quant and particle
            if ioutput+chunklen > output.shape[2]: # If output needs to be bigger, make it bigger
                nq, nt, ns = output.shape
                newOutput = np.zeros((nq,nt,ns*2),dtype=output.dtype) # Make a bigger output array
                newOutput[:,:,:ns] = output # Copy data over into bigger array
                output = newOutput # Switch over to the new output array
                
            output[iquant,itrack-1,ioutput:ioutput+chunklen] = data[idata:idata+chunklen,iquant] # Paste the data
            idata += chunklen # Set cursor past the pasted data in the 'data' attribute
            ioutput_all[iquant, itrack-1] += chunklen # Set cursor to the next open row in the output
    return output

@numba.njit() # This is a faster version of reorderTable using Numba
def reorderTableNumba(data, itermap, ntracks):
    ntracks_times_nsteps, nquants = data.shape
    nsteps = ntracks_times_nsteps//ntracks
    ntracks = np.uint64(ntracks) # The hdf5 file gives a 32-bit int but Numba wants 64-bit
    
    output = np.zeros((nquants,ntracks,nsteps), dtype=data.dtype) # Allocate assuming equal track lengths
    ioutput_all = np.zeros((nquants,ntracks),dtype=np.int64) # Keep a cursor for writing each output column
    for iquant in range(nquants): # For each reported variable
        idata = 0 # Which row of the data we are processing
        for i_itermap in range(itermap.shape[0]): # istart is i(t0) for this chunk and particle
            itrack, chunklen, isimulation = itermap[i_itermap,:] # Next row of the itermap
            ioutput = ioutput_all[iquant, itrack-1] # Output data cursor location for this quant and particle
            if ioutput+chunklen > output.shape[2]: # If output needs to be bigger, make it bigger
                nq, nt, ns = output.shape
                newOutput = np.zeros((nq,nt,ns*2),dtype=output.dtype) # Make a bigger output array
                newOutput[:,:,:ns] = output # Copy data over into bigger array
                output = newOutput # Switch over to the new output array
            output[iquant,itrack-1,ioutput:ioutput+chunklen] = data[idata:idata+chunklen,iquant] # Paste the data
            idata += chunklen # Set cursor past the pasted data in the 'data' attribute
            ioutput_all[iquant, itrack-1] += chunklen # Set cursor to the next open row in the output
    return output



def energy_spectrum_evolution(spectrum_path, tnorm, plot_title='', gamma_max_range = 600, log_x_scale = True, tmax = None):

    base_dir = spectrum_path
    list_of_files = os.listdir(base_dir)
    if tmax == None:
        spectrum = vysxd_get_data(base_dir + list_of_files[-1])
        tmax = spectrum.TIME[0]
        time_range = tmax
    else:
        time_range = tmax
    spectrum = vysxd_get_data(base_dir + list_of_files[0])
    time_range -= spectrum.TIME[0]
    
    min, max = (0, int(round(time_range/tnorm))+1)
    step = 1

    # Setting up a colormap that's a simple transtion
    # mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',[(0,0,1),(0,1,1),(1,1,0),(1,0,0)])
    brightness = 0.9
    mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',[(0,0,brightness),(0,brightness,brightness),
                                                                     (brightness,brightness,0),(brightness,0,0)])

    def rgb_code(s):
        if s<1/3:
            # blue to cyan
            r = 0.
            g = s / (1/3)
            b = 1.
        elif s<2/3:
            # cyan to yellow
            r = (s-1/3)/(1/3)
            g = 1.
            b = (1. - (s-1/3)/(1/3))
        else:
            # yellow to red
            r = 1.
            g = 1.-(s-2/3)/(1/3)
            b = 0.

        return np.array([r, g, b])

    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    levels = range(min,max+step,step)
    CS3 = plt.contourf(Z, levels, cmap=mymap)
    plt.clf()


    fig, ax1 = plt.subplots(figsize=(4.5,3))
    plt.title(plot_title)

    # Plotting what I actually want
    for file in list_of_files:
        spectrum = vysxd_get_data(base_dir + file)
        t = spectrum.TIME[0]/tnorm
        if t > tmax/tnorm: break
        spectrum.DATA /= np.trapz(spectrum.DATA, spectrum.X)
        r, g, b = rgb_code(float(t)/max)*brightness
        plt.loglog(spectrum.X-1, spectrum.DATA, lw = 1., color = (r,g,b))
        
    if log_x_scale == False:
        plt.xscale('linear')
    plt.colorbar(CS3, label = r'$ct/R$') # using the colorbar info I got from contourf
    plt.legend(frameon=False)
    plt.tight_layout()