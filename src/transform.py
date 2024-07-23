import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy

def box_integrate(q: list, xmin, xmax, tmin, tmax, v, plot_flag = True, q_0 = None) -> np.ndarray:
    '''
    Take time averages over lines in the frame stationary to the shock

    Keyword arguments:
    q -- Quantity that will be averaged. The form of which comes from get_osiris_quantity_1d, ie. [Q,dt,dx,t,x]
    xmin -- lower bound of x integration on left side of parallelogram
    xmax -- upper bound of x integration on left side of parallelogram
    tmin, tmax -- bounds of time integration
    v -- speed of shock
    plot_flag -- flag for whether function should automatically plot results
    q_0 -- example timeshot of type vysxd.data_object, used to label axes
    '''

    # Pull out axes from quantity object passed in to function
    t_axis = q[3]
    x_axis = q[4]

    # It is necessary to interpolate your quantity in order to integrate along specified path without too much headache
    q_f = scipy.interpolate.RegularGridInterpolator([t_axis,x_axis], q[0])

    dx = q[2]
    xmin_index = int(xmin/dx)
    xmax_index = int(xmax/dx)

    # Perform line integrations for successive x values in desired box
    def line_integrate(q_f: scipy.interpolate.RegularGridInterpolator,x0) -> float:
        '''
        Goal: Average some quantity in shock stationary frame
        Method: Integrate along a line that starts at inital position in phase space (t0,x0) with slope v

        Keyword Arguments
        q_f -- interpolated function of desired quantity
        '''
        soln = np.mean([q_f([ti,v*(ti-tmin)+x0]) for ti in np.linspace(tmin,tmax,128)]) # resolution is 128, might adjust later

        return soln
    
    solution = np.array([line_integrate(q_f,x) for x in x_axis[xmin_index:xmax_index]])

    # Immediately show a plot to verify, if desired
    if q_0 != None:
        plt.plot(x_axis[xmin_index:xmax_index],solution)
        plt.xlabel(f'{q_0.AXIS1_NAME} [${q_0.AXIS1_UNITS}$]')
        plt.ylabel(f'integrated {q_0.DATA_NAME} density')
        plt.title(f'integrated along lines of v = {v}')

    return np.array(solution)

def illustrate_box(q: list, xmin, xmax, tmin, tmax, v, q_0 = None) -> None:
    '''
    Overlay the paralellogram of integration (all hail) over a colorplot of the quantity of interest

    Keyword arguments:
    q -- Quantity that will be averaged. The form of which comes from get_osiris_quantity_1d, ie. [Q,dt,dx,t,x]
    xmin -- lower bound of x integration on left side of parallelogram
    xmax -- upper bound of x integration on left side of parallelogram
    tmin, tmax -- bounds of time integration
    v -- speed of shock
    '''


    fig, ax = plt.subplots()
    # Make a heatmap of quantity in (t,x) space (sorry Paulo)
    plt.imshow(np.transpose(q[0]), origin='lower', extent=[q[3][0], q[3][-1], q[4][0], q[4][-1]], aspect='auto')

    # If vysxd.data_object timeshot is supplied, use this to label axes
    if (q_0 != None):
        plt.ylabel(f"{q_0.AXIS1_NAME} [${q_0.AXIS1_UNITS}$]")
        plt.xlabel(f"Time [${q_0.TIME_UNITS}$]")
        plt.colorbar(label=q_0.DATA_NAME)

    # Draw a parallelogram to illustrate entire region that is being integrated
    vertices = [[tmin,xmin],[tmin,xmax],[tmax,xmax+v*(tmax-tmin)],[tmax,xmin+v*(tmax-tmin)]]
    # This is very disorganized but it works
    box = patches.Polygon(vertices, ls = '--', fill =False, color = 'red')
    ax.add_patch(box)

    # Draw arrows to illustrate direction of integration
    arrow = patches.Arrow(tmin,xmax,tmax-tmin,v*(tmax-tmin), color = 'red',width=1, ls = '--')
    ax.add_patch(arrow)
    arrow = patches.Arrow(tmin,xmin,tmax-tmin,v*(tmax-tmin), color = 'red',width=1, ls = '--')
    ax.add_patch(arrow)
    arrow = patches.Arrow(tmin,(xmax+xmin)/2,tmax-tmin,v*(tmax-tmin), color = 'red',width=1, ls = '--')
    ax.add_patch(arrow)
    
    # Need to expand limits of ax or drawings will be cut off
    ax.set_ylim(q[4][0],q[4][-1])
    ax.set_xlim(q[3][0],q[3][-1])
    plt.show()

def field_transform(v: float, e_: list, b_: list) -> list:
    '''
    Do a lorentz transform on the fields, what it says on the tin.

    Keyword arguments:
    v -- shock speed
    e_ -- list of the form [e1,e2,e3]
    b_ -- list of the form [b1,b2,b3]
    '''

    gamma = 1/np.sqrt(1-v**2)

    e1 = e_[0]
    b1 = b_[0]

    e2 = gamma*(e_[1] - v*b_[2])
    b2 = gamma*(b_[1] + v*e_[2])

    e3 = gamma*(e_[2] + v*b_[1])
    b3 = gamma*(b_[2] - v*e_[1])
    return [e1,e2,e3], [b1,b2,b3]

def plot_quantity(q: np.ndarray, xmin:float, xmax:float, v:float, x: np.ndarray, dx: float, q_0, shock_front_index: float = None, q_error = None) -> None:
    '''
    Plot an already time averaged quantity, handles the labeling of axes and all that. Useful when you want to see a
    plot without needing to re-run box_integrate

    Keyword arguments:
    q -- Shock-stationary-averaged quantity, can directly take the output of box_integrate
    '''

    xmin_index = int(xmin/dx)
    xmax_index = int(xmax/dx)

    plt.plot(x[xmin_index:xmax_index], q,label = q_0.DATA_NAME)

    # This is scuffed, don't use
    if (q_error == None):
        pass
    else:
        plt.errorbar(x=x[xmin_index:xmax_index],y=q, yerr=q_error, xerr=None, linestyle = 'None',elinewidth=2,capsize=3)

        
    if isinstance(shock_front_index,int):
        plt.hlines(np.mean(q[0:shock_front_index]),xmin=xmin,xmax=shock_front_index*dx, linestyles="--", color = "black")
        plt.errorbar((xmin+shock_front_index*dx)/2,np.mean(q[0:shock_front_index]), yerr=np.std(q[0:shock_front_index]), xerr=None, linestyle = 'None',elinewidth=3,capsize=4, color = 'black')
        print(f"Downstream average is: {round(np.mean(q[0:shock_front_index]),3)} +/- {round(np.std(q[0:shock_front_index]),3)}")
        print(f"Upstream average is: {round(np.mean(q[-15:-1]),3)}") # This might be scuffed
        plt.legend()

    plt.xlabel(f'{q_0.AXIS1_NAME} [${q_0.AXIS1_UNITS}$]')
    plt.ylabel(f'integrated {q_0.DATA_NAME}')
    plt.title(f'integrated along lines of $v_s = {v}$')

def get_temperature(p1x1: list,ufl1: list) -> np.ndarray:
    from skimage.transform import resize
    '''
    Get pressure from phase space data, right now this only works if integrating over x1. Output is NOT NORMALIZED!!!
    '''
    t_phase = np.array(p1x1[4])
    x_phase = np.array(p1x1[5])
    v_phase = np.array(p1x1[6]) 

    temperature = np.zeros(np.shape(p1x1[0][0,0,:]))

    for i in range(len(t_phase)):
        fvsquared = p1x1[0][i,:,:]*np.transpose([v_phase**2]*len(x_phase))
        fv = p1x1[0][i,:,:]*np.transpose([v_phase]*len(x_phase))

        second_moment = np.trapz(fvsquared,axis=0)
        first_moment = np.trapz(fv,axis=0)
        zeroth_moment = np.trapz(p1x1[0][i,:,:],axis=0)

        if (np.shape(p1x1[0][:,0,:]) != np.shape(ufl1[0])):
            ufl1_resize = np.array(resize(ufl1[0], np.shape(p1x1[0][:,0,:]), preserve_range=True))

        temperature = np.vstack((temperature,second_moment-2*ufl1_resize[i]*first_moment+np.square(ufl1_resize[i])*zeroth_moment))
    
    # Don't need that first slice because it is just zeros
    return temperature[1:,:]

def pretty_plot():
    plt.grid(visible=True,which='major',lw=.75)
    plt.grid(visible=True,which='minor',ls= ':',lw=.5)
    plt.minorticks_on()