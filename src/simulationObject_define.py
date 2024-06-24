import os
import sys
sys.path.append('vysxd')
from vysxd_define import *
from vysxd_analysis import *
import matplotlib.animation as manimation
import matplotlib.pyplot as plt

class simulation(object):
    def __init__(self,path_to_data,dim):
        '''
        Input data should be a relative path to a directory that contains your simulation output
        '''
        self.data = path_to_data
        self.simulation_name = self.data.removesuffix("/data")
        self.dim = dim
        self.mi = 1


    def set_box(self, xmin, xmax, tmin, tmax,vshock):
        self.xmin = xmin; self.xmax = xmax; self.tmin = tmin; self.tmax = tmax; self.vshock = vshock

    def get_box(self):
        '''
        Returns the boundaries of integration as a dictionary
        '''
        return {'xmin':self.xmin,\
                'xmax':self.xmax,\
                'tmin':self.tmin,\
                'tmax':self.tmax,\
                'vshock':self.vshock}
    def set_mi(self,mi):
        self.mi = mi
    def get_mi(self):
        return self.mi
    
    def set_simulation_name(self,name):
        self.simulation_name = name
    def get_simulation_name(self):
        return self.simulation_name
    
    def get_dim(self):
        return self.dim
    
    def make_phasespace_mov(self,species):
        p1x1_files = np.sort(os.listdir(f'{self.data}/MS/PHA/p1x1/{species}/')) # Create a sorted list of filenames you will be analyzing

        if not (os.path.isdir(f'{self.simulation_name}/figures')): # Create a directory for the animation, if there isn't one already
            os.makedirs(f'{self.simulation_name}/figures')
        # Define the meta data for the movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='phase-space-animation', artist='Matplotlib',
                        comment='visualizing the phase space evolution of the distribution function') # Describe what the animation is
        writer = FFMpegWriter(fps=24, metadata=metadata) # you can adjust the fps here.

        # Initialize the movie
        fig = plt.figure()

        plt.tight_layout()
        # Update the frames for the movie
        with writer.saving(fig, f"{self.simulation_name}/figures/{species}_phase_space.mp4", dpi=400):
            for file in p1x1_files:
                p1x1 = vysxd_get_data(f'{self.data}/MS/PHA/p1x1/{species}/{file}') # Pull the phase space data
                plt.clf() # This clears the figure so you don't make a million colorbars lol
                plt.title(f't = {round(p1x1.TIME[0],1)} $[1/\omega_p]$')
                plt.imshow(np.log(-p1x1.DATA), origin='lower', extent=[p1x1.X[0], p1x1.X[-1],p1x1.Y[0],p1x1.Y[-1]], aspect='auto', vmin = -6)
                plt.colorbar(label=r'$log(f_'+species[0]+r'(x,v,t))$')
                plt.xlabel(r'$x [c/\omega_{pe}]$') # Might need to rework this if the axes aren't static
                plt.ylabel(r'$\gamma v/c$')
                # plt.ylim(-0.3,0.3)
                writer.grab_frame()



class shockFrameScalar(simulation):
    def __init__(self,path_to_q):
        '''
        Your code here
        add implementation to automatically label axes and take averages
        '''
        if self.dim == 1: 
            self.quantity_raw = get_osiris_quantity_1d(path_to_q)

        if self.dim == 2:
            self.quantity_raw = get_osiris_quantity_2d(path_to_q)
    def show_box(self):
        try:
            box = self.get_box()

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


                _, ax = plt.subplots()
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

        except:
            raise ValueError("box has not been defined for simulation object")
    def plot_quantity(self):
        '''
        your code here
        '''
    def get_averaged_quantity(self):
        return
    def
class quantity3d(object):
    def __init__(self,q1,q2,q3, vshock, xmin, xmax, tmin, tmax, labels = None):
        
        # Start by saving averaged quantities in x y and z
        self.x = q1; self.y = q2; self.z = q3
        
        # It is useful for this object to hold the boundaries of your integration
        self.v = vshock
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        if not isinstance(labels, type(None)):
            pass