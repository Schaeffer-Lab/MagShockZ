import sys
import os
sys.path.append('../src')
sys.path.append('../src/vysxd')
from vysxd_analysis import *
from vysxd_define import *
from make_movie_mp_multi import BlitManager  # Add this import
import osh5def
import osh5vis
import osh5io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# # Set the default font size for axis labels
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
# Set the default font size for tick labels
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

class Simulation:

    def __init__(self, simulation_name,gyrotime = 9570):
        # proj_dir = os.getcwd().removesuffix("analysis_scripts") ### Fix this pathing later
        proj_dir = '/home/dschneidinger/MagShockZ'

        raw_data_dir = f'{proj_dir}/simulations/raw_data/{simulation_name}'
        save_data_dir = f'{proj_dir}/simulations/save_data/{simulation_name}'
        restart_data_dir = f'{proj_dir}/simulations/restart/{simulation_name}'
        
        if os.path.exists(f'{raw_data_dir}/MS/DENSITY'):
            data_dir = raw_data_dir
            species_list = os.listdir(f'{data_dir}/MS/DENSITY')
        elif os.path.exists(f'{save_data_dir}/MS/DENSITY'):
            data_dir = save_data_dir
            species_list = os.listdir(f'{data_dir}/MS/DENSITY')
        elif os.path.exists(f'{restart_data_dir}/MS/DENSITY'):
            data_dir = restart_data_dir
            species_list = os.listdir(f'{data_dir}/MS/DENSITY')
        else:
            raise FileNotFoundError(f"Could not find data in either {raw_data_dir} or {save_data_dir}")

        self.gyrotime = gyrotime
        self.proj_dir = proj_dir
        self.simulation_name = simulation_name
        self.species_list = species_list
        self.data_dir = data_dir

    def make_n_panel_plot(self, TIME):
        """
        Make an n_speciesx1 plot of the phase space of each species
        """
        # Create a figure
        fig, axs = plt.subplots(len(self.species_list),1, figsize=(12,12))

        # Flatten the axes array for easy iteration
        axs = axs.flatten()


        for i, species in enumerate(self.species_list):
            phase_space = get_osiris_quantity_2d(f'{self.data_dir}/MS/PHA/p1x1/{species}/')
            im = axs[i].imshow(np.abs(phase_space[0][TIME]), origin='lower',
                            extent=[phase_space[5][0], phase_space[5][-1],phase_space[6][0],phase_space[6][-1]], aspect='auto',vmax=500,
                            cmap='hot')
            fig.colorbar(im,label=r'$f(x,v,t)$')
            axs[i].set_title(f'{species}')
            axs[i].set_xlabel(r'$x [c/\omega_{pe}]$')
            axs[i].set_ylabel(r'$v/c$')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def make_overlaid_plot(self, TIME):
        # Make a lovely color plot that uses different colors for background and piston species

        # Initialize figures for electron and ion species
        fig, ax = plt.subplots(figsize=(10, 6))

        # Variables to aggregate data
        ion_data = {}
        phase_space_bounds = {}

        omega_pi_over_omega_pe = 0.36055512754639896

        # Loop through species_list and process data
        for species in self.species_list:
            quantity = get_osiris_quantity_2d(f'{self.data_dir}/MS/PHA/p1x1/{species}/')
            
            phase_space_bounds[species] = {
                'xmin': quantity[5][0]*omega_pi_over_omega_pe,
                'xmax': quantity[5][-1]*omega_pi_over_omega_pe,
                'vmin': quantity[6][0],
                'vmax': quantity[6][-1]
            }
            
            data = np.log(np.abs(quantity[0][TIME]))  # Example: using the last timestep data and log scale for visualization
            
            if not species.startswith("electron"):
                ion_data[species] = data


        # Plot the second ion dataset on the same axes with a different color map and correct boundaries
        plot_num = 0
        for species in ion_data:
            if species == 'Silicon':
                im_ion = ax.imshow(ion_data[species], origin='lower', aspect='auto', cmap='viridis', alpha=0.8,
                                    extent=[phase_space_bounds[species]['xmin'], phase_space_bounds[species]['xmax'],
                                            phase_space_bounds[species]['vmin'], phase_space_bounds[species]['vmax']])
                cbar = fig.colorbar(im_ion, ax=ax, fraction=0.046, location='right')
                cbar.set_label('piston ions', rotation=270, labelpad=-3, size = 13)
                cbar.set_ticks([])
                plot_num += 1
            else:
                im_ion = ax.imshow(ion_data[species], origin='lower', aspect='auto', cmap='plasma', alpha=0.8, 
                                    extent=[phase_space_bounds[species]['xmin'], phase_space_bounds[species]['xmax'],
                                            phase_space_bounds[species]['vmin'], phase_space_bounds[species]['vmax']])
                cbar = fig.colorbar(im_ion, ax=ax, fraction=0.046)
                cbar.set_label('background ions', rotation=270, labelpad=-3, size = 13)
                cbar.set_ticks([])
                plot_num += 1


        # Set titles and labels
        ax.set_title(r'$Z n_i$ at ' +f't = {round(quantity[4][TIME]/self.gyrotime,2)}' + r'$\omega_{ci}$')
        ax.set_xlabel(r'$x [c/\omega_{pi}]$')
        ax.set_ylabel(r'$v/c$')
        plt.show()

    def check_energy_conservation(self):
        # Check energy conservation for the simulation
        # This is broken, need to fix
        energy_analysis = ene_analysis(f"{self.data_dir}/HIST/",osirisv='osiris4')
        print(energy_analysis.keys())
        print(len(energy_analysis))

        plt.title("Energy conservation in simulation")
        plt.scatter(energy_analysis['time'],energy_analysis['ene_conserv'], label = 'Energy Conservation',s=1)
        plt.scatter(energy_analysis['time'],energy_analysis['Ek'], label = 'Kinetic energy',s=1)
        plt.scatter(energy_analysis['time'],energy_analysis['Eemf'], label = 'Energy of EMF',s=1)

        plt.yscale('log')
        plt.xlabel(r'Time $[1/\omega_{pe}]$')
        plt.ylabel(r'Energy $[m_e c^2]$')
        plt.legend()

    def make_xt_plot(self,field):
        from pathlib import Path
        field = Path(field)
        fig, ax = plt.subplots()
        # print(Path(field).stem)
        print(f'{self.data_dir}/MS/{field}')
        q = get_osiris_quantity_1d(f'{self.data_dir}/MS/{field}/')
        q_0 = vysxd_get_data(f'{self.data_dir}/MS/{field}/{field.stem}-000000.h5')
        # Make a heatmap of quantity in (t,x) space (sorry Paulo)
        plt.imshow(np.transpose(q[0]), origin='lower', extent=[q[3][0], q[3][-1], q[4][0], q[4][-1]], aspect='auto')

        # If vysxd.data_object timeshot is supplied, use this to label axes
        plt.ylabel(f"${q_0.AXIS1_NAME}$ [${q_0.AXIS1_UNITS}$]")
        plt.xlabel(f"Time [${q_0.TIME_UNITS}$]")
        plt.colorbar(label=q_0.DATA_NAME)
        plt.show()

    def make_movie(self, plot_function, start_time=0, end_time=-1, fps=40, filename='animation.mp4'):
        """
        Creates a movie from any plotting function in the class
        Args:
            plot_function: function that creates a single plot
            start_time: starting time index
            end_time: ending time index (-1 for last frame)
            fps: frames per second
            filename: output filename
        """
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=fps)

        # Get first frame to setup figure
        fig = plt.figure()
        
        # Handle end_time = -1 case
        if end_time == -1:
            # Get length from first phase space file
            species = self.species_list[0]
            phase_space = get_osiris_quantity_2d(f'{self.data_dir}/MS/PHA/p1x1/{species}/')
            end_time = len(phase_space[0])

        with writer.saving(fig, filename,100):
            for time in range(start_time, end_time):
                plot_function(time)
                writer.grab_frame()
                plt.clf()
        
        plt.close()
    
    def make_phase_space_movie(self, species_dir, output_file='phase_space.mp4', start_time=0, end_time=-1, fps=30, vmax=500):
        """Creates a movie of phase space data using BlitManager for efficient animation.
        
        Args:
            species_dir: Path to the phase space data directory
            output_file: Output filename for the movie
            start_time: Starting time index
            end_time: Ending time index (-1 for last frame)
            fps: Frames per second
            vmax: Maximum value for colormap scaling
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        
        # Get the phase space data
        phase_space = get_osiris_quantity_2d(species_dir)
        
        if end_time == -1:
            end_time = len(phase_space[0])
        
        # Setup the figure and artist
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(np.abs(phase_space[0][0]), origin='lower',
                      extent=[phase_space[5][0], phase_space[5][-1],
                             phase_space[6][0], phase_space[6][-1]], 
                      aspect='auto', vmax=vmax, cmap='hot')
        
        title = ax.set_title("")
        fig.colorbar(im, label=r'$f(x,v,t)$')
        ax.set_xlabel(r'$x [c/\omega_{pe}]$')
        ax.set_ylabel(r'$v/c$')
        
        # Create BlitManager
        bm = BlitManager(fig.canvas, [im, title])
        
        # Draw initial frame
        plt.draw()
        plt.pause(0.1)  # Ensure the GUI is ready
        
        writer = FFMpegWriter(fps=fps)
        
        with writer.saving(fig, output_file, dpi=100):
            for frame in range(start_time, end_time):
                # Update data without redrawing everything
                im.set_array(np.abs(phase_space[0][frame]))
                title.set_text(f't = {round(phase_space[4][frame]/self.gyrotime,2)} ' + r'$\omega_{ci}^{-1}$')
                
                # Use BlitManager to update only changed artists
                bm.update()
                
                # Grab the frame
                writer.grab_frame()
        
        plt.close()

    def get_species_list(self):
        return self.species_list
    def get_data_dir(self):
        return self.data_dir
    def get_simulation_name(self):
        return self.simulation_name
    def get_proj_dir(self):
        return self.proj_dir
    def get_gyrotime(self):
        return self.gyrotime


# Create a simulation object
sim = Simulation('magshockz-v3.1.1d-200-100-100ppc')
# sim.make_n_panel_plot(-1)
# sim.make_overlaid_plot(2)
sim.make_overlaid_plot(-1)

# sim.make_movie(sim.make_overlaid_plot, filename='test.mp4')
# sim.check_energy_conservation()
# sim.make_xt_plot('FLD/b2-savg')

# from pathlib import Path
# x = Path(sim.get_proj_dir())

if __name__ == "__main__":
    sim = Simulation('magshockz-v3.1.1d-200-100-100ppc')
    
    # Make a movie for a specific species
    species = sim.get_species_list()[0]  # get first species
    species_dir = f'{sim.get_data_dir()}/MS/PHA/p1x1/{species}/'

    sim.make_phase_space_movie(species_dir, output_file=f'{species}_phase_space.mp4')