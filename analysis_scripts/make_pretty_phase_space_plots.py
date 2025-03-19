import sys
import os
sys.path.append('../src')
sys.path.append('../src/vysxd')
from vysxd_analysis import *
from vysxd_define import *
import osh5def
import osh5vis
import osh5io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

class simulation:

    def __init__(self, simulation_name,gyrotime = 9570):
        # proj_dir = os.getcwd().removesuffix("analysis_scripts") ### Fix this pathing later
        proj_dir = '/home/dschneidinger/MagShockZ'
        data_dir = f'{proj_dir}/simulations/raw_data/{simulation_name}'
        species_list = os.listdir(f'{data_dir}/MS/DENSITY')

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
            quantity = get_osiris_quantity_2d(f'{self.proj_dir}/simulations/raw_data/{self.simulation_name}/MS/PHA/p1x1/{species}/')
            
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
            if plot_num == 0:
                im_ion = ax.imshow(ion_data[species], origin='lower', aspect='auto', cmap='plasma', alpha=0.8,
                                    extent=[phase_space_bounds[species]['xmin'], phase_space_bounds[species]['xmax'],
                                            phase_space_bounds[species]['vmin'], phase_space_bounds[species]['vmax']])
                cbar = fig.colorbar(im_ion, ax=ax, fraction=0.046, location='right')
                cbar.set_label('background ions', rotation=270, labelpad=-3, size = 13)
                cbar.set_ticks([])
                plot_num += 1
            else:
                im_ion = ax.imshow(ion_data[species], origin='lower', aspect='auto', cmap='viridis', alpha=0.8, 
                                    extent=[phase_space_bounds[species]['xmin'], phase_space_bounds[species]['xmax'],
                                            phase_space_bounds[species]['vmin'], phase_space_bounds[species]['vmax']])
                cbar = fig.colorbar(im_ion, ax=ax, fraction=0.046)
                cbar.set_label('piston ions', rotation=270, labelpad=-3, size = 13)
                cbar.set_ticks([])
                plot_num += 1


        # Set titles and labels
        ax.set_title(r'$Z n_i$ (log) at ' +f't = {round(quantity[4][TIME]/self.gyrotime,2)}' + r'$\omega_{ci}$')
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
        # q_0 = vysxd_get_data(f'{self.data_dir}/MS/{field}/{field.stem}-000000.h5')
        # Make a heatmap of quantity in (t,x) space (sorry Paulo)
        plt.imshow(np.transpose(q[0]), origin='lower', extent=[q[3][0], q[3][-1], q[4][0], q[4][-1]], aspect='auto')

        # If vysxd.data_object timeshot is supplied, use this to label axes
        # plt.ylabel(f"${q_0.AXIS1_NAME}$ [${q_0.AXIS1_UNITS}$]")
        # plt.xlabel(f"Time [${q_0.TIME_UNITS}$]")
        # plt.colorbar(label=q_0.DATA_NAME)
        plt.show()

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
sim = simulation('magshockz-v3.1.1d')
# sim.make_n_panel_plot(-1)
sim.make_overlaid_plot(2)
sim.make_overlaid_plot(-1)
# sim.check_energy_conservation()
# sim.make_xt_plot('DENSITY/Aluminum/charge')

# from pathlib import Path
# x = Path(sim.get_proj_dir()).