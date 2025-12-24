### USAGE INSTRUCTIONS ###
# FIRST, LINK THIS FILE TO ~/.config/yt/my_plugins.py (`ln -s /absolute/path/to/this/file ~/.config/yt/my_plugins.py`)
# THEN, ADD `yt.enable_plugins()` TO YOUR SCRIPT


# This file is a plugin for yt that adds OSIRIS-relevant fields to yt datasets

import numpy as np

@derived_field(name=("flash","idens"), sampling_type="cell", units="1/code_length**3",)
def _ion_number_density(field, data):
        return data['gas', 'ion_number_density']
        
@derived_field(name=("flash","edens"), sampling_type="cell", units="1/code_length**3",)
def _electron_number_density(field, data):
        return data['gas', 'El_number_density']

@derived_field(name=("flash","Ex"), sampling_type="cell", units="code_velocity*code_magnetic")
def _Ex(field, data):
        Ex = data['flash','velz']*data["flash","magy"]-data["flash","vely"]*data["flash","magz"]
        return Ex

@derived_field(name=("flash","Ey"), sampling_type="cell", units="code_velocity*code_magnetic")
def _Ey(field, data):
        Ey = data['flash','velx']*data["flash","magz"]-data["flash","velz"]*data["flash","magx"]
        return Ey

@derived_field(name=("flash","Ez"), sampling_type="cell", units="code_velocity*code_magnetic")
def _Ez(field, data):
        Ez = data['flash','vely']*data["flash","magx"]-data["flash","velx"]*data["flash","magy"]
        return Ez



def load_for_osiris(filename:str, rqm_factor:float = 1):        
        """"
        "Load a FLASH simulation with the FLASH frontend.
        "This function is a wrapper around yt.load() that adds some
        "additional functionality for FLASH simulations.
        """

        MOLAR_WEIGHTS = {
        "H": 1.0078,
        "He": 4.0026,
        "Li": 6.94,
        "Be": 9.0122,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Ne": 20.180,
        "Na": 22.990,
        "Mg": 24.305,
        "Al": 26.982,
        "Si": 28.085,
        "P": 30.974,
        "S": 32.06,
        "Cl": 35.45,
        "Ar": 39.948,
        }


        ds = load(filename)

        c = units.speed_of_light_cgs
        e = units.electron_charge_cgs
        m_e = units.mass_electron_cgs
        
        ion_mass_thresholds = [25.5] # Units of proton mass

        def make_species_mask(field, data):
                """
                Split up the simulation domain based on whether the average ion mass at a given point is above or below the specified threshold(s)
                Will return an object where each point is assigned a value (1, 2, ..., len(ion_mass_thresholds)+1)

                Counterintuitively, a value of 1 corresponds with the highest ion mass bin and a value of len(ion_mass_thresholds)+1 corresponds with the lowest ion mass
                """
                bins = [-np.inf] + ion_mass_thresholds + [np.inf]
                species_mask = np.digitize(1/data["flash","sumy"], bins)
                
                return species_mask

        ds.add_field(("flash","species_mask"),
                function=make_species_mask,
                units="",
                sampling_type="cell",
                force_override=False) 
        
        # Silicon and Aluminum are both hard-coded for MagShockZ
        def make_silicon_density(field, data):
                silicon_density = data['flash','edens'] * (data["flash","species_mask"] == 1)
                return silicon_density
        
        ds.add_field(("flash",f"sidens"),
                function=make_silicon_density,
                units="1/code_length**3",
                sampling_type="cell",
                force_override=False)
        

        def make_aluminum_density(field, data):
                aluminum_density = data['flash','edens'] * (data["flash","species_mask"] == 2)
                return aluminum_density

        ds.add_field(("flash","aldens"),
                function=make_aluminum_density,
                units="1/code_length**3",
                sampling_type="cell",
                force_override=False)

        def make_vthele(field, data):
               vthele = np.sqrt(data['flash','tele'] * units.boltzmann_constant / units.electron_mass_cgs)
               return vthele.to('code_velocity')
        
        ds.add_field(('flash', 'vthele'), function=make_vthele, units='code_velocity', sampling_type='cell', force_override=False)

        def make_vthion(field, data):
               vthion = np.sqrt(data['flash','tion'] * units.boltzmann_constant / (units.proton_mass/data['flash', 'sumy']))
               return vthion.to('code_velocity')

        def make_vthal(field, data):
               return data['flash','vthion']
        
        def make_vthsi(field, data):
               return data['flash','vthion']
        
        ds.add_field(('flash', 'vthion'), function=make_vthion, units='code_velocity', sampling_type='cell', force_override=False)
        ds.add_field(('flash', 'vthal'), function=make_vthal, units='code_velocity', sampling_type='cell', force_override=False)
        ds.add_field(('flash', 'vthsi'), function=make_vthsi, units='code_velocity', sampling_type='cell', force_override=False)
        
        # We need the gradients in order to calculate ampere's law
        ds.add_gradient_fields(('flash', 'magx'))
        ds.add_gradient_fields(('flash', 'magy'))
        ds.add_gradient_fields(('flash', 'magz'))



        def make_Jx(field, data):
            return c/(4*np.pi) * (data["flash","magz_gradient_y"] - data["flash","magy_gradient_z"])
        def make_Jy(field, data):
            return c/(4*np.pi) * (data["flash","magx_gradient_z"] - data["flash","magz_gradient_x"])
        def make_Jz(field, data):
            return c/(4*np.pi) * (data["flash","magy_gradient_x"] - data["flash","magx_gradient_y"])
        

        ds.add_field(('flash', 'Jx'), function=make_Jx, units='code_magnetic/code_time', sampling_type='cell')
        ds.add_field(('flash', 'Jy'), function=make_Jy, units='code_magnetic/code_time', sampling_type='cell')
        ds.add_field(('flash', 'Jz'), function=make_Jz, units='code_magnetic/code_time', sampling_type='cell')


        # NOTE: The contribution to the velocities due to currents, which are calculated from ampere's law, must be exempt from the velocity scaling used later
        # that is done to preserve mach number. Therefore, we divide now by the square root of this rqm_factor, so that the current contribution remains the same
        def _v_ix(field, data):
                return ((m_e*data['flash','Jx']/(data['flash','dens']*e)) + data['flash','velx'])
        def _v_iy(field, data):
                return ((m_e*data['flash','Jy']/(data['flash','dens']*e)) + data['flash','vely'])
        def _v_iz(field, data):
                return ((m_e*data['flash','Jz']/(data['flash','dens']*e)) + data['flash','velz'])
        
        def _v_ex(field, data):
                return ((m_e*data['flash','Jx']/(data['flash','dens']*e) - data['flash','Jx']/(data['flash','edens']*e))/np.sqrt(rqm_factor) + data['flash','velx'])
        def _v_ey(field, data):
                return ((m_e*data['flash','Jy']/(data['flash','dens']*e) - data['flash','Jy']/(data['flash','edens']*e))/np.sqrt(rqm_factor) + data['flash','vely'])
        def _v_ez(field, data):
                return ((m_e*data['flash','Jz']/(data['flash','dens']*e) - data['flash','Jz']/(data['flash','edens']*e))/np.sqrt(rqm_factor) + data['flash','velz'])
        
        # Update the field definitions
        ds.add_field(('flash', 'v_ix'), function=_v_ix, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_iy'), function=_v_iy, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_iz'), function=_v_iz, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_ex'), function=_v_ex, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_ey'), function=_v_ey, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_ez'), function=_v_ez, units='code_velocity', sampling_type='cell')

       
        return ds

