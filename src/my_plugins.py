### USAGE INSTRUCTIONS ###
# FIRST, COPY THIS FILE TO ~/.config/yt/my_plugins.py
# THEN, ADD `yt.enable_plugins()` TO YOUR SCRIPT


# This file is a plugin for yt that adds custom fields and functions to create OSIRIS-relevant fields from FLASH data

import numpy as np

@derived_field(name=("flash","idens"), sampling_type="cell", units="1/code_length**3",)
def _ion_number_density(field, data):
        avogadro = 6.02214076e23
        ion_number_density = avogadro*data["flash","dens"]*data["flash","sumy"]/units.gram
        return ion_number_density

@derived_field(name=("flash","edens"), sampling_type="cell", units="1/code_length**3",)
def _electron_number_density(field, data):
        avogadro = 6.02214076e23
        electron_number_density = avogadro*data["flash","dens"]*data["flash","ye"]/units.gram
        return electron_number_density

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

def load_for_osiris(filename:str, rqm:float = None, B_background: float = None, ion_2:str = 'Si'):
        """"
        "Load a FLASH simulation with the FLASH frontend.
        "This function is a wrapper around yt.load() that adds some
        "additional functionality for FLASH simulations.
        """

        molar_weights = {
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

        ion_1 = 'Al'

        ds = load(filename)
        ds.add_field(('flash','rqm'),
                       lambda field, data: rqm,
                       sampling_type="cell")
        c = units.speed_of_light_cgs
        e = units.electron_charge_cgs
        m_e = units.mass_electron_cgs
        
        conv_factor = c/(4*np.pi)
        
        ds.add_gradient_fields(('flash', 'magx'))
        ds.add_gradient_fields(('flash', 'magy'))
        ds.add_gradient_fields(('flash', 'magz'))


        def make_Jx(field, data):
            return conv_factor * (data["flash","magz_gradient_y"] - data["flash","magy_gradient_z"])
        def make_Jy(field, data):
            return conv_factor * (data["flash","magx_gradient_z"] - data["flash","magz_gradient_x"])
        def make_Jz(field, data):
            return conv_factor * (data["flash","magy_gradient_x"] - data["flash","magx_gradient_y"])
        

        ds.add_field(('flash', 'Jx'), function=make_Jx, units='code_magnetic/code_time', sampling_type='cell')
        ds.add_field(('flash', 'Jy'), function=make_Jy, units='code_magnetic/code_time', sampling_type='cell')
        ds.add_field(('flash', 'Jz'), function=make_Jz, units='code_magnetic/code_time', sampling_type='cell')



        def _v_ix(field, data):
                return ((m_e*data['flash','Jx']/(data['flash','dens']*e)) + data['flash','velx'])
        def _v_iy(field, data):
                return ((m_e*data['flash','Jy']/(data['flash','dens']*e)) + data['flash','vely'])
        def _v_iz(field, data):
                return ((m_e*data['flash','Jz']/(data['flash','dens']*e)) + data['flash','velz'])
        
        def _v_ex(field, data):
                return ((m_e*data['flash','Jx']/(data['flash','dens']*e) - data['flash','Jx']/(data['flash','edens']*e)) + data['flash','velx'])
        def _v_ey(field, data):
                return ((m_e*data['flash','Jy']/(data['flash','dens']*e) - data['flash','Jy']/(data['flash','edens']*e)) + data['flash','vely'])
        def _v_ez(field, data):
                return ((m_e*data['flash','Jz']/(data['flash','dens']*e) - data['flash','Jz']/(data['flash','edens']*e)) + data['flash','velz'])
        
        # Update the field definitions
        ds.add_field(('flash', 'v_ix'), function=_v_ix, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_iy'), function=_v_iy, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_iz'), function=_v_iz, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_ex'), function=_v_ex, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_ey'), function=_v_ey, units='code_velocity', sampling_type='cell')
        ds.add_field(('flash', 'v_ez'), function=_v_ez, units='code_velocity', sampling_type='cell')

        # Add the background magnetic field
        # THIS ASSUMES THAT THE BACKGROUND MAGNETIC FIELD IS IN X
        if B_background is None:
            B_background = 0.0

        ds.add_field(("flash","Bx_ext"), 
                       lambda field, data: B_background * units.gauss * data["index", "ones"],
                       units = "Gauss",
                       sampling_type="cell")
        
        ds.add_field(("flash","By_ext"),
                       lambda field, data: data["index", "zeros"] * units.gauss,
                       units = "Gauss",
                       sampling_type="cell")
        
        ds.add_field(("flash","Bz_ext"),
                       lambda field, data: data["index", "zeros"] * units.gauss,
                       units = "Gauss",
                       sampling_type="cell")

        # Internal magnetic fields
        ds.add_field(("flash","Bx_int"),
                       lambda field, data: data["flash", "magx"] - B_background * units.gauss * data["index", "ones"],
                       units = "Gauss",
                       sampling_type="cell")
        
        ds.add_field(("flash","By_int"),
                       lambda field, data: data["flash", "magy"],
                       units = "Gauss",
                       sampling_type="cell")
        
        ds.add_field(("flash","Bz_int"),
                        lambda field, data: data["flash", "magz"],
                        units = "Gauss",
                        sampling_type="cell")

        # Internal electric fields
        ds.add_field(("flash","Ex_int"),
                       lambda field, data: data['flash', 'velz'] * data["flash", "By_int"] -
                                        data["flash", "vely"] * data["flash", "Bz_int"],
                       units = "code_magnetic*code_length/code_time",
                       sampling_type="cell")
        
        ds.add_field(("flash","Ey_int"),
                       lambda field, data: data['flash', 'velz'] * data["flash", "Bx_int"] -
                                        data["flash", "velx"] * data["flash", "Bz_int"],
                       units = "code_magnetic*code_length/code_time",
                       sampling_type="cell")
        
        ds.add_field(("flash","Ez_int"),
                       lambda field, data: data['flash', 'vely'] * data["flash", "Bx_int"] -
                                        data["flash", "velx"] * data["flash", "By_int"],
                       units = "code_magnetic*code_length/code_time",
                       sampling_type="cell")

        # External electric fields
        ds.add_field(("flash","Ex_ext"),
                       lambda field, data: data["flash", "velz"] * data["flash", "By_ext"] -
                                        data["flash", "vely"] * data["flash", "Bz_ext"],
                       units = "code_magnetic*code_length/code_time",
                       sampling_type="cell")
        ds.add_field(("flash","Ey_ext"),
                       lambda field, data: data["flash", "velx"] * data["flash", "Bz_ext"] -
                                        data["flash", "velz"] * data["flash", "Bx_ext"],
                       units = "code_magnetic*code_length/code_time",
                       sampling_type="cell")
        
        ds.add_field(("flash","Ez_ext"),
                       lambda field, data: data['flash', 'vely'] * data["flash", "Bx_ext"] - 
                                        data["flash", "velx"] * data["flash", "By_ext"],
                       units = "code_magnetic*code_length/code_time",
                       sampling_type="cell")
        
        def make_vth_ele(field, data):
                return np.sqrt(data['flash','tele']*units.kb_cgs/m_e)

        def make_vth_ion_1(field, data):
                return np.sqrt(data['flash','tion']*units.kb_cgs/(m_e*rqm))
    
        def make_vth_ion_2(field, data):
                return np.sqrt(data['flash','tion']*units.kb_cgs/(m_e*rqm*(molar_weights[ion_2]/molar_weights[ion_1])))

        ds.add_field(("flash", 'vthele'), function=make_vth_ele, units="code_velocity",sampling_type="cell",force_override=True)

        ds.add_field(("flash", f'vth{str.lower(ion_1)}'), function=make_vth_ion_1, units="code_velocity",sampling_type="cell",force_override=True)

        ds.add_field(("flash", f'vth{str.lower(ion_2)}'), function=make_vth_ion_2, units="code_velocity",sampling_type="cell",force_override=True)

        return ds

