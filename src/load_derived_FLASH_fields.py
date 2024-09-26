import yt
import numpy as np
import sys
from pathlib import Path

def derive_fields(path_to_data:str)->yt.data_objects.static_output.Dataset:
    """
    args: path_to_data (str): absolute path to the FLASH dataset to load
    Returns a yt dataset with derived fields for magnesium and aluminum number densities.
    """
    ds = yt.load(path_to_data)

    aluminum_molecular_weight = 26.981 # from wikipedia
    al_r = 1/aluminum_molecular_weight # reciprocal molecular weight, needs to be same units as sumy
    magnesium_molecular_weight = 24.305 # from wikipedia
    mg_r = 1/magnesium_molecular_weight # reciprocal molecular weight

    rqm = 100

    def make_ion_number_density(field, data):
        avogadros_number = 6.02214076e23
        ion_number_density = avogadros_number*data["flash","dens"]*data["flash","sumy"]
        return ion_number_density
    ds.add_field(("flash", "idens"), function=make_ion_number_density, units="code_mass/code_length**3",sampling_type="cell") # technically the units are wrong here, should be massless

    def make_electron_number_density(field, data):
        avogadros_number = 6.02214076e23
        electron_number_density = avogadros_number*data["flash","dens"]*data["flash","ye"]
        return electron_number_density
    ds.add_field(("flash", "edens"), function=make_electron_number_density, units="code_mass/code_length**3",sampling_type="cell") # same here


    def make_magnesium_number_density(field, data):
        mg_percentage = (data["flash","sumy"]-al_r)/(mg_r-al_r)
        mg_number_density = mg_percentage*data["flash","edens"] # we did this because Zbar is ye/sumy, so we can just use edens instead to get the number density in units that are useful to osiris
        return mg_number_density
    ds.add_field(("flash", "mgdens"), function=make_magnesium_number_density, units="code_mass/code_length**3",sampling_type="cell", force_override=True) # technically the units are wrong here, should be massless

    def make_aluminum_number_density(field, data):
        mg_percentage = (data["flash","sumy"]-al_r)/(mg_r-al_r)
        al_number_density = (1-mg_percentage)*data["flash","edens"]
        return al_number_density
    ds.add_field(("flash", "aldens"), function=make_aluminum_number_density, units="code_mass/code_length**3",sampling_type="cell",force_override=True)

    def make_Ex(field, data):
        Ex = data['flash','velz']*data["flash","magy"]-data["flash","vely"]*data["flash","magz"]
        return Ex

    def make_Ey(field, data):
        Ey = data['flash','velx']*data["flash","magz"]-data["flash","velz"]*data["flash","magx"]
        return Ey

    def make_Ez(field, data):
        Ez = data['flash','vely']*data["flash","magx"]-data["flash","velx"]*data["flash","magy"]
        return Ez

    ds.add_field(("flash", "Ex"), function=make_Ex, units="code_magnetic*code_length/code_time",sampling_type="cell")
    ds.add_field(("flash", "Ey"), function=make_Ey, units="code_magnetic*code_length/code_time",sampling_type="cell")
    ds.add_field(("flash", "Ez"), function=make_Ez, units="code_magnetic*code_length/code_time",sampling_type="cell")

    def make_vth_ele_osiris(field, data):
        return np.sqrt(data['flash','tele']*yt.units.kb_cgs/yt.units.electron_mass_cgs)

    def make_vth_ion_osiris(field, data):
        return np.sqrt(data['flash','tion']*yt.units.kb_cgs/(yt.units.electron_mass_cgs*rqm))

    ds.add_field(("flash", 'vthele'), function=make_vth_ele_osiris, units="code_velocity",sampling_type="cell",force_override=True)
    ds.add_field(("flash", 'vthion'), function=make_vth_ion_osiris, units="code_velocity",sampling_type="cell",force_override=True)

    return ds