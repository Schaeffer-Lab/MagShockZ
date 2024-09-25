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


    def make_ion_number_density(field, data):
        ion_number_density = yt.units.avogadros_number_mks*data["flash","dens"]*data["flash","sumy"]
        return ion_number_density
    ds.add_field(("flash", "idens"), function=make_ion_number_density, units="1/code_length**3",sampling_type="cell") # technically the units are wrong here, should be massless

    def make_electron_number_density(field, data):
        electron_number_density = yt.units.avogadros_number_mks*data["flash","dens"]*data["flash","ye"]
        return electron_number_density
    ds.add_field(("flash", "edens"), function=make_electron_number_density, units="1/code_length**3",sampling_type="cell") # same here


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

    return ds