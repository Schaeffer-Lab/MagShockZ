import yt
import numpy as np

def derive_fields(path_to_data:str, rqm:int = 100,ion_2='Si') -> yt.data_objects.static_output.Dataset:
    """
    args: path_to_data (str): absolute path to the FLASH dataset to load
            rqm (int): ratio of ion mass to electron mass. Default is 100.
    Returns a yt dataset with derived fields for magnesium and aluminum number densities.
    """
    ds = yt.load(path_to_data)

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


    def make_ion_number_density(field, data):
        avogadros_number = 6.02214076e23
        ion_number_density = avogadros_number*data["flash","dens"]*data["flash","sumy"]
        return ion_number_density
    try:
        ds.add_field(("flash", "idens"), function=make_ion_number_density, units="code_mass/code_length**3",sampling_type="cell") # technically the units are wrong here, should be massless
    except Exception as e:
        print(f"Error adding field idens: {e}")

    def make_electron_number_density(field, data):
        avogadros_number = 6.02214076e23
        electron_number_density = avogadros_number*data["flash","dens"]*data["flash","ye"]
        return electron_number_density
    try:
        ds.add_field(("flash", "edens"), function=make_electron_number_density, units="code_mass/code_length**3",sampling_type="cell") # same here
    except Exception as e:
        print(f"Error adding field edens: {e}")

    def make_ion_2_number_density(field, data):
        ion2_percentage = (data["flash","sumy"] - 1/molar_weights[ion_1]) / (1/molar_weights[ion_2] - 1/molar_weights[ion_1])
        ion2_number_density = ion2_percentage*data["flash","edens"] # we did this because Zbar is ye/sumy, so we can just use edens instead to get the number density in units that are useful to osiris
        return ion2_number_density
    try:
        ds.add_field(("flash", f"{str.lower(ion_2)}dens"), function=make_ion_2_number_density, units="code_mass/code_length**3",sampling_type="cell", force_override=True) # technically the units are wrong here, should be massless
    except Exception as e:
        print(f"Error adding field {ion_2}dens: {e}")

    def make_ion_1_number_density(field, data):
        ion2_percentage = (data["flash","sumy"]-1/molar_weights[ion_1])/(1/molar_weights[ion_2]-1/molar_weights[ion_1])
        ion1_number_density = (1 - ion2_percentage) * data["flash", "edens"]
        return ion1_number_density
    try:
        ds.add_field(("flash", f"{str.lower(ion_1)}dens"), function=make_ion_1_number_density, units="code_mass/code_length**3",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field {str.lower(ion_1)}dens: {e}")

    def make_Ex(field, data):
        Ex = data['flash','velz']*data["flash","magy"]-data["flash","vely"]*data["flash","magz"]
        return Ex

    def make_Ey(field, data):
        Ey = data['flash','velx']*data["flash","magz"]-data["flash","velz"]*data["flash","magx"]
        return Ey

    def make_Ez(field, data):
        Ez = data['flash','vely']*data["flash","magx"]-data["flash","velx"]*data["flash","magy"]
        return Ez

    try:
        ds.add_field(("flash", "Ex"), function=make_Ex, units="code_magnetic*code_length/code_time",sampling_type="cell")
    except Exception as e:
        print(f"Error adding field Ex: {e}")
    try:
        ds.add_field(("flash", "Ey"), function=make_Ey, units="code_magnetic*code_length/code_time",sampling_type="cell")
    except Exception as e:
        print(f"Error adding field Ey: {e}")
    try:
        ds.add_field(("flash", "Ez"), function=make_Ez, units="code_magnetic*code_length/code_time",sampling_type="cell")
    except Exception as e:
        print(f"Error adding field Ez: {e}")

    def make_vth_ele(field, data):
        return np.sqrt(data['flash','tele']*yt.units.kb_cgs/yt.units.electron_mass_cgs)

    def make_vth_ion_1(field, data):
        return np.sqrt(data['flash','tion']*yt.units.kb_cgs/(yt.units.electron_mass_cgs*rqm))
    
    def make_vth_ion_2(field, data):
        return np.sqrt(data['flash','tion']*yt.units.kb_cgs/(yt.units.electron_mass_cgs*rqm*(molar_weights[ion_2]/molar_weights[ion_1])))

    try:
        ds.add_field(("flash", 'vthele'), function=make_vth_ele, units="code_velocity",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field vthele: {e}")
    try:
        ds.add_field(("flash", f'vth{str.lower(ion_1)}'), function=make_vth_ion_1, units="code_velocity",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field vth{str.lower(ion_1)}: {e}")
    try:
        ds.add_field(("flash", f'vth{str.lower(ion_2)}'), function=make_vth_ion_2, units="code_velocity",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field vth{str.lower(ion_2)}: {e}")

    return ds