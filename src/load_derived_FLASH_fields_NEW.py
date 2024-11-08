import yt
import numpy as np

def derive_fields(path_to_data:str, rqm:int = 100) -> yt.data_objects.static_output.Dataset:
    """
    args: path_to_data (str): absolute path to the FLASH dataset to load
    Returns a yt dataset with derived fields for magnesium and aluminum number densities.
    """
    ds = yt.load(path_to_data)

    molar_weights = {
        "hydrogen": 1.00784, "helium": 4.0026, "lithium": 6.94,
        "beryllium": 9.0122, "boron": 10.81, "carbon": 12.011,
        "nitrogen": 14.007, "oxygen": 15.999, "fluorine": 18.998,
        "neon": 20.180, "sodium": 22.990, "magnesium": 24.305,
        "aluminum": 26.982, "silicon": 28.085, "phosphorus": 30.974,
        "sulfur": 32.06, "chlorine": 35.45, "argon": 39.948,
    }        


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

    def make_silicon_number_density(field, data):
        si_percentage = (data["flash","sumy"]-1/molar_weights["aluminum"])/(1/molar_weights['silicon']-1/molar_weights["aluminum"])
        si_number_density = si_percentage*data["flash","edens"] # we did this because Zbar is ye/sumy, so we can just use edens instead to get the number density in units that are useful to osiris
        return si_number_density
    try:
        ds.add_field(("flash", "sidens"), function=make_silicon_number_density, units="code_mass/code_length**3",sampling_type="cell", force_override=True) # technically the units are wrong here, should be massless
    except Exception as e:
        print(f"Error adding field sidens: {e}")

    def make_aluminum_number_density(field, data):
        si_percentage = (data["flash","sumy"]-1/molar_weights["aluminum"])/(1/molar_weights['silicon']-1/molar_weights["aluminum"])
        al_number_density = (1-si_percentage)*data["flash","edens"]
        return al_number_density
    try:
        ds.add_field(("flash", "aldens"), function=make_aluminum_number_density, units="code_mass/code_length**3",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field aldens: {e}")

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

    def make_vth_al(field, data):
        return np.sqrt(data['flash','tion']*yt.units.kb_cgs/(yt.units.electron_mass_cgs*rqm))
    
    def make_vth_si(field, data):
        return np.sqrt(data['flash','tion']*yt.units.kb_cgs/(yt.units.electron_mass_cgs*rqm*(molar_weights["silicon"]/molar_weights["aluminum"])))

    try:
        ds.add_field(("flash", 'vthele'), function=make_vth_ele, units="code_velocity",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field vthele: {e}")
    try:
        ds.add_field(("flash", 'vthal'), function=make_vth_al, units="code_velocity",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field vthal: {e}")
    try:
        ds.add_field(("flash", 'vthsi'), function=make_vth_si, units="code_velocity",sampling_type="cell",force_override=True)
    except Exception as e:
        print(f"Error adding field vthsi: {e}")

    return ds