import numpy as np
import sys
import os

'''
You must input these quantities yourself!
'''
print_flag = True
if print_flag == False:
     sys.stdout = open(os.devnull, 'w')

ne_cgs = 5e+18 # [cm^-3]
Te_eV = 30.0 # [eV]
# B_gauss = 50000  # [G]
B_gauss = 110120  # [G]
Z = 1 # [e]
ion_mass = 100 # [m_e]
piston_ion_mass = 0.979*ion_mass # this assumes that the piston is Mg and the background is Al
v_piston = None
print("-"*10+f" n_e = {ne_cgs} cm^-3 "+"-"*10)


e = 4.80320425e-10 # [statC] = [cm^3/2⋅g^1/2⋅s^−1]
m_e = 9.1093837139e-28 # [g].
c = 2.99792458e10 # [cm/s]
kb = 8.617333262e-5 # [eV/K]
ergs_per_eV = 1.602e-12

omega_pe = np.sqrt(4*np.pi*ne_cgs*e**2/m_e) # [rad/s]

print(f"angular electron plasma frequency is: {np.format_float_scientific(omega_pe,2)} rad/s")
print("-"*50)
print(f"times are normalized to inverse plasma frequency, one unit of simulation time corresponds to "\
      f"{np.format_float_scientific(1e9/omega_pe,2)} ns")
print(f"space is normalized to electron inertial length, one unit of simulation length corresponds to "\
      f"{np.format_float_scientific(c/omega_pe,2)} cm")
print("-"*50)


B_sim_units = B_gauss*e/(omega_pe*m_e*c)
print(f'{B_gauss} G in simulation units would be {round(B_sim_units,3)}')
print("")


omega_ce = e*B_gauss/(m_e*c)
print(f"electron gyrotime would be {round(1e9/omega_ce,4)} ns")
print(f"this corresponds to {round(omega_pe/(omega_ce),3)} simulation times")
print('')

omega_ci = Z*e*B_gauss/(m_e*ion_mass*c)
print(f"ion gyrotime (with ion mass of {ion_mass}) would be {round(1e9/omega_ci,4)} ns")
print(f"this corresponds to {round(omega_pe/omega_ci,3)} simulation times")
print("-"*50)
alfven_speed_cgs = B_gauss/np.sqrt(4*np.pi*ne_cgs*ion_mass*m_e)
print(f"alfven speed is {np.format_float_scientific(alfven_speed_cgs,2)} cm/s")
print(f"in simulation units this would be {round(alfven_speed_cgs/c,6)}")
print("")

print(f"In order to simulate 5 ion gyrotimes by the time the shock has reached halfway across the box")
print(f"we should set tmax = {int(5*(omega_pe/omega_ci))}")
Mach_number = 10
vshock_over_vpiston = 4/3
if v_piston:
      L_box = (vshock_over_vpiston*v_piston/c)*(5*(omega_pe/omega_ci))*2
else:
      L_box = (vshock_over_vpiston*Mach_number*alfven_speed_cgs/c)*(5*(omega_pe/omega_ci))*2
print("and the maximum bound of the box should be approx "\
      f"{int(L_box)}")
print("-"*50)

## assume that OSIRIS uses the convention where v_th = sqrt(kT/m)
lambda_d = np.sqrt(Te_eV*ergs_per_eV/(4*np.pi*ne_cgs*e**2))
print(f"Te of {Te_eV} eV corresponds to debye length of {np.format_float_scientific(lambda_d,2)} cm")
print(f"in simulation units this would be = {(lambda_d*omega_pe/c):.4f}")
print("")
print(f"Te of {Te_eV} eV corresponds to v_th = {np.format_float_scientific(np.sqrt(Te_eV*ergs_per_eV/m_e),2)} cm/s")
print(f"in simulation units this would be = {(np.sqrt(Te_eV*ergs_per_eV/m_e)/c):.4f}")
print(f"for the ions, this same temperature would be = {np.format_float_scientific(np.sqrt(1/ion_mass)*np.sqrt(Te_eV*ergs_per_eV/m_e)/c,4)}")

print(f"in order to resolve the debye length, this would require the simulation to have {int(L_box/(lambda_d*omega_pe/c))} cells")

# Function to convert the input quantity to simulation units
def convert_to_simulation_units(quantity, units):
    # Conversion factors (example values, adjust as necessary)
    if '^' in units:
      units = units.split('^')
      units[1] = int(units[1])
    else:
      units = [units, 1]

    conversion_factors = {
        'cm/s': c,
        'cm': c/omega_pe,
        'G': (omega_pe*m_e*c)/e,
        's': 1/omega_pe,
        # Add other units and their conversion to simulation units here
    }
    
    if units[0] in conversion_factors:
        return np.format_float_scientific(quantity/(conversion_factors[units[0]]**units[1]),4)
    else:
        print("Unsupported units.")
        return None

while False: # Set to true if you would like to use the notebook interactively
      # Take input from the user
      user_input = input("Enter a quantity and its units to convert to simulation units (e.g., '1000 cm/s'): ")
      if user_input.lower() == 'exit':
            break
      # Split the input into quantity and units
      try:
            quantity_str, units = user_input.split()
            quantity = float(quantity_str)
      except ValueError:
            print("Error: Please enter a valid quantity and units.")
      else:
            # Convert to simulation units
            converted_quantity = convert_to_simulation_units(quantity, units)
            if converted_quantity is not None:
                  print(f"{quantity} {units} in simulation units is approximately {converted_quantity}")
