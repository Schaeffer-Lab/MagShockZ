import numpy as np
'''
You must input these quantities yourself!
'''
ne_cgs = 1e18 # [cm^-3]
Te_eV = 5.0 # [eV]
B_gauss = 50000 # [G]
Z = 1 # [e]
ion_mass = 100 # [m_e]
print("-"*10+f" n_e = {ne_cgs} cm^-3 "+"-"*10)


e = 4.80320425e-10 # [statC] = [cm^3/2⋅g^1/2⋅s^−1]
m_e = 9.1093837139e-28 # [g]
c = 2.99792458e10 # [cm/s]
kb = 8.617333262e-5 # [eV/K]
ergs_per_eV = 1.602e-12

omega_pe = np.sqrt(4*np.pi*ne_cgs*e**2/m_e) # [rad/s]
f_pe = omega_pe/(2*np.pi) # [1/s]
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
print(f"we should set tmax = {round(5*(omega_pe/omega_ci))}")
Mach_number = 10
vshock_over_vpiston = 4/3
L_box = (vshock_over_vpiston*Mach_number*alfven_speed_cgs/c)*(5*(omega_pe/omega_ci))*2
print("and the maximum bound of the box should be approx "\
      f"{int(L_box)}")
print("-"*50)

## assume that OSIRIS uses the convention where v_th = sqrt(kT/m)
lambda_d = np.sqrt(Te_eV*ergs_per_eV/(4*np.pi*ne_cgs*e**2))
print(f"Te of {Te_eV} eV corresponds to debye length of {np.format_float_scientific(lambda_d,2)} cm")
print(f"in simulation units this would be = {round(lambda_d*omega_pe/c,4)}")
print("")
print(f"Te of {Te_eV} eV corresponds to v_th = {np.format_float_scientific(np.sqrt(Te_eV*ergs_per_eV/m_e),2)} cm/s")
print(f"in simulation units this would be = {round(np.sqrt(Te_eV*ergs_per_eV/m_e)/c,4)}")
print(f"for the ions, this same temperature would be = {np.format_float_scientific(np.sqrt(1/ion_mass)*np.sqrt(Te_eV*ergs_per_eV/m_e)/c,4)}")

print(f"in order to resolve the debye length, this would require the simulation to have {int(L_box/(lambda_d*omega_pe/c))} cells")