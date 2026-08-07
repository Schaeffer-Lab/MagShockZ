# %% [markdown]
# Verify the yt-plugin thermal-velocity fields against the FLASH temperatures
#
# The flash2osiris yt plugin (`flash_osiris/yt_plugin.py`, symlinked to
# ~/.config/yt/my_plugins.py) builds thermal-velocity fields OSIRIS uses to load particles:
#   vthele = sqrt(k_B tele / m_e)            (electron, full electron mass)
#   vthion = sqrt(k_B tion / (m_p / ye))     (ion, MASS-PER-CHARGE  m_eff = m_p/ye = m_i/Z)
#
# This script checks that those velocities round-trip back to the FLASH temperatures, i.e.
# the temperature recovered from vth equals the temperature field it was built from:
#   T_e = m_e      * vthele^2 / k_B   ==  tele
#   T_i = (m_p/ye) * vthion^2 / k_B   ==  tion
# Everything is kept in real (CGS / eV) units — no OSIRIS normalization here.
#
# It also shows the WRONG ion recovery (using the full ion mass m_p/sumy instead of the
# mass-per-charge m_p/ye), which lands a factor Z = ye/sumy too low — the bug the
# m_p/ye fix corrects.
#
# Run with the `analysis` conda env, cell-by-cell or as a script.

# %%
import warnings

import numpy as np
import matplotlib.pyplot as plt
import yt

warnings.filterwarnings("ignore")
yt.set_log_level(50)
yt.enable_plugins()  # registers load_for_osiris + vthele/vthion from the flash2osiris plugin
# ~/.config/yt/my_plugins.py is symlinked to flash2osiris/flash_osiris/yt_plugin.py.

FLASH_FILE = "/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2026-03/MagShockZ_hdf5_plt_cnt_0009"
LINE_START = (0.0, 0.07, 0.0)  # cm
LINE_END   = (0.0, 0.70, 0.0)  # cm
RQM_FACTOR = 100               # only affects velocity fields, not the temperature round-trip

# %%
# Load FLASH through the plugin and sample the line of sight.
ds = yt.load_for_osiris(FLASH_FILE, rqm_factor=RQM_FACTOR)
ray = ds.ray(LINE_START, LINE_END)
order = np.argsort(ray["t"])

length_cm = float(np.linalg.norm(np.subtract(LINE_END, LINE_START)))
x_um = np.asarray(ray["t"][order]) * length_cm * 1e4   # distance along ray [microns]

# Real-unit fields (CGS).  yt carries units, so the arithmetic below is dimensionally safe.
kB = yt.units.boltzmann_constant
m_e = yt.units.mass_electron
m_p = yt.units.proton_mass

tele = ray["flash", "tele"][order]      # K
tion = ray["flash", "tion"][order]      # K
ye   = ray["flash", "ye"][order]        # electrons per nucleon  (Z/A)
sumy = ray["flash", "sumy"][order]      # ions per nucleon       (1/A)
vthele = ray["flash", "vthele"][order].to("cm/s")
vthion = ray["flash", "vthion"][order].to("cm/s")

Z_local = (ye / sumy).to_value("dimensionless")   # local ionization = ye * A
print(f"local ionization Z = ye/sumy : min {Z_local.min():.2f}, median {np.median(Z_local):.2f}, max {Z_local.max():.2f}")

# %%
# Recover temperatures from the thermal velocities, in eV.  (T = m vth^2 / k_B, OSIRIS
# convention with no 1/2; equivalently the loading energy m vth^2 equals k_B T.)
def to_eV(energy):
    return energy.to_value("eV")

Te_flash = to_eV(kB * tele)
Ti_flash = to_eV(kB * tion)

Te_from_vth = to_eV(m_e * vthele**2)                 # electron mass — correct
Ti_from_vth = to_eV((m_p / ye) * vthion**2)          # mass-per-charge m_p/ye — correct (matches my_plugins)
Ti_full_mass = to_eV((m_p / sumy) * vthion**2)       # full ion mass m_p/sumy — WRONG: = Z * tion

# Relative errors of the round-trip.
err_e = np.abs(Te_from_vth - Te_flash) / Te_flash
err_i = np.abs(Ti_from_vth - Ti_flash) / Ti_flash

print(f"electron round-trip (m_e)      max rel err : {err_e.max():.2e}")
print(f"ion round-trip      (m_p/ye)   max rel err : {err_i.max():.2e}")
print(f"ion with full mass  (m_p/sumy) : over by ~{np.median(Ti_full_mass / Ti_flash):.2f}  (= median Z)")
print(f"  i.e. vthion is sqrt(Z) larger than the full-mass thermal speed: "
      f"median ratio {np.median((vthion / np.sqrt(kB*tion/(m_p/sumy))).to_value('dimensionless')):.3f} "
      f"vs sqrt(median Z) {np.sqrt(np.median(Z_local)):.3f}")

# %%
# Plots — real units throughout.
fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Electron temperature
ax[0].plot(x_um, Te_flash, "k-", lw=2, label=r"FLASH $T_e$ (tele)")
ax[0].plot(x_um, Te_from_vth, "C0--", lw=2, label=r"from vthele: $m_e\,v_{th}^2/k_B$")
ax[0].set_ylabel(r"$T_e$ [eV]"); ax[0].set_yscale("log"); ax[0].legend()
ax[0].set_title("Electron temperature round-trip")

# Ion temperature
ax[1].plot(x_um, Ti_flash, "k-", lw=2, label=r"FLASH $T_i$ (tion)")
ax[1].plot(x_um, Ti_from_vth, "C1--", lw=2, label=r"from vthion: $(m_p/y_e)\,v_{th}^2/k_B$  (correct)")
ax[1].plot(x_um, Ti_full_mass, "C3:", lw=2, label=r"from vthion w/ full mass $m_p/\mathrm{sumy}$  ($=Z\,T_i$, over by $Z$)")
ax[1].set_ylabel(r"$T_i$ [eV]"); ax[1].set_yscale("log"); ax[1].legend()
ax[1].set_title("Ion temperature round-trip (dotted = the full-mass mistake, $Z\\times$ high)")

# Relative error
ax[2].semilogy(x_um, err_e + 1e-18, "C0-", label=r"electron")
ax[2].semilogy(x_um, err_i + 1e-18, "C1-", label=r"ion (m_p/ye)")
ax[2].set_ylabel("relative error"); ax[2].set_xlabel(r"distance along line of sight [$\mu$m]")
ax[2].axhline(1e-12, color="gray", ls=":", lw=1, label="1e-12 (machine-precision band)")
ax[2].legend(); ax[2].set_title("Round-trip relative error (should be ~machine precision)")

fig.tight_layout()
plt.show()
