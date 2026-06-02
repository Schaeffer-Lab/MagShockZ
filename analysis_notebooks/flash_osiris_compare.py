# %% [markdown]
# FLASH vs. OSIRIS — 1-to-1 comparison along the line of sight
#
# Compares the FLASH 3D MHD run against the OSIRIS 1D PIC run for densities,
# electron/ion temperatures, and magnetic-field compression (B_z), along the same
# line of sight the OSIRIS run was built from.
#
# Run interactively cell-by-cell (VS Code / Jupyter `# %%` cells) with the `analysis`
# conda env, or `import flash_osiris_compare as cmp` and call the functions directly.
#
# Key conventions
# ---------------
# * Space: electron skin depths c/omega_pe with n0 = 5e18 cm^-3. The FLASH lineout
#   (0.63 cm) is 2651 c/omega_pe; the OSIRIS box is 2648 c/omega_pe — they match.
# * Line of sight: (0, 0.07, 0) cm -> (0, 0.70, 0) cm  (from runme_perlmutter.sh).
# * Time: electron units tau = t[s] * omega_pe. The OSIRIS IC was extracted from FLASH
#   plt_cnt_0009 (2.5 ns), so OSIRIS t=0 == FLASH plt_cnt_0009. OSIRIS then runs only
#   ~0.22 ns, less than one FLASH dump interval, so the one true time overlap is the IC.
# * Repo helpers: MagShockZRun (units), moments.moment (phase-space moments),
#   Ray (FLASH lineout in OSIRIS-normalised units).

# %%
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py
import osh5io
import astropy.units as u
import astropy.constants as const
import yt

warnings.filterwarnings("ignore")
yt.set_log_level(50)

REPO = Path("/home/dschneidinger/MagShockZ")
sys.path.append(str(REPO / "src"))
sys.path.append(str(REPO / "init_nopython"))
from analysis_utils import MagShockZRun   # OSIRIS unit conversions
from moments import moment                # phase-space velocity moments
from fitting_functions import Ray         # FLASH lineout in OSIRIS-normalised units

# ---- Run configuration (mirrors runme_perlmutter.sh that built the OSIRIS run) ----
OSIRIS_DIR    = Path("/mnt/cellar/shared/simulations/OSIRIS_MagShockZ1D_1.3-Schneidinger-2026-05")
FLASH_DIR     = Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2026-03")
FLASH_IC_DUMP = 9                  # OSIRIS IC was extracted from plt_cnt_0009
LINE_START    = (0.0, 0.07, 0.0)   # cm  (target surface)
LINE_END      = (0.0, 0.70, 0.0)   # cm  (downstream)
REF_DENSITY   = 5e18 * u.cm**-3    # OSIRIS normalisation density n0
RQM_FACTOR    = 100                # artificial mass-ratio reduction
B0            = 1e5 * u.G
OSIRIS_TMAX   = 27340              # 1/omega_pe  (from the input deck)

# %%
# Unit conversions come from MagShockZRun so they match the rest of the repo.
run = MagShockZRun(input_deck=str(OSIRIS_DIR / "magshockz_gpu.1d"),
                   norm_density=REF_DENSITY, B0=B0, Z=13, m_i=27 * const.m_p)

omega_pe = run.omega_pe                              # rad/s
wpe = float(omega_pe.to_value(u.rad / u.s))          # plain 1/s value (rad is dimensionless)
me_c2_eV = (const.m_e * const.c**2).to("eV").value   # 511 keV
RQM = {"e": 1.0,
       "al": run.deck.species["al"].rqm,             # 38
       "si": run.deck.species["si"].rqm}             # 39

# Ion charge states used to build the OSIRIS IC (runme_perlmutter.sh: al/si = 13).
Z_CHARGE = {"al": 13, "si": 14}

# Factor converting the velocity variance <(u-<u>)^2> to temperature in units of m_e c^2.
#   electrons: just rqm (=1)            -> T_e = m_e c^2 <du^2>   (canonical, matches repo)
#   ions:      rqm * Z                  -> extra charge-state factor Z
# The Z is needed because the IC velocity normalization (c/sqrt(rqm_factor) in
# fitting_functions.py) was applied with the *real* ion mass m_p/sumy, while OSIRIS
# rqm = m/q folds in the charge q = Z e.  The rqm-only recovery (temperature_anisotropy.py)
# therefore underestimates T_i by Z.  Verified by the t=0 round-trip: OSIRIS t=0 must
# reproduce the FLASH tion it was built from, and the closing factor is exactly Z (=13).
TEMP_MASS_FACTOR = {sp: RQM[sp] * Z_CHARGE.get(sp, 1) for sp in RQM}


def seconds_to_wpe(t_s):
    # Real time [s] -> OSIRIS electron time units [1/omega_pe]
    return t_s * wpe


print(f"omega_pe                = {omega_pe:.3e}")
print(f"c/omega_pe (skin depth) = {(const.c.cgs.value / wpe * u.cm).to('um'):.3f}")
print(f"reduced ion mass ratios = {RQM}")
print(f"OSIRIS runtime          = 0 -> {OSIRIS_TMAX}/wpe  =  {OSIRIS_TMAX / wpe * 1e9:.3f} ns")
print(f"FLASH dump {FLASH_IC_DUMP} defines OSIRIS t=0")


# %%
# ---- FLASH side: lineout in OSIRIS-normalised units (same Ray used by the IC generator) ----
def flash_lineout(dump_index):
    path = FLASH_DIR / f"MagShockZ_hdf5_plt_cnt_{dump_index:04d}"
    ray = Ray(str(path), LINE_START, LINE_END, rqm_factor=RQM_FACTOR,
              reference_density=REF_DENSITY.value * yt.units.cm**-3)
    raw, order = ray._get_sorted_ray()
    kB = const.k_B.to("eV/K").value
    return {
        "x":    ray.osiris_length,                            # c/omega_pe
        "t_ns": float(ray.ds.current_time.to("s")) * 1e9,     # ns (absolute FLASH time)
        "ne":   ray._get_field_values("edens"),               # n / n0
        "nal":  ray._get_field_values("aldens"),              # n / n0
        "nsi":  ray._get_field_values("sidens"),              # n / n0
        "Te":   np.asarray(raw["flash", "tele"][order]) * kB, # eV
        "Ti":   np.asarray(raw["flash", "tion"][order]) * kB, # eV
        "Bz":   ray._get_rotated_vector_component("magz"),    # B / B_osiris  (= OSIRIS b3)
    }


# %%
# ---- OSIRIS side: read normalised diagnostics directly ----
def _x_axis(d):
    return np.linspace(d.axes[0].min, d.axes[0].max, d.axes[0].size)


def osiris_density(path):
    # Number density n/n0 along x1 (|charge| = n/n0 in OSIRIS units)
    d = osh5io.read_h5(path)
    return _x_axis(d), np.abs(np.asarray(d)), d.run_attrs["TIME"][0]


def osiris_bz(path):
    d = osh5io.read_h5(path)
    return _x_axis(d), np.asarray(d), d.run_attrs["TIME"][0]


def osiris_temperature(species, path):
    # Temperature [eV] from the 2nd velocity moment of a p1x1 phase space:
    #   T = TEMP_MASS_FACTOR * m_e c^2 * <(p1 - <p1>)^2>
    # moment() returns the velocity variance <du^2>; see TEMP_MASS_FACTOR above for the
    # rqm (and ion charge-state Z) scaling that recovers the real temperature.
    # Empty cells -> NaN.
    d = osh5io.read_h5(path)
    variance = moment(d, order=2, axis="p1")              # <(p1-<p1>)^2> per x cell
    x = np.linspace(d.axes[1].min, d.axes[1].max, d.axes[1].size)
    T = TEMP_MASS_FACTOR[species] * me_c2_eV * variance
    T[variance <= 0] = np.nan
    return x, T, d.run_attrs["TIME"][0]


# %%
# ---- pick the OSIRIS dump closest to a target time (electron units) ----
def _dump_time(path):
    with h5py.File(path, "r") as f:
        return float(f.attrs["TIME"][0])


def nearest_dump(glob_pattern, target_wpe):
    files = sorted(glob.glob(glob_pattern))
    times = np.array([_dump_time(f) for f in files])
    i = int(np.argmin(np.abs(times - target_wpe)))
    return files[i], times[i]


def osiris_state(target_wpe):
    # All comparison fields from the OSIRIS dump nearest target_wpe [1/omega_pe]
    s = {"target": target_wpe}
    for sp, key in [("e", "ne"), ("al", "nal"), ("si", "nsi")]:
        f, _ = nearest_dump(str(OSIRIS_DIR / f"MS/DENSITY/{sp}/charge-savg/*.h5"), target_wpe)
        s[f"x_{key}"], s[key], s["t_dens"] = osiris_density(f)
    f, _ = nearest_dump(str(OSIRIS_DIR / "MS/FLD/b3-savg/*.h5"), target_wpe)
    s["x_Bz"], s["Bz"], s["t_Bz"] = osiris_bz(f)
    for sp, key in [("e", "Te"), ("al", "Ti_al"), ("si", "Ti_si")]:
        f, _ = nearest_dump(str(OSIRIS_DIR / f"MS/PHA/p1x1/{sp}/*.h5"), target_wpe)
        s[f"x_{key}"], s[key], s[f"t_{key}"] = osiris_temperature(sp, f)
    return s


# %%
# ---- the 1-to-1 comparison figure (FLASH dashed, OSIRIS solid) ----
def compare(flash, osiris, suptitle):
    fig, ax = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

    ax[0].plot(flash["x"], flash["ne"], "k--", label="FLASH")
    ax[0].plot(osiris["x_ne"], osiris["ne"], "C0-", label="OSIRIS")
    ax[0].set_ylabel(r"$n_e / n_0$"); ax[0].legend()

    ax[1].plot(flash["x"], flash["nal"], "C1--")
    ax[1].plot(osiris["x_nal"], osiris["nal"], "C1-", label="Al")
    ax[1].plot(flash["x"], flash["nsi"], "C2--")
    ax[1].plot(osiris["x_nsi"], osiris["nsi"], "C2-", label="Si")
    ax[1].set_ylabel(r"$n_i / n_0$"); ax[1].legend(title="solid = OSIRIS, dashed = FLASH")

    ax[2].plot(flash["x"], flash["Te"], "k--", label="FLASH")
    ax[2].plot(osiris["x_Te"], osiris["Te"], "C3-", label="OSIRIS")
    ax[2].set_ylabel(r"$T_e$ [eV]"); ax[2].legend()

    ax[3].plot(flash["x"], flash["Ti"], "k--", label="FLASH (single fluid)")
    ax[3].plot(osiris["x_Ti_al"], osiris["Ti_al"], "C1-", label="OSIRIS Al")
    ax[3].plot(osiris["x_Ti_si"], osiris["Ti_si"], "C2-", label="OSIRIS Si")
    ax[3].set_ylabel(r"$T_i$ [eV]"); ax[3].legend()

    ax[4].plot(flash["x"], flash["Bz"], "k--", label="FLASH")
    ax[4].plot(osiris["x_Bz"], osiris["Bz"], "C4-", label="OSIRIS")
    ax[4].set_ylabel(r"$B_z$ [OSIRIS units]")
    ax[4].set_xlabel(r"distance along line of sight  [$c/\omega_{pe}$]")
    ax[4].legend()

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


# %% [markdown]
# ## 1. Initial-condition comparison (same physical time)
# OSIRIS t=0 against FLASH plt_cnt_0009 — the only instant the two runs share. Checks
# that the kinetic IC reproduces the FLASH state along the line of sight.

# %%
flash_ic  = flash_lineout(FLASH_IC_DUMP)
osiris_ic = osiris_state(target_wpe=0.0)

compare(
    flash_ic, osiris_ic,
    f"Line-of-sight comparison at matched time\n"
    f"FLASH plt_cnt_{FLASH_IC_DUMP:04d} (t = {flash_ic['t_ns']:.2f} ns)   vs   OSIRIS t = 0",
)
plt.show()


# %% [markdown]
# ## 2. OSIRIS kinetic evolution
# OSIRIS runs ~0.22 ns past the IC — a window with no FLASH data (FLASH's last dump *is*
# the OSIRIS IC). Electron density and B_z evolution, with the FLASH IC dashed for
# reference.

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
ax[0].plot(flash_ic["x"], flash_ic["ne"], "k--", lw=2, label="FLASH IC")
ax[1].plot(flash_ic["x"], flash_ic["Bz"], "k--", lw=2, label="FLASH IC")

for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    target = frac * OSIRIS_TMAX
    fE, _ = nearest_dump(str(OSIRIS_DIR / "MS/DENSITY/e/charge-savg/*.h5"), target)
    xe, ne, te = osiris_density(fE)
    fB, _ = nearest_dump(str(OSIRIS_DIR / "MS/FLD/b3-savg/*.h5"), target)
    xb, bz, _ = osiris_bz(fB)
    ps = te / wpe * 1e12
    label = f"OSIRIS t = {te:5.0f}/wpe  ({ps:.0f} ps)"
    ax[0].plot(xe, ne, label=label)
    ax[1].plot(xb, bz, label=label)

ax[0].set_ylabel(r"$n_e / n_0$")
ax[0].set_title("OSIRIS kinetic evolution (no FLASH data in this window)")
ax[0].legend(fontsize=8)
ax[1].set_ylabel(r"$B_z$ [OSIRIS units]")
ax[1].set_xlabel(r"distance along line of sight  [$c/\omega_{pe}$]")
ax[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
