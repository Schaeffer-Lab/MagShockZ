# %% [markdown]
# FLASH vs. OSIRIS — 1-to-1 comparison along the line of sight
#
# Compares the FLASH 3D MHD run against the OSIRIS 1D PIC run for densities,
# electron/ion temperatures, and magnetic-field compression (B_z), along the same
# line of sight the OSIRIS run was built from.
#
# Run interactively cell-by-cell (VS Code / Jupyter `# %%` cells) with the
# `analysis` conda env.
#
# Key conventions
# ---------------
# * Space: electron skin depths c/omega_pe.
# * Time: OSIRIS t=0 == FLASH IC dump (read from runme_perlmutter.sh --data_path).
#   FLASH times are absolute [s]; OSIRIS times are 1/omega_pe.
#   Conversion: t[wpe] = t[s] * omega_pe.
# * All FLASH init params (line of sight, B0, rqm_factor, charge states) are read
#   directly from runme_perlmutter.sh in the OSIRIS sim directory — not duplicated here.

# %%
import os
import re
import sys
import glob
import shlex
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py
import osh5io
import astropy.units as u
import astropy.constants as const
import yaml
import yt

warnings.filterwarnings("ignore")
yt.set_log_level(50)

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "src"))
sys.path.append(str(REPO / "init_nopython"))

from analysis_utils import MagShockZRun
from moments import moment
from fitting_functions import Ray


# %%
# ---- Parse runme_perlmutter.sh for FLASH initialisation parameters ----

def parse_runme(path: str) -> dict:
    """Extract --key value pairs from a python-invocation shell script.

    Handles backslash line continuations and strips comments. Multi-value
    flags (e.g. --start_point 0 0.07 0) are returned as lists.
    """
    with open(path) as f:
        text = f.read()
    text = re.sub(r'#[^\n]*', '', text)   # strip comments
    text = text.replace('\\\n', ' ')       # join continuation lines
    tokens = shlex.split(text)

    # Advance past the 'python' call and script path to the first flag
    i = 0
    while i < len(tokens) and not tokens[i].startswith('--'):
        i += 1

    args = {}
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('--'):
            key = tok[2:]
            vals = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith('--'):
                vals.append(tokens[j])
                j += 1
            args[key] = vals[0] if len(vals) == 1 else vals
            i = j
        else:
            i += 1
    return args


# %%
# ---- Load config ----

CFG_PATH = REPO / "config" / "perlmutter_1.3.1d.yaml"
with open(CFG_PATH) as _f:
    cfg = yaml.safe_load(_f)

OSIRIS_DIR = Path(os.environ.get("MAGSHOCKZ_SIM_DIR", cfg["sim_dir"]))
runme      = parse_runme(OSIRIS_DIR / "runme_perlmutter.sh")

# FLASH location and IC dump come from --data_path in the runme
_data_path    = Path(runme["data_path"])
FLASH_DIR     = _data_path.parent
FILE_PREFIX   = _data_path.name[:-4]   # strip the 4-digit dump index
FLASH_IC_DUMP = int(_data_path.name[-4:])

LINE_START  = tuple(float(v) for v in runme["start_point"])
LINE_END    = tuple(float(v) for v in runme["end_point"])
RQM_FACTOR  = float(runme["rqm_factor"])
B0_GAUSS    = float(runme["B0_Gauss"])   # last occurrence wins (listed twice in runme)
AL_CHARGE   = int(runme["al_charge_state"])
SI_CHARGE   = int(cfg["si_charge_state"])

REF_DENSITY = float(cfg["norm_density_cm3"]) * u.cm**-3
B0          = B0_GAUSS * u.G


# %%
# ---- Derive unit-conversion quantities from MagShockZRun ----

run = MagShockZRun(
    input_deck=str(OSIRIS_DIR / cfg.get("input_deck", "magshockz_gpu.1d")),
    norm_density=REF_DENSITY,
    B0=B0,
    Z=AL_CHARGE,
    m_i=AL_CHARGE * const.m_p,
)

omega_pe = run.omega_pe
wpe_hz   = float(omega_pe.to_value(u.rad / u.s))
me_c2_eV = (const.m_e * const.c**2).to("eV").value

RQM = {"e": 1.0}
for sp in ("al", "si"):
    if sp in run.deck.species:
        RQM[sp] = run.deck.species[sp].rqm

Z_CHARGE = {"e": 1, "al": AL_CHARGE, "si": SI_CHARGE}

# TEMP_MASS_FACTOR converts velocity variance <(u-<u>)^2> → temperature in eV.
# For ions an extra Z factor is needed because the IC velocity normalisation in
# fitting_functions.py used the real ion mass while OSIRIS rqm folds in charge Z.
TEMP_MASS_FACTOR = {sp: RQM[sp] * Z_CHARGE[sp] for sp in RQM}

print(f"Config          : {CFG_PATH}")
print(f"Runme           : {OSIRIS_DIR / 'runme_perlmutter.sh'}")
print(f"OSIRIS dir      : {OSIRIS_DIR}")
print(f"FLASH dir       : {FLASH_DIR}")
print(f"FLASH IC dump   : {FLASH_IC_DUMP}  ({FILE_PREFIX}{FLASH_IC_DUMP:04d})")
print(f"Line of sight   : {LINE_START}  ->  {LINE_END}  [cm]")
print(f"n0              : {cfg['norm_density_cm3']:.2e} cm⁻³")
print(f"B0              : {B0}")
print(f"rqm_factor      : {RQM_FACTOR}")
print(f"omega_pe        : {omega_pe:.3e}")
print(f"c/omega_pe      : {(const.c.cgs.value / wpe_hz * u.cm).to('um'):.3f}")
print(f"RQM             : {RQM}")
print(f"Z_CHARGE        : {Z_CHARGE}")
print(f"TEMP_MASS_FACTOR: {TEMP_MASS_FACTOR}")


# %%
# ---- Time conversion helpers ----

def _flash_time_s(dump_index: int) -> float:
    """Read the absolute simulation time [s] from a FLASH dump (cheap h5py read)."""
    path = FLASH_DIR / f"{FILE_PREFIX}{dump_index:04d}"
    with h5py.File(path, "r") as f:
        return float({k.decode().strip(): v for k, v in f["real scalars"][:]}.get("time", 0.0))


def flash_to_osiris_wpe(flash_dump_index: int) -> float:
    """Convert a FLASH dump index to the equivalent OSIRIS time in 1/omega_pe.

    OSIRIS t=0 is defined as the absolute time of the IC dump from the runme.
    The elapsed FLASH time since the IC is multiplied by omega_pe.
    """
    t_elapsed_s = _flash_time_s(flash_dump_index) - _flash_time_s(FLASH_IC_DUMP)
    return t_elapsed_s * wpe_hz


def _dump_time_wpe(path: str) -> float:
    with h5py.File(path, "r") as f:
        return float(f.attrs["TIME"][0])


def nearest_osiris_dump(glob_pattern: str, target_wpe: float):
    """Return (path, actual_time_wpe) of the OSIRIS dump closest to target_wpe."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No OSIRIS dumps matched: {glob_pattern}")
    times = np.array([_dump_time_wpe(f) for f in files])
    i = int(np.argmin(np.abs(times - target_wpe)))
    return files[i], times[i]


def osiris_tmax() -> float:
    """Discover the maximum available OSIRIS time from electron density dumps."""
    files = sorted(glob.glob(str(OSIRIS_DIR / "MS/DENSITY/e/charge-savg/*.h5")))
    if not files:
        raise FileNotFoundError("No OSIRIS electron density dumps found.")
    return max(_dump_time_wpe(f) for f in files)


# %%
# ---- FLASH side: lineout in OSIRIS-normalised units ----

def flash_lineout(dump_index: int) -> dict:
    path = str(FLASH_DIR / f"{FILE_PREFIX}{dump_index:04d}")
    ray  = Ray(path, LINE_START, LINE_END, rqm_factor=RQM_FACTOR,
               reference_density=float(REF_DENSITY.value) * yt.units.cm**-3)
    raw, order = ray._get_sorted_ray()
    kB = const.k_B.to("eV/K").value
    return {
        "x":    ray.osiris_length,
        "t_s":  float(ray.ds.current_time.to("s")),
        "t_ns": float(ray.ds.current_time.to("s")) * 1e9,
        "ne":   ray._get_field_values("edens"),
        "nal":  ray._get_field_values("aldens"),
        "nsi":  ray._get_field_values("sidens"),
        "Te":   np.asarray(raw["flash", "tele"][order]) * kB,
        "Ti":   np.asarray(raw["flash", "tion"][order]) * kB,
        "Bz":   ray._get_rotated_vector_component("magz"),
    }


# %%
# ---- OSIRIS side: read normalised diagnostics ----

def _x_axis(d):
    return np.linspace(d.axes[0].min, d.axes[0].max, d.axes[0].size)


def osiris_density(path: str):
    d = osh5io.read_h5(path)
    return _x_axis(d), np.abs(np.asarray(d)), d.run_attrs["TIME"][0]


def osiris_bz(path: str):
    d = osh5io.read_h5(path)
    return _x_axis(d), np.asarray(d), d.run_attrs["TIME"][0]


def osiris_temperature(species: str, path: str):
    """Temperature [eV] from the 2nd velocity moment of a p1x1 phase space."""
    d = osh5io.read_h5(path)
    variance = moment(d, order=2, axis="p1")
    x = np.linspace(d.axes[1].min, d.axes[1].max, d.axes[1].size)
    T = TEMP_MASS_FACTOR[species] * me_c2_eV * variance
    T[variance <= 0] = np.nan
    return x, T, d.run_attrs["TIME"][0]


def osiris_state(target_wpe: float) -> dict:
    """All OSIRIS comparison fields from the dump nearest target_wpe."""
    s = {"target_wpe": target_wpe}
    for sp, key in [("e", "ne"), ("al", "nal"), ("si", "nsi")]:
        f, _ = nearest_osiris_dump(
            str(OSIRIS_DIR / f"MS/DENSITY/{sp}/charge-savg/*.h5"), target_wpe)
        s[f"x_{key}"], s[key], s["t_dens"] = osiris_density(f)
    f, _ = nearest_osiris_dump(str(OSIRIS_DIR / "MS/FLD/b3-savg/*.h5"), target_wpe)
    s["x_Bz"], s["Bz"], s["t_Bz"] = osiris_bz(f)
    for sp, key in [("e", "Te"), ("al", "Ti_al"), ("si", "Ti_si")]:
        f, _ = nearest_osiris_dump(
            str(OSIRIS_DIR / f"MS/PHA/p1x1/{sp}/*.h5"), target_wpe)
        s[f"x_{key}"], s[key], s[f"t_{key}"] = osiris_temperature(sp, f)
    return s


# %%
# ---- Comparison figure (FLASH dashed, OSIRIS solid) ----

def compare(flash: dict, osiris: dict, suptitle: str):
    fig, ax = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

    ax[0].plot(flash["x"], flash["ne"],  "k--", label="FLASH")
    ax[0].plot(osiris["x_ne"], osiris["ne"], "C0-", label="OSIRIS")
    ax[0].set_ylabel(r"$n_e / n_0$"); ax[0].legend()

    ax[1].plot(flash["x"], flash["nal"], "C1--")
    ax[1].plot(osiris["x_nal"], osiris["nal"], "C1-", label="Al")
    ax[1].plot(flash["x"], flash["nsi"], "C2--")
    ax[1].plot(osiris["x_nsi"], osiris["nsi"], "C2-", label="Si")
    ax[1].set_ylabel(r"$n_i / n_0$"); ax[1].legend(title="solid=OSIRIS  dashed=FLASH")

    ax[2].plot(flash["x"], flash["Te"],  "k--", label="FLASH")
    ax[2].plot(osiris["x_Te"], osiris["Te"], "C3-", label="OSIRIS")
    ax[2].set_ylabel(r"$T_e$ [eV]"); ax[2].legend()

    ax[3].plot(flash["x"], flash["Ti"],  "k--", label="FLASH (single fluid)")
    ax[3].plot(osiris["x_Ti_al"], osiris["Ti_al"], "C1-", label="OSIRIS Al")
    ax[3].plot(osiris["x_Ti_si"], osiris["Ti_si"], "C2-", label="OSIRIS Si")
    ax[3].set_ylabel(r"$T_i$ [eV]"); ax[3].legend()

    ax[4].plot(flash["x"], flash["Bz"],  "k--", label="FLASH")
    ax[4].plot(osiris["x_Bz"], osiris["Bz"], "C4-", label="OSIRIS")
    ax[4].set_ylabel(r"$B_z$ [OSIRIS units]")
    ax[4].set_xlabel(r"distance along line of sight  [$c/\omega_{pe}$]")
    ax[4].legend()

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


def matched_compare(flash_dump_index: int):
    """Load a FLASH dump and the closest OSIRIS dump, then plot the comparison.

    Converts the FLASH absolute time to OSIRIS units using omega_pe and finds
    the nearest available OSIRIS dump automatically.
    """
    target_wpe = flash_to_osiris_wpe(flash_dump_index)
    t_flash_ns = _flash_time_s(flash_dump_index) * 1e9
    t_ic_ns    = _flash_time_s(FLASH_IC_DUMP) * 1e9

    print(f"FLASH dump {flash_dump_index:04d}  :  t = {t_flash_ns:.3f} ns  "
          f"(+{t_flash_ns - t_ic_ns:.3f} ns since IC)")
    print(f"Target OSIRIS t  :  {target_wpe:.1f} / wpe  "
          f"= {target_wpe / wpe_hz * 1e12:.1f} ps")

    flash  = flash_lineout(flash_dump_index)
    osiris = osiris_state(target_wpe)

    t_osiris_wpe = osiris["t_dens"]
    t_osiris_ns  = t_osiris_wpe / wpe_hz * 1e9
    print(f"Nearest OSIRIS t :  {t_osiris_wpe:.1f} / wpe  "
          f"(Δ = {abs(target_wpe - t_osiris_wpe):.1f} / wpe)")

    title = (
        f"Line-of-sight comparison\n"
        f"FLASH plt_cnt_{flash_dump_index:04d}  (t = {t_flash_ns:.3f} ns)   vs   "
        f"OSIRIS  t = {t_osiris_wpe:.0f} / wpe  "
        f"({t_ic_ns + t_osiris_ns:.3f} ns abs)"
    )
    return compare(flash, osiris, title)


# %% [markdown]
# ## 1. Initial-condition comparison  (FLASH IC dump vs OSIRIS t=0)

# %%
flash_ic  = flash_lineout(FLASH_IC_DUMP)
osiris_ic = osiris_state(target_wpe=0.0)

compare(
    flash_ic, osiris_ic,
    f"IC comparison\n"
    f"FLASH {FILE_PREFIX}{FLASH_IC_DUMP:04d}  (t = {flash_ic['t_ns']:.3f} ns)   vs   OSIRIS  t = 0",
)
plt.show()


# %% [markdown]
# ## 2. Time-matched comparison for any FLASH dump
# Change `flash_dump_index` to any available FLASH dump; the script converts
# to OSIRIS time units and finds the nearest available dump automatically.

# %%
matched_compare(flash_dump_index=FLASH_IC_DUMP + 1)
plt.show()


# %% [markdown]
# ## 3. OSIRIS kinetic evolution  (no FLASH data in this window)

# %%
OSIRIS_TMAX = osiris_tmax()
print(f"OSIRIS runtime: 0 -> {OSIRIS_TMAX:.0f} / wpe  =  {OSIRIS_TMAX / wpe_hz * 1e9:.3f} ns")

fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
ax[0].plot(flash_ic["x"], flash_ic["ne"], "k--", lw=2, label="FLASH IC")
ax[1].plot(flash_ic["x"], flash_ic["Bz"], "k--", lw=2, label="FLASH IC")

for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    target = frac * OSIRIS_TMAX
    fE, _ = nearest_osiris_dump(str(OSIRIS_DIR / "MS/DENSITY/e/charge-savg/*.h5"), target)
    xe, ne, te = osiris_density(fE)
    fB, _ = nearest_osiris_dump(str(OSIRIS_DIR / "MS/FLD/b3-savg/*.h5"), target)
    xb, bz, _ = osiris_bz(fB)
    label = f"OSIRIS t = {te:5.0f}/wpe  ({te / wpe_hz * 1e12:.0f} ps)"
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
