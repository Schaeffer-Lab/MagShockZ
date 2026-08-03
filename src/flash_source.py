"""flash_source.py — resolve *which* FLASH data a FLASH-side analysis config refers to.

A FLASH analysis config names its dataset in one of two ways; :func:`resolve`
returns the same :class:`FlashSource` either way, so the scripts don't care which:

  **direct** — the config points straight at a FLASH output directory and states
  the line of sight itself.  Use this for FLASH runs that have no OSIRIS deck
  (yet)::

      flash_data_dir: /mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_2026-07
      line_of_sight:
        start_point: [0.0, 0.07, 0.0]   # cm
        end_point:   [0.6, 0.07, 0.0]   # cm
      ic_index: 0                       # optional; dump that anchors t=0 (default 0)

  **via-run** — the config points at an OSIRIS run whose ``run.yaml`` records the
  FLASH ``data_path`` it was seeded from and the LOS it was extracted along; those
  are read back through :class:`run_spec.RunSpec` and never duplicated::

      sim_dir: /pscratch/sd/d/dschnei/perlmutter_1.3.1d

``flash_data_dir`` wins if both are present.  ``$MAGSHOCKZ_FLASH_DIR`` overrides the
directory in either mode (the analogue of ``$MAGSHOCKZ_SIM_DIR`` in
``analysis_utils.load_config``), which is how you re-point a whole config at a
recomputed copy of the same run without editing it.

Deliberately dependency-light (stdlib + :mod:`run_spec`): it resolves *paths and
numbers*, never opens a dump, so it is unit-testable without yt.  Listing the plot
files stays with the caller (``flash_utils.find_plot_files``), which owns the
filename convention.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from run_spec import RunSpec

# Config keys that select the direct mode / the via-run mode.
DIRECT_KEY = "flash_data_dir"
RUN_KEY = "sim_dir"
ENV_OVERRIDE = "MAGSHOCKZ_FLASH_DIR"


@dataclass
class FlashSource:
    """Where a FLASH analysis config's data lives, and how to slice it.

    Attributes
    ----------
    flash_dir :
        Directory holding the ``MagShockZ_hdf5_plt_cnt_*`` plot files.
    line_start, line_end :
        Line-of-sight endpoints [cm], as in the generator's
        ``--start_point`` / ``--end_point``.
    ic_index :
        Plot-file index that anchors t=0.  In via-run mode this is the dump that
        seeded the OSIRIS deck (so FLASH and OSIRIS times line up); in direct mode
        it defaults to 0 (the FLASH run's own start).
    reference_density, rqm_factor :
        OSIRIS normalisation parameters, when they are knowable — ``None`` in
        direct mode unless the config states them.  Reporting only.
    spec :
        The backing :class:`RunSpec` in via-run mode, else ``None``.
    source :
        Human-readable provenance, for script banners.
    """

    flash_dir: str
    line_start: Tuple[float, float, float]
    line_end: Tuple[float, float, float]
    ic_index: int = 0
    reference_density: Optional[float] = None
    rqm_factor: Optional[float] = None
    spec: Optional[RunSpec] = None
    source: str = ""

    @property
    def name(self) -> str:
        """Run name for output paths: the FLASH directory's basename."""
        return os.path.basename(self.flash_dir.rstrip("/"))


def _endpoints(cfg: dict, where: str) -> Tuple[tuple, tuple]:
    """Read the LOS from ``line_of_sight: {start_point, end_point}`` or top level."""
    los = cfg.get("line_of_sight") or {}
    pts = {}
    for key in ("start_point", "end_point"):
        val = los.get(key, cfg.get(key))
        if val is None:
            raise KeyError(
                f"{where} sets '{DIRECT_KEY}' but no {key}. A config that points "
                f"directly at FLASH data must state the line of sight itself:\n"
                f"  line_of_sight:\n"
                f"    start_point: [x, y, z]   # cm\n"
                f"    end_point:   [x, y, z]   # cm"
            )
        if len(val) != 3:
            raise ValueError(f"{where}: {key} must be 3 coordinates [cm], got {val!r}")
        pts[key] = tuple(float(v) for v in val)
    return pts["start_point"], pts["end_point"]


def _optional_float(cfg: dict, key: str) -> Optional[float]:
    v = cfg.get(key)
    return None if v is None else float(v)


def resolve(cfg: dict, config_path: str = "config") -> FlashSource:
    """Resolve a FLASH analysis config to a :class:`FlashSource`.

    Parameters
    ----------
    cfg :
        The loaded config dict (``analysis_utils.load_config``).
    config_path :
        Path to that config, used only in error messages.
    """
    env_dir = os.environ.get(ENV_OVERRIDE)

    # -- direct mode ----------------------------------------------------
    if env_dir or cfg.get(DIRECT_KEY):
        flash_dir = os.path.abspath(os.path.expanduser(env_dir or cfg[DIRECT_KEY]))
        line_start, line_end = _endpoints(cfg, config_path)
        src = f"{ENV_OVERRIDE}={env_dir}" if env_dir else f"{config_path} ({DIRECT_KEY})"
        return FlashSource(
            flash_dir=flash_dir,
            line_start=line_start,
            line_end=line_end,
            ic_index=int(cfg.get("ic_index", 0)),
            reference_density=_optional_float(cfg, "reference_density"),
            rqm_factor=_optional_float(cfg, "rqm_factor"),
            spec=None,
            source=src,
        )

    # -- via-run mode ---------------------------------------------------
    if cfg.get(RUN_KEY):
        spec = RunSpec.from_sim_dir(cfg[RUN_KEY])
        data_path = spec["data_path"]
        ref_density = spec.get("reference_density")
        return FlashSource(
            flash_dir=str(os.path.dirname(data_path)),
            line_start=tuple(float(v) for v in spec["start_point"]),
            line_end=tuple(float(v) for v in spec["end_point"]),
            ic_index=int(os.path.basename(data_path)[-4:]),
            reference_density=None if ref_density is None else float(ref_density),
            rqm_factor=spec.rqm_factor,
            spec=spec,
            source=spec.source,
        )

    raise KeyError(
        f"{config_path} names no FLASH data. Set '{DIRECT_KEY}' (plus a "
        f"line_of_sight) to point at a FLASH output directory, or '{RUN_KEY}' to "
        f"inherit data_path/start_point/end_point from an OSIRIS run's run.yaml."
    )
