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

A config may instead state **several** lines of sight through the same dataset — a
fan of rays, say — as a mapping keyed by label.  :func:`resolve` then returns one of
them (chosen with ``los=``) and :func:`resolve_all` returns them all::

      lines_of_sight:
        los00: {start_point: [0.0, 0.07, 0.0], end_point: [0.0, 1.40, 0.0]}
        los30: {start_point: [0.035, 0.061, 0.0], end_point: [0.70, 1.212, 0.0]}

A **mapping**, not a list, because the per-ray shock parameters the tuners write back
(``flash:``, ``flash_dump_params:``, read through :func:`los_params`) are keyed by the
same labels, and ``yaml_edit`` addresses config entries by dotted key path — it has no
way to name a list element.  Each ray also carries its ``label`` into the output
directory, so one ray's ``flash_overview_*.npz`` is never read back for another.

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

from magshockz.common.run_spec import RunSpec

# Config keys that select the direct mode / the via-run mode.
DIRECT_KEY = "flash_data_dir"
RUN_KEY = "sim_dir"
ENV_OVERRIDE = "MAGSHOCKZ_FLASH_DIR"

# Optional multi-LOS block: {label: {start_point, end_point}}.
LOS_KEY = "lines_of_sight"

# Per-dump / per-trajectory keys the tuners write.  Seeing one of these at the top of
# a `flash:` / `flash_dump_params:` block means the block is flat, which cannot be
# right for a config that defines several rays — see :func:`los_params`.
_FLAT_MARKERS = ("v_shock_est_cms", "x_shock_0_cm", "t_shock_0_s",
                 "x_shock_cm", "x_downstream_start_cm")


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
    label :
        Which entry of a ``lines_of_sight`` mapping this is; ``""`` for a config that
        states a single LOS.  Doubles as the output sub-directory, so each ray's
        results stay in their own tree.
    """

    flash_dir: str
    line_start: Tuple[float, float, float]
    line_end: Tuple[float, float, float]
    ic_index: int = 0
    reference_density: Optional[float] = None
    rqm_factor: Optional[float] = None
    spec: Optional[RunSpec] = None
    source: str = ""
    label: str = ""

    @property
    def name(self) -> str:
        """Run name for output paths: the FLASH directory's basename."""
        return os.path.basename(self.flash_dir.rstrip("/"))


def _endpoints(primary: dict, where: str, fallback: dict = None) -> Tuple[tuple, tuple]:
    """Read ``start_point`` / ``end_point`` [cm] from ``primary``, else ``fallback``.

    ``primary`` is one ``lines_of_sight`` entry, or the ``line_of_sight:`` block;
    ``fallback`` lets a single-LOS config state the two points at the top level.
    """
    fallback = fallback or {}
    pts = {}
    for key in ("start_point", "end_point"):
        val = primary.get(key, fallback.get(key))
        if val is None:
            raise KeyError(
                f"{where} sets '{DIRECT_KEY}' but no {key}. A config that points "
                f"directly at FLASH data must state the line of sight itself:\n"
                f"  line_of_sight:\n"
                f"    start_point: [x, y, z]   # cm\n"
                f"    end_point:   [x, y, z]   # cm\n"
                f"or one entry per ray under '{LOS_KEY}:'."
            )
        if len(val) != 3:
            raise ValueError(f"{where}: {key} must be 3 coordinates [cm], got {val!r}")
        pts[key] = tuple(float(v) for v in val)
    return pts["start_point"], pts["end_point"]


def _select_los(cfg: dict, config_path: str, los: Optional[str]) -> Tuple[str, Optional[dict]]:
    """Pick one entry of a ``lines_of_sight`` mapping.

    Returns ``(label, entry)``, or ``("", None)`` when the config states a single LOS.
    With several rays available and no ``los`` asked for, the first is used and named
    on stdout — analysing the wrong ray must never be silent.
    """
    rays = cfg.get(LOS_KEY)
    if not rays:
        if los:
            raise KeyError(
                f"{config_path} has no '{LOS_KEY}:' block, so --los {los!r} selects "
                f"nothing. It states a single line of sight.")
        return "", None

    labels = list(rays)
    if los is None:
        label = labels[0]
        print(f"[flash_source] {len(labels)} lines of sight; using {label!r} "
              f"(--los to choose from: {', '.join(labels)})")
    elif los in rays:
        label = los
    else:
        raise KeyError(
            f"{config_path} has no line of sight {los!r}. Available: {', '.join(labels)}")
    return label, rays[label]


def los_params(cfg: dict, section: str, label: str) -> dict:
    """Read a per-LOS config block (``flash:``, ``flash_dump_params:``).

    Each ray has its own shock speed and per-dump front positions, so on a multi-LOS
    config these blocks are keyed by LOS label; on a single-LOS config the block sits
    at the top level.  Returns ``{}`` when the block, or this ray's entry, is absent.
    """
    block = cfg.get(section) or {}
    if not label:
        return block
    if any(key in block for key in _FLAT_MARKERS):
        raise KeyError(
            f"'{section}:' holds front parameters directly, but this config defines "
            f"several lines of sight and each ray needs its own. Key the block by LOS "
            f"label:\n  {section}:\n    {label}:\n      ...")
    return block.get(label) or {}


def add_los_arg(parser):
    """Add the shared ``--los`` flag (mirrors ``plot_style.add_publication_arg``)."""
    parser.add_argument(
        "--los", default=None,
        help=f"Which line of sight to analyse, for a config with a '{LOS_KEY}:' "
             f"mapping (default: its first entry). Output goes to a sub-directory "
             f"named after the label, so rays never overwrite each other.")
    return parser


def _optional_float(cfg: dict, key: str) -> Optional[float]:
    v = cfg.get(key)
    return None if v is None else float(v)


def resolve(cfg: dict, config_path: str = "config",
            los: Optional[str] = None) -> FlashSource:
    """Resolve a FLASH analysis config to a :class:`FlashSource`.

    Parameters
    ----------
    cfg :
        The loaded config dict (``analysis_utils.load_config``).
    config_path :
        Path to that config, used only in error messages.
    los :
        Which entry of a ``lines_of_sight`` mapping to take.  ``None`` means the
        config's single LOS, or the mapping's first entry.
    """
    env_dir = os.environ.get(ENV_OVERRIDE)
    label, entry = _select_los(cfg, config_path, los)

    # -- direct mode ----------------------------------------------------
    if env_dir or cfg.get(DIRECT_KEY):
        flash_dir = os.path.abspath(os.path.expanduser(env_dir or cfg[DIRECT_KEY]))
        if entry is None:
            line_start, line_end = _endpoints(
                cfg.get("line_of_sight") or {}, config_path, fallback=cfg)
        else:
            line_start, line_end = _endpoints(entry, f"{config_path} [{LOS_KEY}.{label}]")
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
            label=label,
        )

    # -- via-run mode ---------------------------------------------------
    if cfg.get(RUN_KEY):
        spec = RunSpec.from_sim_dir(cfg[RUN_KEY])
        data_path = spec["data_path"]
        ref_density = spec.get("reference_density")
        # An explicit lines_of_sight block overrides the LOS the deck was extracted
        # along, so a fan can be cut through the FLASH data an OSIRIS run was seeded
        # from; everything else still comes from the run spec.
        if entry is None:
            line_start = tuple(float(v) for v in spec["start_point"])
            line_end = tuple(float(v) for v in spec["end_point"])
        else:
            line_start, line_end = _endpoints(entry, f"{config_path} [{LOS_KEY}.{label}]")
        return FlashSource(
            flash_dir=str(os.path.dirname(data_path)),
            line_start=line_start,
            line_end=line_end,
            ic_index=int(os.path.basename(data_path)[-4:]),
            reference_density=None if ref_density is None else float(ref_density),
            rqm_factor=spec.rqm_factor,
            spec=spec,
            source=spec.source,
            label=label,
        )

    raise KeyError(
        f"{config_path} names no FLASH data. Set '{DIRECT_KEY}' (plus a "
        f"line_of_sight) to point at a FLASH output directory, or '{RUN_KEY}' to "
        f"inherit data_path/start_point/end_point from an OSIRIS run's run.yaml."
    )


def resolve_all(cfg: dict, config_path: str = "config") -> list:
    """Every line of sight the config defines, in the order it lists them.

    A single-LOS config gives a one-element list, so callers that draw or tabulate
    all rays need no special case.
    """
    rays = cfg.get(LOS_KEY)
    if not rays:
        return [resolve(cfg, config_path)]
    return [resolve(cfg, config_path, los=label) for label in rays]
