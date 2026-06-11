"""run_spec.py — the single source of truth for a simulation's run parameters.

A run records its parameters once, in its own directory; analysis reads them via
:class:`RunSpec` instead of re-copying them into the analysis config.  This module
is intentionally dependency-light (stdlib + PyYAML, with astropy imported lazily)
so it can be imported and unit-tested without the full OSIRIS/astropy stack.
"""

import glob
import os
import re
import shlex

import yaml

# run.yaml groups that are one level of nesting purely for readability; their
# sub-keys (start_point, emf_reports, ...) are flattened to the top level so they
# read like the original CLI flags.  ``charge_states`` is metadata and stays nested.
_RUN_YAML_GROUPS = ("geometry", "solver", "diagnostics")


def _parse_cli_flags(text: str) -> dict:
    """Parse ``--key value [value ...]`` pairs out of a command string.

    Backs the run_manifest.yaml ``cli_command`` and legacy ``runme*.sh``
    fallbacks.  Strips comments / line-continuations; multi-value flags become
    lists; the last occurrence of a repeated flag wins.
    """
    text = re.sub(r"#[^\n]*", "", text)   # strip comments
    text = re.sub(r"\\\s*\n", " ", text)  # join continuation lines
    text = text.rstrip().rstrip("\\")
    tokens = shlex.split(text)

    i = 0
    while i < len(tokens) and not tokens[i].startswith("--"):
        i += 1
    out: dict = {}
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            vals = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                vals.append(tokens[j])
                j += 1
            out[key] = vals[0] if len(vals) == 1 else vals
            i = j
        else:
            i += 1
    return out


class RunSpec:
    """Run parameters resolved from a single source of truth in the run directory.

    Resolution order for a ``sim_dir`` (first hit wins):

      1. ``<sim_dir>/run.yaml``          — frozen by FLASH_OSIRIS_define.py
      2. ``<sim_dir>/run_manifest.yaml`` — parse its ``cli_command``
      3. ``<sim_dir>/runme*.sh``         — parse the python-invocation flags (legacy)

    Access parameters with item syntax (``spec["start_point"]``), :meth:`get`, or
    the named properties below.
    """

    def __init__(self, params: dict, charge_states: dict, source: str):
        self.params = params
        self.charge_states = charge_states or {}
        self.source = source

    # -- construction ---------------------------------------------------
    @classmethod
    def from_sim_dir(cls, sim_dir: str) -> "RunSpec":
        run_yaml = os.path.join(sim_dir, "run.yaml")
        if os.path.exists(run_yaml):
            return cls._from_run_yaml(run_yaml)
        manifest = os.path.join(sim_dir, "run_manifest.yaml")
        if os.path.exists(manifest):
            with open(manifest) as f:
                cli = (yaml.safe_load(f) or {}).get("cli_command")
            if cli:
                return cls._from_cli_flags(cli, source=manifest)
        runmes = sorted(glob.glob(os.path.join(sim_dir, "runme*.sh")))
        if runmes:
            with open(runmes[0]) as f:
                return cls._from_cli_flags(f.read(), source=runmes[0])
        raise FileNotFoundError(
            f"No run parameters found in {sim_dir}: expected run.yaml, "
            f"run_manifest.yaml, or runme*.sh. Add a run.yaml (see runs/*.run.yaml)."
        )

    @classmethod
    def _from_run_yaml(cls, path: str) -> "RunSpec":
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        charge_states = cfg.get("charge_states", {})
        flat = {}
        for k, v in cfg.items():
            if k == "charge_states":
                continue
            if k in _RUN_YAML_GROUPS and isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        return cls(flat, charge_states, source=path)

    @classmethod
    def _from_cli_flags(cls, text: str, source: str) -> "RunSpec":
        flags = _parse_cli_flags(text)
        charge_states = {
            sp: int(flags[f"{sp}_charge_state"])
            for sp in ("al", "si") if f"{sp}_charge_state" in flags
        }
        return cls(flags, charge_states, source=source)

    # -- access ---------------------------------------------------------
    def __getitem__(self, key):
        return self.params[key]

    def get(self, key, default=None):
        return self.params.get(key, default)

    @property
    def reference_density(self) -> float:
        if "reference_density" not in self.params:
            raise KeyError(
                f"reference_density not in run spec ({self.source}); add it to run.yaml."
            )
        return float(self.params["reference_density"])

    @property
    def norm_density(self):
        """Normalisation density as an astropy Quantity [cm^-3] (astropy is lazy)."""
        import astropy.units
        return self.reference_density * astropy.units.cm**-3

    @property
    def rqm_factor(self):
        v = self.params.get("rqm_factor")
        return None if v is None else float(v)

    @property
    def inputfile_name(self):
        return self.params.get("inputfile_name")

    @property
    def deck_name(self):
        """OSIRIS deck filename: ``inputfile_name`` with the ``.<dim>d`` suffix the
        generator appends (the run.yaml/manifest store the base name)."""
        name = self.params.get("inputfile_name")
        if name is None:
            return None
        dim = self.params.get("dim")
        if dim is not None:
            suffix = f".{int(dim)}d"
            if not name.endswith(suffix):
                name = name + suffix
        return name

    @property
    def dx(self):
        v = self.params.get("dx")
        return None if v is None else float(v)

    @property
    def ppc(self):
        v = self.params.get("ppc")
        return None if v is None else int(v)

    def charge_state(self, species: str) -> int:
        try:
            return int(self.charge_states[species])
        except KeyError:
            raise KeyError(
                f"No charge state for '{species}' in run spec ({self.source}); "
                f"add charge_states.{species} to run.yaml."
            )
