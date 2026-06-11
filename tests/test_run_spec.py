"""Tests for run_spec.RunSpec — the single source of truth for run parameters.

run_spec is dependency-light (stdlib + PyYAML), so these run in CI without the
full OSIRIS/astropy stack. The astropy-backed norm_density property is covered by
an importorskip-guarded test.
"""

import importlib.util
import os

import pytest

# Load the real src/run_spec.py directly (conftest puts tests/ first on the path,
# but there is no run_spec stub, so a normal import would also work; loading by
# path keeps this independent of sys.path ordering).
_SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "run_spec.py")
_spec = importlib.util.spec_from_file_location("run_spec", _SPEC_PATH)
run_spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_spec)
RunSpec = run_spec.RunSpec


# ---------------------------------------------------------------------------
# CLI-flag parsing (run_manifest.yaml cli_command / legacy runme*.sh)
# ---------------------------------------------------------------------------

def test_parse_cli_flags_single_and_multi():
    flags = run_spec._parse_cli_flags(
        "FLASH_OSIRIS_define.py --dim 1 --start_point 0 0.07 0 --reference_density 5e18")
    assert flags["dim"] == "1"
    assert flags["start_point"] == ["0", "0.07", "0"]
    assert flags["reference_density"] == "5e18"


def test_parse_cli_flags_strips_comments_and_continuations():
    text = "python x.py \\\n--dim 1 \\\n--rqm_factor 100  # trailing comment\n"
    flags = run_spec._parse_cli_flags(text)
    assert flags["dim"] == "1"
    assert flags["rqm_factor"] == "100"


def test_parse_cli_flags_last_occurrence_wins():
    flags = run_spec._parse_cli_flags("p.py --dx 0.3 --dx 0.1")
    assert flags["dx"] == "0.1"


# ---------------------------------------------------------------------------
# run.yaml resolution (preferred source)
# ---------------------------------------------------------------------------

RUN_YAML = """\
data_path: /data/plt_cnt_0009
dim: 1
inputfile_name: magshockz_rqm100_dx0.1_ppc500_g20
reference_density: 5.0e18
rqm_factor: 100
dx: 0.1
ppc: 500
charge_states: {al: 13, si: 14}
geometry:
  start_point: [0, 0.07, 0]
  end_point: [0, 0.70, 0]
diagnostics:
  emf_reports: [b1, b2, b3]
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_from_run_yaml_flattens_groups_and_keeps_charge_states(tmp_path):
    _write(tmp_path, "run.yaml", RUN_YAML)
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.source.endswith("run.yaml")
    # groups flattened to top level
    assert s["start_point"] == [0, 0.07, 0]
    assert s["emf_reports"] == ["b1", "b2", "b3"]
    # charge_states stays nested, accessed via charge_state()
    assert s.charge_state("al") == 13
    assert s.charge_state("si") == 14
    assert "charge_states" not in s.params  # not flattened into params


def test_run_yaml_typed_accessors(tmp_path):
    _write(tmp_path, "run.yaml", RUN_YAML)
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.reference_density == 5.0e18
    assert isinstance(s.reference_density, float)
    assert s.rqm_factor == 100.0
    assert s.dx == 0.1
    assert s.ppc == 500


def test_deck_name_appends_dim_suffix(tmp_path):
    _write(tmp_path, "run.yaml", RUN_YAML)
    s = RunSpec.from_sim_dir(str(tmp_path))
    # base name in run.yaml; .1d suffix appended from dim
    assert s.deck_name == "magshockz_rqm100_dx0.1_ppc500_g20.1d"


def test_deck_name_no_double_suffix(tmp_path):
    _write(tmp_path, "run.yaml", "inputfile_name: magshockz_gpu.1d\ndim: 1\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.deck_name == "magshockz_gpu.1d"  # already suffixed, not doubled


# ---------------------------------------------------------------------------
# Resolution order: run.yaml > run_manifest.yaml > runme*.sh
# ---------------------------------------------------------------------------

def test_manifest_fallback_when_no_run_yaml(tmp_path):
    _write(tmp_path, "run_manifest.yaml",
           "cli_command: define.py --reference_density 5e18 --inputfile_name foo "
           "--dim 1 --al_charge_state 13 --si_charge_state 14\nderived: {}\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.source.endswith("run_manifest.yaml")
    assert s.reference_density == 5e18
    assert s.deck_name == "foo.1d"
    assert s.charge_state("al") == 13


def test_runme_fallback_when_no_yaml(tmp_path):
    _write(tmp_path, "runme_x.sh",
           "python define.py --reference_density 5e18 --inputfile_name bar --dim 2\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.source.endswith("runme_x.sh")
    assert s.deck_name == "bar.2d"


def test_run_yaml_preferred_over_manifest(tmp_path):
    _write(tmp_path, "run.yaml", "reference_density: 1.0e18\ninputfile_name: a\ndim: 1\n")
    _write(tmp_path, "run_manifest.yaml", "cli_command: x --reference_density 9e18\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.reference_density == 1.0e18  # run.yaml wins


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunSpec.from_sim_dir(str(tmp_path))


def test_missing_charge_state_raises(tmp_path):
    _write(tmp_path, "run.yaml", "reference_density: 5e18\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    with pytest.raises(KeyError):
        s.charge_state("al")


def test_missing_reference_density_raises(tmp_path):
    _write(tmp_path, "run.yaml", "inputfile_name: a\ndim: 1\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    with pytest.raises(KeyError):
        _ = s.reference_density


# ---------------------------------------------------------------------------
# astropy-backed property (skipped where astropy is absent, e.g. minimal CI)
# ---------------------------------------------------------------------------

def test_norm_density_quantity(tmp_path):
    pytest.importorskip("astropy")
    import astropy.units as u
    _write(tmp_path, "run.yaml", "reference_density: 5.0e18\n")
    s = RunSpec.from_sim_dir(str(tmp_path))
    assert s.norm_density == 5.0e18 * u.cm**-3
