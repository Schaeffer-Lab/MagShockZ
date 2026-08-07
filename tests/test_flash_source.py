"""Tests for flash_source.resolve — which FLASH data an analysis config points at.

flash_source is dependency-light (stdlib + run_spec) and never opens a dump:
resolve() only ever handles paths and numbers, never opens a dump.
"""

import importlib.util
import os

import pytest


def _load(name):
    path = os.path.join(os.path.dirname(__file__), "..", "src", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


flash_source = _load("flash_source")
resolve = flash_source.resolve

LOS_START = [0, 0.07, 0]
LOS_END = [0, 0.70, 0]


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """The env override is opt-in per test, never inherited from the shell."""
    monkeypatch.delenv(flash_source.ENV_OVERRIDE, raising=False)


@pytest.fixture
def run_dir(tmp_path):
    """An OSIRIS run directory with a frozen run.yaml, for the via-run mode."""
    flash = tmp_path / "FLASH_3D_noshield"
    flash.mkdir()
    sim = tmp_path / "perlmutter_1.3.1d"
    sim.mkdir()
    (sim / "run.yaml").write_text(
        f"data_path: {flash}/MagShockZ_hdf5_plt_cnt_0009\n"
        "reference_density: 5.0e18\n"
        "rqm_factor: 0.01\n"
        "geometry:\n"
        f"  start_point: {LOS_START}\n"
        f"  end_point: {LOS_END}\n"
    )
    return sim, flash


# ---------------------------------------------------------------------------
# Direct mode: the config names the FLASH directory and the LOS itself
# ---------------------------------------------------------------------------

def test_direct_mode_reads_dir_and_los(tmp_path):
    src = resolve({
        "flash_data_dir": str(tmp_path / "FLASH_2026-07"),
        "line_of_sight": {"start_point": LOS_START, "end_point": LOS_END},
    })
    assert src.flash_dir == str(tmp_path / "FLASH_2026-07")
    assert src.line_start == (0.0, 0.07, 0.0)
    assert src.line_end == (0.0, 0.70, 0.0)
    assert src.ic_index == 0            # default: the FLASH run's own start
    assert src.spec is None
    assert src.name == "FLASH_2026-07"


def test_direct_mode_accepts_top_level_endpoints(tmp_path):
    """start_point/end_point may sit at the top level, as they do in a run.yaml."""
    src = resolve({
        "flash_data_dir": str(tmp_path / "d"),
        "start_point": LOS_START,
        "end_point": LOS_END,
    })
    assert src.line_start == (0.0, 0.07, 0.0)


def test_direct_mode_expands_user_and_normalises(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    src = resolve({
        "flash_data_dir": "~/data/flash/",
        "line_of_sight": {"start_point": LOS_START, "end_point": LOS_END},
    })
    assert src.flash_dir == str(tmp_path / "data" / "flash")
    assert src.name == "flash"


def test_direct_mode_optional_normalisation_params(tmp_path):
    bare = resolve({"flash_data_dir": str(tmp_path), "start_point": LOS_START,
                    "end_point": LOS_END})
    assert bare.reference_density is None and bare.rqm_factor is None

    stated = resolve({"flash_data_dir": str(tmp_path), "start_point": LOS_START,
                      "end_point": LOS_END, "reference_density": 5e18,
                      "rqm_factor": 0.01})
    assert stated.reference_density == 5e18
    assert stated.rqm_factor == 0.01


def test_direct_mode_custom_ic_index(tmp_path):
    src = resolve({"flash_data_dir": str(tmp_path), "start_point": LOS_START,
                   "end_point": LOS_END, "ic_index": 9})
    assert src.ic_index == 9


def test_direct_mode_without_los_is_an_error(tmp_path):
    with pytest.raises(KeyError, match="line of sight"):
        resolve({"flash_data_dir": str(tmp_path)}, "config/flash.yaml")


def test_direct_mode_rejects_malformed_endpoint(tmp_path):
    with pytest.raises(ValueError, match="3 coordinates"):
        resolve({"flash_data_dir": str(tmp_path), "start_point": [0, 0.07],
                 "end_point": LOS_END})


# ---------------------------------------------------------------------------
# Via-run mode: inherit data_path / LOS from an OSIRIS run's run.yaml
# ---------------------------------------------------------------------------

def test_via_run_mode_inherits_from_run_spec(run_dir):
    sim, flash = run_dir
    src = resolve({"sim_dir": str(sim)})
    assert src.flash_dir == str(flash)
    assert src.line_start == (0.0, 0.07, 0.0)
    assert src.line_end == (0.0, 0.70, 0.0)
    assert src.ic_index == 9            # the dump that seeded the deck
    assert src.reference_density == 5e18
    assert src.rqm_factor == 0.01
    assert src.spec is not None


def test_via_run_mode_without_normalisation_keys(tmp_path):
    """reference_density is optional: absent means None, not a KeyError."""
    sim = tmp_path / "run"
    sim.mkdir()
    (sim / "run.yaml").write_text(
        f"data_path: {tmp_path}/flash/MagShockZ_hdf5_plt_cnt_0000\n"
        f"start_point: {LOS_START}\n"
        f"end_point: {LOS_END}\n"
    )
    src = resolve({"sim_dir": str(sim)})
    assert src.reference_density is None
    assert src.ic_index == 0


# ---------------------------------------------------------------------------
# Precedence and failure modes
# ---------------------------------------------------------------------------

def test_direct_mode_wins_over_sim_dir(run_dir, tmp_path):
    sim, flash = run_dir
    src = resolve({
        "sim_dir": str(sim),
        "flash_data_dir": str(tmp_path / "other"),
        "start_point": LOS_START, "end_point": LOS_END,
    })
    assert src.flash_dir == str(tmp_path / "other")
    assert src.spec is None


def test_env_override_beats_both(run_dir, tmp_path, monkeypatch):
    sim, _flash = run_dir
    monkeypatch.setenv(flash_source.ENV_OVERRIDE, str(tmp_path / "recomputed"))
    # The LOS still has to come from the config in this mode.
    src = resolve({"sim_dir": str(sim), "start_point": LOS_START, "end_point": LOS_END})
    assert src.flash_dir == str(tmp_path / "recomputed")
    assert flash_source.ENV_OVERRIDE in src.source


def test_no_source_at_all_is_an_error():
    with pytest.raises(KeyError, match="names no FLASH data"):
        resolve({"times": [0, 1]}, "config/oops.yaml")
