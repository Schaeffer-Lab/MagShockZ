"""Tests for src/yaml_edit.py — comment-preserving scalar edits (PyYAML only, CI-safe)."""

import pytest
import yaml

from magshockz.common import yaml_edit
from magshockz.common.yaml_edit import set_scalar, set_dump_param, assert_roundtrip


SAMPLE = """\
# header comment
sim_dir: /scratch/run

shock:
  v_shock: 0.04      # detected fit, M_A=8.28
  x_shock_0: 750     # shock position at t=0 [c/wpe]

dump_params:
  400:
    x_shock: 1111.7          # detected front at t=400
    x_downstream_start: 930.0
  512:
    x_shock: 1234.0
    x_downstream_start: 1050.0
"""


# ---------------------------------------------------------------------------
# set_scalar: edit in place, keep comment, round-trip
# ---------------------------------------------------------------------------

def test_set_scalar_replaces_value_and_keeps_comment():
    out = set_scalar(SAMPLE, "shock.v_shock", 0.038)
    assert "v_shock: 0.038      # detected fit, M_A=8.28" in out
    assert assert_roundtrip(out, "shock.v_shock", 0.038)


def test_set_scalar_integer_position_renders_without_decimal():
    out = set_scalar(SAMPLE, "shock.x_shock_0", 760.0)
    assert "x_shock_0: 760     # shock position at t=0 [c/wpe]" in out
    assert yaml.safe_load(out)["shock"]["x_shock_0"] == 760


def test_set_scalar_only_touches_target_line():
    out = set_scalar(SAMPLE, "shock.v_shock", 0.05)
    # every other line is byte-identical
    before = SAMPLE.split("\n")
    after = out.split("\n")
    diff = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(diff) == 1
    assert before[diff[0]].lstrip().startswith("v_shock:")


def test_set_scalar_inserts_missing_nested_key():
    """A tuner may write a key the config never carried (e.g. flash.t_shock_0_s)."""
    out = set_scalar(SAMPLE, "shock.t_shock_0", 2.5e-9)
    assert assert_roundtrip(out, "shock.t_shock_0", 2.5e-9)
    # inserted inside the shock: block, at its indentation, siblings untouched
    assert "  t_shock_0: 2.5e-09" in out
    loaded = yaml.safe_load(out)["shock"]
    assert loaded["v_shock"] == 0.04 and loaded["x_shock_0"] == 750
    # and it groups with the existing entries, not after the blank line
    lines = out.split("\n")
    assert lines[lines.index("shock:") + 3].lstrip().startswith("t_shock_0:")


def test_set_scalar_creates_missing_parent_block():
    out = set_scalar(SAMPLE, "flash.t_shock_0_s", 2.25e-9)
    assert assert_roundtrip(out, "flash.t_shock_0_s", 2.25e-9)
    assert yaml.safe_load(out)["sim_dir"] == "/scratch/run"


def test_set_scalar_missing_deep_key_still_raises():
    try:
        set_scalar(SAMPLE, "a.b.c", 1.0)
    except KeyError:
        return
    raise AssertionError("expected KeyError for a missing 3-deep key path")


def test_set_scalar_appends_missing_top_level_key():
    # upstream_window_ncells is only a commented example in real configs.
    out = set_scalar(SAMPLE, "upstream_window_ncells", 77)
    assert assert_roundtrip(out, "upstream_window_ncells", 77)
    # pre-existing keys untouched
    assert yaml.safe_load(out)["shock"]["v_shock"] == 0.04


# ---------------------------------------------------------------------------
# set_dump_param: edit existing, insert missing key / block / section
# ---------------------------------------------------------------------------

def test_set_dump_param_edits_existing_key():
    out = set_dump_param(SAMPLE, 400, "x_shock", 1100.0)
    assert "x_shock: 1100          # detected front at t=400" in out
    assert assert_roundtrip(out, "dump_params.400.x_shock", 1100)


def test_set_dump_param_inserts_missing_key_in_existing_block():
    # dump 512 has no x_downstream_start change; add a brand new key instead.
    text = SAMPLE.replace("    x_downstream_start: 1050.0\n", "")
    out = set_dump_param(text, 512, "x_downstream_start", 1051.0)
    data = yaml.safe_load(out)
    assert data["dump_params"][512]["x_downstream_start"] == 1051
    # untouched sibling still present
    assert data["dump_params"][512]["x_shock"] == 1234


def test_set_dump_param_appends_new_block():
    out = set_dump_param(SAMPLE, 240, "x_shock", 980.0)
    data = yaml.safe_load(out)
    assert data["dump_params"][240]["x_shock"] == 980
    # pre-existing blocks are intact
    assert data["dump_params"][400]["x_downstream_start"] == 930
    assert data["dump_params"][512]["x_shock"] == 1234


def test_set_dump_param_creates_section_when_absent():
    text = """\
sim_dir: /scratch/run

shock:
  v_shock: 0.04
  x_shock_0: 750
"""
    out = set_dump_param(text, 100, "x_downstream_start", 500.0)
    data = yaml.safe_load(out)
    assert data["dump_params"][100]["x_downstream_start"] == 500
    # original keys survive
    assert data["shock"]["v_shock"] == 0.04


def test_inserted_block_is_valid_yaml_and_reparses():
    out = set_dump_param(SAMPLE, 240, "x_downstream_start", 905.5)
    # full document still parses and the new value is reachable
    assert assert_roundtrip(out, "dump_params.240.x_downstream_start", 905.5)


# ---------------------------------------------------------------------------
# set_dump_param with a custom section (FLASH per-dump positions)
# ---------------------------------------------------------------------------

def test_set_dump_param_custom_section_creates_and_coexists():
    # The FLASH tuner writes to a separate top-level section so its cm-unit
    # positions never collide with the OSIRIS c/wpe dump_params.
    out = set_dump_param(SAMPLE, 20, "x_shock_cm", 0.51, section="flash_dump_params")
    out = set_dump_param(out, 20, "x_downstream_start_cm", 0.49, section="flash_dump_params")
    data = yaml.safe_load(out)
    assert data["flash_dump_params"][20]["x_shock_cm"] == 0.51
    assert data["flash_dump_params"][20]["x_downstream_start_cm"] == 0.49
    # the OSIRIS dump_params section is untouched
    assert data["dump_params"][400]["x_shock"] == 1111.7
    assert assert_roundtrip(out, "flash_dump_params.20.x_shock_cm", 0.51)


# ---------------------------------------------------------------------------
# Per-line-of-sight sections: <section>.<label>.<dump>.<key>
# ---------------------------------------------------------------------------
# A config with a fan of rays gives each ray its own front positions, so the tuner's
# write-back path gains one nesting level.

FAN_CONFIG = """\
flash_data_dir: /data/FLASH_corrected

lines_of_sight:
  los00:
    start_point: [0.0, 0.07, 0.0]
    end_point: [0.0, 1.40, 0.0]
  los30:
    start_point: [0.035, 0.061, 0.0]
    end_point: [0.7, 1.212, 0.0]

flash:
  los00:
    v_shock_est_cms: 90000000.0
  los30:
    v_shock_est_cms: 70000000.0

flash_dump_params:
  los00:
    36:
      x_shock_cm: 0.62
      x_downstream_start_cm: 0.55
  los30: {}
"""


def test_set_dump_param_edits_a_per_los_value_in_place():
    out = yaml_edit.set_dump_param(FAN_CONFIG, 36, "x_shock_cm", 0.71,
                                   section="flash_dump_params.los00")
    yaml_edit.assert_roundtrip(out, "flash_dump_params.los00.36.x_shock_cm", 0.71)
    # the other ray is untouched
    assert "los30: {}" in out


def test_set_dump_param_inserts_a_missing_key_in_an_existing_block():
    out = yaml_edit.set_dump_param(FAN_CONFIG, 36, "x_downstream_start_cm", 0.5,
                                   section="flash_dump_params.los00")
    yaml_edit.assert_roundtrip(out, "flash_dump_params.los00.36.x_downstream_start_cm", 0.5)


def test_set_dump_param_inserts_a_missing_dump_block():
    out = yaml_edit.set_dump_param(FAN_CONFIG, 52, "x_shock_cm", 0.9,
                                   section="flash_dump_params.los00")
    yaml_edit.assert_roundtrip(out, "flash_dump_params.los00.52.x_shock_cm", 0.9)
    yaml_edit.assert_roundtrip(out, "flash_dump_params.los00.36.x_shock_cm", 0.62)


def test_set_dump_param_refuses_to_invent_a_missing_los_block():
    """Inventing the intermediate mapping would guess its indentation and ordering."""
    with pytest.raises(KeyError, match="section not found"):
        yaml_edit.set_dump_param(FAN_CONFIG, 36, "x_shock_cm", 0.7,
                                 section="flash_dump_params.los99")


def test_set_scalar_edits_a_per_los_trajectory_value():
    out = yaml_edit.set_scalar(FAN_CONFIG, "flash.los30.v_shock_est_cms", 3.5e7)
    yaml_edit.assert_roundtrip(out, "flash.los30.v_shock_est_cms", 3.5e7)
    yaml_edit.assert_roundtrip(out, "flash.los00.v_shock_est_cms", 9.0e7)


def test_set_scalar_inserts_a_missing_key_three_levels_down():
    """t_shock_0_s is written by the tuner into a block that may not carry it yet."""
    out = yaml_edit.set_scalar(FAN_CONFIG, "flash.los30.t_shock_0_s", 2.0e-9)
    yaml_edit.assert_roundtrip(out, "flash.los30.t_shock_0_s", 2.0e-9)
    yaml_edit.assert_roundtrip(out, "flash.los30.v_shock_est_cms", 7.0e7)


def test_set_scalar_still_refuses_a_missing_deep_parent():
    with pytest.raises(KeyError, match="key path not found"):
        yaml_edit.set_scalar(FAN_CONFIG, "flash.los99.v_shock_est_cms", 1.0e7)


def test_set_scalar_opens_an_empty_placeholder_block():
    """`los45: {}` declares a ray whose front has not been tuned yet; nesting a
    block entry under a flow mapping would make the file unparseable."""
    cfg = "flash:\n  los00: {}\n  los30: {}\n"
    out = yaml_edit.set_scalar(cfg, "flash.los00.v_shock_est_cms", 9.0e7)
    out = yaml_edit.set_scalar(out, "flash.los00.t_shock_0_s", 2.0e-9)
    yaml_edit.assert_roundtrip(out, "flash.los00.v_shock_est_cms", 9.0e7)
    yaml_edit.assert_roundtrip(out, "flash.los00.t_shock_0_s", 2.0e-9)
    assert yaml.safe_load(out)["flash"]["los30"] == {}


def test_set_dump_param_opens_an_empty_placeholder_block():
    cfg = "flash_dump_params:\n  los00: {}\n  los30: {}\n"
    out = yaml_edit.set_dump_param(cfg, 36, "x_shock_cm", 0.62,
                                   section="flash_dump_params.los00")
    out = yaml_edit.set_dump_param(out, 36, "x_downstream_start_cm", 0.55,
                                   section="flash_dump_params.los00")
    yaml_edit.assert_roundtrip(out, "flash_dump_params.los00.36.x_shock_cm", 0.62)
    yaml_edit.assert_roundtrip(out, "flash_dump_params.los00.36.x_downstream_start_cm", 0.55)
    assert yaml.safe_load(out)["flash_dump_params"]["los30"] == {}


def test_opening_a_placeholder_keeps_its_trailing_comment():
    cfg = "flash:\n  los45: {}   # truncated by xmax\n"
    out = yaml_edit.set_scalar(cfg, "flash.los45.v_shock_est_cms", 5.0e7)
    assert "# truncated by xmax" in out
    yaml_edit.assert_roundtrip(out, "flash.los45.v_shock_est_cms", 5.0e7)
