"""Tests for src/yaml_edit.py — comment-preserving scalar edits (PyYAML only, CI-safe)."""

import yaml

from yaml_edit import set_scalar, set_dump_param, assert_roundtrip


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


def test_set_scalar_missing_nested_key_raises():
    try:
        set_scalar(SAMPLE, "shock.nonexistent", 1.0)
    except KeyError:
        return
    raise AssertionError("expected KeyError for a missing nested key path")


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
