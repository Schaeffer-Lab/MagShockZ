"""Tests for yaml_edit.out_dir — which results directory a config writes to.

The rule that matters: two configs analysing the SAME dataset must not silently share
an output directory, because the FLASH scripts hand data to each other through it
(flash_rh_prediction reads flash_overview_*.npz back out of it), so a collision is a
wrong-line-of-sight bug and not just clutter.
"""

import importlib.util
import os
import shutil

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "yaml_edit.py")
_spec = importlib.util.spec_from_file_location("yaml_edit", _PATH)
yaml_edit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yaml_edit)

# out_dir creates directories under the real <repo>/results, so the tests use a
# throwaway dataset name and delete its tree afterwards rather than littering it.
NAME = "_test_dataset_FLASH_2026-07"
DATASET = f"/data/sims/{NAME}"
_RESULTS = os.path.realpath(os.path.join(os.path.dirname(_PATH), "..", "results"))


@pytest.fixture(autouse=True)
def clean_results():
    yield
    shutil.rmtree(os.path.join(_RESULTS, NAME), ignore_errors=True)


def _rel(path):
    """Path relative to <repo>/results, with the '..' segments resolved."""
    return os.path.relpath(os.path.realpath(path), _RESULTS)


def test_default_is_keyed_on_the_dataset(tmp_path):
    assert _rel(yaml_edit.out_dir(DATASET)) == NAME


def test_trailing_slash_on_the_dataset_is_ignored():
    assert _rel(yaml_edit.out_dir(DATASET + "/")) == NAME


def test_results_subdir_nests_under_the_dataset():
    out = yaml_edit.out_dir(DATASET, cfg={"results_subdir": "offaxis"})
    assert _rel(out) == os.path.join(NAME, "offaxis")


def test_results_subdir_auto_uses_the_config_stem():
    out = yaml_edit.out_dir(DATASET, cfg={"results_subdir": "auto"},
                            config_path="config/flash_3d_2026-07_offaxis.yaml")
    assert _rel(out) == os.path.join(NAME, "flash_3d_2026-07_offaxis")


def test_two_configs_on_one_dataset_do_not_collide():
    on_axis = yaml_edit.out_dir(DATASET, cfg={}, config_path="config/a.yaml")
    off_axis = yaml_edit.out_dir(DATASET, cfg={"results_subdir": "auto"},
                                 config_path="config/a_offaxis.yaml")
    assert on_axis != off_axis
    assert not os.path.samefile(on_axis, off_axis)


def test_results_subdir_auto_without_a_config_path_is_an_error():
    with pytest.raises(ValueError, match="needs the config path"):
        yaml_edit.out_dir(DATASET, cfg={"results_subdir": "auto"})


def test_results_dir_decouples_from_the_dataset_entirely(tmp_path):
    out = yaml_edit.out_dir(DATASET, cfg={"results_dir": str(tmp_path / "elsewhere")})
    assert os.path.realpath(out) == os.path.realpath(str(tmp_path / "elsewhere"))


def test_relative_results_dir_is_repo_relative():
    out = yaml_edit.out_dir(DATASET, cfg={"results_dir": f"results/{NAME}_custom"})
    try:
        assert _rel(out) == f"{NAME}_custom"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_override_beats_every_config_key(tmp_path):
    out = yaml_edit.out_dir(DATASET, str(tmp_path / "cli"),
                            cfg={"results_subdir": "auto",
                                 "results_dir": str(tmp_path / "cfg")},
                            config_path="config/x.yaml")
    assert os.path.realpath(out) == os.path.realpath(str(tmp_path / "cli"))


def test_the_directory_is_created(tmp_path):
    out = yaml_edit.out_dir(str(tmp_path / "dataset"),
                            cfg={"results_dir": str(tmp_path / "made" / "deep")})
    assert os.path.isdir(out)


# ---------------------------------------------------------------------------
# _fmt / assert_roundtrip — floats must come back as FLOATS
# ---------------------------------------------------------------------------
# PyYAML implements YAML 1.1, whose float resolver rejects an exponent form with no
# '.' in the mantissa: a bare `1e-09` (what %g gives for 1 ns in seconds) loads as the
# string '1e-09'. tune_flash_shock's `t 1` + `save` hit exactly that.

import yaml as _yaml


@pytest.mark.parametrize("value", [1e-9, 2.5e-9, -1e-9, 1e-15, 1e20, 8.7e-10])
def test_small_and_large_floats_reparse_as_floats(value):
    text = f"x: {yaml_edit._fmt(value)}"
    back = _yaml.safe_load(text)["x"]
    assert isinstance(back, float), f"{value!r} rendered as {text!r}, read back {back!r}"
    assert back == pytest.approx(value, rel=1e-6)


def test_exponent_mantissa_is_padded_with_a_dot():
    assert yaml_edit._fmt(1e-9) == "1.0e-09"


def test_plain_and_integral_floats_are_left_compact():
    assert yaml_edit._fmt(0.2) == "0.2"
    assert yaml_edit._fmt(350.0) == "350"
    assert yaml_edit._fmt(35000000) == "35000000"


def test_non_finite_floats_use_yaml_spellings():
    assert yaml_edit._fmt(float("inf")) == ".inf"
    assert yaml_edit._fmt(float("-inf")) == "-.inf"
    assert yaml_edit._fmt(float("nan")) == ".nan"
    assert _yaml.safe_load("x: .inf")["x"] == float("inf")


def test_set_scalar_then_roundtrip_survives_a_nanosecond_anchor():
    """The reported failure: `t 1` writes 1e-09 and save verifies it."""
    text = "flash:\n  t_shock_0_s: 8.7e-10  # set by tune_shock\n"
    new = yaml_edit.set_scalar(text, "flash.t_shock_0_s", float(f"{1e-9:.6g}"))
    assert yaml_edit.assert_roundtrip(new, "flash.t_shock_0_s", 1e-9)


def test_roundtrip_tolerates_percent_g_rounding():
    """%g keeps 6 significant digits, so a longer float comes back legitimately rounded."""
    text = yaml_edit.set_scalar("a:\n  b: 0\n", "a.b", 1.2345678901e-9)
    assert yaml_edit.assert_roundtrip(text, "a.b", 1.2345678901e-9)


def test_roundtrip_still_rejects_a_numeric_looking_string():
    """The guard that caught the original bug must keep firing."""
    with pytest.raises(AssertionError, match="str"):
        yaml_edit.assert_roundtrip("a:\n  b: '1e-09'\n", "a.b", 1e-9)


def test_roundtrip_rejects_a_genuinely_wrong_number():
    with pytest.raises(AssertionError):
        yaml_edit.assert_roundtrip("a:\n  b: 2.0\n", "a.b", 1.0)
