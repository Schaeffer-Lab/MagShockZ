"""Every numeric scalar in a checked-in YAML must parse as a number, not a string.

PyYAML implements **YAML 1.1**, whose float resolver is stricter than it looks: an
exponent form needs *both* a ``.`` in the mantissa *and* an explicit sign on the
exponent.  So ``1e-9`` (no dot) and ``5.0e18`` (unsigned exponent) load as **strings**,
while ``1.0e-09``, ``1.0e-9`` and ``5.0e+18`` load as floats — a negative exponent
already carries its sign, which is why only some of these bite.  Nothing complains at
load time, and most consumers call
``float(...)`` so they keep working — until something compares or does arithmetic and
gets a confusing failure (``yaml_edit.assert_roundtrip`` hit exactly this after
``tune_flash_shock``'s ``t 1`` wrote ``t_shock_0_s: 1e-9``).

This walks the repo's configs and run specs and fails on any scalar that is a string
Python can nonetheless parse as a number.
"""

import glob
import os

import pytest
import yaml

_REPO = os.path.join(os.path.dirname(__file__), "..")
_PATTERNS = ("config/*.yaml", "runs/*.yaml")


def _yaml_files():
    out = []
    for pattern in _PATTERNS:
        out += sorted(glob.glob(os.path.join(_REPO, pattern)))
    return out


def _scalars(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _scalars(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _scalars(v, f"{path}[{i}]")
    else:
        yield path, node


def _numeric_strings(data):
    bad = []
    for path, value in _scalars(data or {}):
        if not isinstance(value, str):
            continue
        try:
            float(value)
        except ValueError:
            continue                     # ordinary text, fine
        bad.append((path, value))
    return bad


def test_there_are_yaml_files_to_check():
    assert _yaml_files(), "no config/*.yaml or runs/*.yaml found — bad glob?"


@pytest.mark.parametrize("path", _yaml_files(), ids=os.path.basename)
def test_no_numeric_looking_strings(path):
    bad = _numeric_strings(yaml.safe_load(open(path)))
    assert not bad, (
        f"{os.path.basename(path)} has scalars that look numeric but load as strings: "
        + ", ".join(f"{k} = {v!r}" for k, v in bad)
        + ". YAML 1.1 needs a dot in the mantissa AND a sign on the exponent "
          "(1.0e-09, 5.0e+18).")


def test_the_check_would_catch_the_known_traps():
    """Guard the guard: these are the spellings that silently become strings.

    ``a`` has no dot in the mantissa; ``b`` has an unsigned exponent.  Both are the
    real traps — the ones that were checked into this repo.
    """
    trap = yaml.safe_load("a: 1e-9\nb: 5.0e18\n")
    assert [k for k, _ in _numeric_strings(trap)] == ["a", "b"]

    # A negative exponent already carries its sign, so 1.0e-9 is fine as written.
    good = yaml.safe_load("a: 1.0e-09\nb: 1.0e-9\nc: 5.0e+18\n"
                          "d: 0.2\ne: 350\nf: not_a_number\n")
    assert _numeric_strings(good) == []
