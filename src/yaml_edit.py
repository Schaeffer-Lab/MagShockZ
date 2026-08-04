"""Comment-preserving, dependency-light scalar edits to an analysis YAML.

Pure ``str -> str`` functions that locate a key by its nested path and replace
ONLY the scalar value token on that line, leaving indentation, key order and any
trailing ``# comment`` byte-for-byte intact.  This is what ``tune_shock.py`` uses
to write tuned shock parameters back into a comment-rich ``config/*.yaml`` without
a full YAML round-trip (PyYAML's ``safe_dump`` would strip every comment).

PyYAML is used here only to *verify* (``assert_roundtrip``) that the edited text
re-parses to the intended value — never to re-emit it.  The module stays in the
CI-pure layer (stdlib + PyYAML only) so it is unit-tested without the OSIRIS stack.

Supported shapes (all that the MagShockZ analysis config needs):

    shock:
      v_shock: 0.04        # set_scalar("shock.v_shock", ...)
      x_shock_0: 750
    dump_params:
      400:
        x_shock: 1111.7    # set_dump_param(400, "x_shock", ...)
        x_downstream_start: 930.0

``set_dump_param`` also *inserts* a correctly-indented key (or a whole ``<idx>:``
block, or the ``dump_params:`` section) when it does not yet exist.
"""

import math
import os
import re

import yaml

# A "key:" line: leading indent, an identifier, a colon, then the remainder.
_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_]+):(.*)$")
# Split a scalar line into (prefix up to & incl. the colon+space)(value)(trailing
# whitespace + optional comment) so only the middle group is rewritten.
_VALUE_RE = re.compile(r"^(\s*[A-Za-z0-9_]+:[ \t]*)([^#\n]*?)([ \t]*(?:#.*)?)$")

_INSERT_COMMENT = "  # set by tune_shock"


def _fmt(value) -> str:
    """Render a Python value as a compact YAML scalar token.

    Floats are rendered so that PyYAML reads them back as floats.  That needs care:
    PyYAML implements **YAML 1.1**, whose float resolver only accepts an exponent form
    when the mantissa carries a ``.`` — so a bare ``1e-09`` (what ``%g`` gives for
    1 ns in seconds) round-trips as the *string* ``'1e-09'``, not a number.  The
    mantissa is padded to ``1.0e-09`` to keep it a float.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        text = f"{value:g}"
        if "e" in text and "." not in text:
            mantissa, _, exponent = text.partition("e")
            text = f"{mantissa}.0e{exponent}"
        return text
    return str(value)


def _find_line(lines, keys):
    """Index of the line whose nested key-path equals ``keys`` (list of str), or None."""
    stack = []  # (indent, key) of the enclosing mappings, outermost first
    for i, line in enumerate(lines):
        m = _KEY_RE.match(line)
        if not m:
            continue  # blank line, comment, or non-mapping content
        indent = len(m.group(1))
        key = m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = [k for _, k in stack] + [key]
        stack.append((indent, key))
        if path == keys:
            return i
    return None


def _set_line(line, value):
    """Replace only the value token on a ``key: value  # comment`` line."""
    m = _VALUE_RE.match(line)
    if not m:
        raise ValueError(f"cannot parse scalar line: {line!r}")
    return f"{m.group(1)}{_fmt(value)}{m.group(3)}"


def set_scalar(text, dotted_key, value):
    """Return ``text`` with the scalar at ``dotted_key`` (e.g. ``"shock.v_shock"``)
    replaced by ``value``.

    A missing key is inserted rather than an error: a *top-level* key (no dots) is
    appended at end-of-file; a missing ``parent.key`` is inserted under ``parent:``
    with the block's own indentation, or as a fresh ``parent:`` block at EOF when the
    parent is absent too (this is how a tuner adds a value the config never carried,
    e.g. ``flash.t_shock_0_s``).  Deeper missing paths still raise ``KeyError``."""
    keys = dotted_key.split(".")
    lines = text.split("\n")
    idx = _find_line(lines, keys)
    if idx is not None:
        lines[idx] = _set_line(lines[idx], value)
        return "\n".join(lines)

    val = _fmt(value)
    if len(keys) == 1:
        new_line = f"{dotted_key}: {val}{_INSERT_COMMENT}"
        if lines and lines[-1] == "":
            lines[-1:] = [new_line, ""]
        else:
            lines.append(new_line)
        return "\n".join(lines)

    if len(keys) == 2:
        parent, key = keys
        p_idx = _find_line(lines, [parent])
        if p_idx is None:                       # no parent block at all -> append one
            block = [f"{parent}:", f"  {key}: {val}{_INSERT_COMMENT}"]
            if lines and lines[-1] == "":
                lines[-1:] = block + [""]
            else:
                lines += block
            return "\n".join(lines)
        p_indent = len(_KEY_RE.match(lines[p_idx]).group(1))
        start, end = _section_bounds(lines, p_idx)
        child_indent, _ = _child_indents(lines, start, end, p_indent)
        pos = _last_content_idx(lines, start, end, child_indent) + 1
        lines.insert(pos, f"{' ' * child_indent}{key}: {val}{_INSERT_COMMENT}")
        return "\n".join(lines)

    raise KeyError(f"key path not found: {dotted_key}")


def _section_bounds(lines, parent_idx):
    """[start, end) line range of the block nested under ``lines[parent_idx]``."""
    parent_indent = len(_KEY_RE.match(lines[parent_idx]).group(1))
    end = len(lines)
    for j in range(parent_idx + 1, len(lines)):
        m = _KEY_RE.match(lines[j])
        if m and len(m.group(1)) <= parent_indent:
            end = j
            break
    return parent_idx + 1, end


def _last_content_idx(lines, start, end, min_indent):
    """Index of the last ``key:`` line in [start, end) at indent >= ``min_indent``.

    Trailing comments / blank lines are skipped so insertions group with the real
    entries instead of landing after a comment block.  Returns ``start - 1`` when
    the range holds no such content line."""
    last = start - 1
    for j in range(start, end):
        m = _KEY_RE.match(lines[j])
        if m and len(m.group(1)) >= min_indent:
            last = j
    return last


def _child_indents(lines, start, end, parent_indent):
    """Infer (child_indent, grandchild_indent) from existing entries in a section."""
    child = None
    grand = None
    for j in range(start, end):
        m = _KEY_RE.match(lines[j])
        if not m:
            continue
        ind = len(m.group(1))
        if ind > parent_indent and (child is None or ind < child):
            child = ind
    if child is not None:
        for j in range(start, end):
            m = _KEY_RE.match(lines[j])
            if m and len(m.group(1)) > child:
                grand = len(m.group(1))
                break
    if child is None:
        child = parent_indent + 2
    if grand is None:
        grand = child + 2
    return child, grand


def set_dump_param(text, dump_idx, key, value, section="dump_params"):
    """Return ``text`` with ``<section>.<dump_idx>.<key>`` set to ``value``.

    Edits the line in place when it exists; otherwise inserts the key (or the whole
    ``<dump_idx>:`` block, or the ``<section>:`` section) with matching indentation.

    ``section`` is the top-level mapping the per-dump blocks live under; it defaults
    to ``"dump_params"`` (the OSIRIS analysis config).  The FLASH tuner passes
    ``"flash_dump_params"`` so its physical-unit per-dump positions stay separate
    from the OSIRIS c/ωpe ``dump_params``.
    """
    keys = [section, str(dump_idx), key]
    lines = text.split("\n")

    idx = _find_line(lines, keys)
    if idx is not None:
        lines[idx] = _set_line(lines[idx], value)
        return "\n".join(lines)

    dp_idx = _find_line(lines, [section])
    val = _fmt(value)

    # No <section> mapping at all -> append one at EOF.
    if dp_idx is None:
        block = [f"{section}:",
                 f"  {dump_idx}:",
                 f"    {key}: {val}{_INSERT_COMMENT}"]
        if lines and lines[-1] == "":
            lines[-1:] = block + [""]
        else:
            lines += block
        return "\n".join(lines)

    dp_indent = len(_KEY_RE.match(lines[dp_idx]).group(1))
    start, end = _section_bounds(lines, dp_idx)
    child_indent, grand_indent = _child_indents(lines, start, end, dp_indent)

    # Does the <dump_idx>: block already exist (just missing this key)?
    block_idx = _find_line(lines, [section, str(dump_idx)])
    if block_idx is not None:
        b_start, b_end = _section_bounds(lines, block_idx)
        pos = _last_content_idx(lines, b_start, b_end, grand_indent) + 1
        lines.insert(pos, f"{' ' * grand_indent}{key}: {val}{_INSERT_COMMENT}")
        return "\n".join(lines)

    # Block absent: insert a fresh <dump_idx>: block right after the last existing
    # entry (before any trailing comments), so it groups with the other dumps.
    pos = _last_content_idx(lines, start, end, child_indent) + 1
    new_block = [f"{' ' * child_indent}{dump_idx}:",
                 f"{' ' * grand_indent}{key}: {val}{_INSERT_COMMENT}"]
    lines[pos:pos] = new_block
    return "\n".join(lines)


def assert_roundtrip(text, dotted_path, expected):
    """Verify the edited ``text`` parses (PyYAML) to ``expected`` at ``dotted_path``.

    ``dotted_path`` segments index into nested mappings; numeric segments match
    integer keys (e.g. ``"dump_params.400.x_shock"``).  Raises ``AssertionError``
    on mismatch.  Used by ``tune_shock.py`` as a post-write sanity check.
    """
    data = yaml.safe_load(text)
    node = data
    for seg in dotted_path.split("."):
        if isinstance(node, dict) and seg not in node and seg.isdigit() and int(seg) in node:
            seg = int(seg)
        node = node[seg]
    # Numbers are compared numerically: _fmt renders with %g, so a value carrying more
    # than 6 significant digits comes back legitimately rounded.  A string that merely
    # *looks* numeric is NOT accepted — that is the YAML 1.1 exponent trap _fmt guards
    # against ('1e-09' resolves to a str), and it must keep failing loudly.
    # rel_tol is set just above %g's worst case: 6 significant digits means the last
    # kept digit can move by half an ulp, i.e. up to ~5e-6 relative.
    ok = (math.isclose(node, expected, rel_tol=1e-5, abs_tol=0.0)
          if _is_number(node) and _is_number(expected) else node == expected)
    assert ok, f"{dotted_path} = {node!r} ({type(node).__name__}), expected {expected!r}"
    return True


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
# Shared interactive-tuner helpers
# ---------------------------------------------------------------------------
# ``tune_shock.py`` (OSIRIS) and ``tune_flash_shock.py`` (FLASH) drive the same
# "show a trial value, then write it back to the comment-rich config" loop, so the
# loop's plumbing lives here next to the set_scalar / set_dump_param edits it wraps.
# These do touch the filesystem / stdin, unlike the pure str->str editors above, but
# they add no new dependency (stdlib only) so the module stays CI-importable.

def out_dir(base_dir, override=None, *, cfg=None, config_path=None):
    """Resolve (and create) a script's results output directory.

    The default is ``<repo>/results/<basename of base_dir>`` — keyed on the *dataset*
    (``base_dir`` is the run's sim/FLASH directory), so everything derived from one
    run stays in one tree and dataset-level caches (``movie3d/grids``) are shared.

    That is wrong as soon as **two configs analyse the same dataset differently** —
    a second line of sight, say.  Their outputs then overwrite each other, and
    ``flash_rh_prediction`` reads ``flash_overview_*.npz`` back out of this directory,
    so it would silently pick up the other config's line-out.  A variant config
    therefore has to claim its own directory, cheapest first:

    ``results_subdir``
        A sub-directory of the dataset's tree: ``results/<dataset>/<results_subdir>``.
        Use ``auto`` to name it after the config file (``flash_3d_2026-07_offaxis.yaml``
        → ``flash_3d_2026-07_offaxis``).  Preferred — one line, keeps the run's
        outputs together, and sits alongside ``movie3d/`` rather than duplicating it.
    ``results_dir``
        A path, absolute or relative to the repo root, for output somewhere else
        entirely.  Fully decoupled from the dataset.

    ``override`` (the scripts' ``--output-dir``) beats both.
    """
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cfg = cfg or {}

    if override:
        out = override
    elif cfg.get("results_dir"):
        out = cfg["results_dir"]
        if not os.path.isabs(out):
            out = os.path.join(repo, out)
    else:
        out = os.path.join(repo, "results", os.path.basename(base_dir.rstrip("/")))
        sub = cfg.get("results_subdir")
        if sub:
            if sub == "auto":
                if not config_path:
                    raise ValueError(
                        "results_subdir: auto needs the config path — this script has "
                        "not passed one; name the sub-directory explicitly instead.")
                sub = os.path.splitext(os.path.basename(config_path))[0]
            out = os.path.join(out, str(sub))

    os.makedirs(out, exist_ok=True)
    return out


def ask_yes(prompt):
    """Prompt for a y/N confirmation; treat EOF (piped/empty stdin) as 'no'."""
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def normalize_scalar(val):
    """Match ``_fmt``'s compact rendering when verifying a round-trip."""
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def aligned_diff(old, new):
    """Pair lines for display; insertions show against an empty old line."""
    n = max(len(old), len(new))
    old = old + [""] * (n - len(old))
    new = new + [""] * (n - len(new))
    return list(zip(old, new))


def confirm_write(config_path, edits, no_write=False):
    """Apply ``edits`` (``(dotted_path, value)`` pairs) to a config YAML, then write.

    A path whose first segment ends with ``dump_params`` (``dump_params`` for the
    OSIRIS config, ``flash_dump_params`` for FLASH) is treated as
    ``<section>.<idx>.<key>`` and routed to :func:`set_dump_param`; everything else is
    a plain :func:`set_scalar`.  Each edit is verified to round-trip before the file is
    touched; the line-level diff is shown and a y/N confirmation is asked (unless
    ``no_write``, which only prints the would-be edits).  Returns True iff written.
    """
    if no_write:
        print("  [--no-write] would write:")
        for path, val in edits:
            print(f"    {path} = {val}")
        return False

    with open(config_path) as f:
        text = f.read()
    new_text = text
    for path, val in edits:
        parts = path.split(".")
        if len(parts) == 3 and parts[0].endswith("dump_params"):
            new_text = set_dump_param(new_text, int(parts[1]), parts[2], val,
                                      section=parts[0])
        else:
            new_text = set_scalar(new_text, path, val)
        assert_roundtrip(new_text, path, normalize_scalar(val))

    print("  pending edits:")
    for a, b in aligned_diff(text.split("\n"), new_text.split("\n")):
        if a != b:
            print(f"    - {a}")
            print(f"    + {b}")
    if not ask_yes(f"  write these to {os.path.basename(config_path)}? [y/N] "):
        print("  not written.")
        return False
    with open(config_path, "w") as f:
        f.write(new_text)
    print(f"  wrote → {config_path}")
    return True
