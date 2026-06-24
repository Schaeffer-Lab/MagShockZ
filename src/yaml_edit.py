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

import re

import yaml

# A "key:" line: leading indent, an identifier, a colon, then the remainder.
_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_]+):(.*)$")
# Split a scalar line into (prefix up to & incl. the colon+space)(value)(trailing
# whitespace + optional comment) so only the middle group is rewritten.
_VALUE_RE = re.compile(r"^(\s*[A-Za-z0-9_]+:[ \t]*)([^#\n]*?)([ \t]*(?:#.*)?)$")

_INSERT_COMMENT = "  # set by tune_shock"


def _fmt(value) -> str:
    """Render a Python value as a compact YAML scalar token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:g}"
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

    A missing *top-level* key (no dots) is appended at end-of-file; a missing
    *nested* key raises ``KeyError`` (its parent structure is assumed to exist)."""
    keys = dotted_key.split(".")
    lines = text.split("\n")
    idx = _find_line(lines, keys)
    if idx is not None:
        lines[idx] = _set_line(lines[idx], value)
        return "\n".join(lines)
    if len(keys) == 1:
        new_line = f"{dotted_key}: {_fmt(value)}{_INSERT_COMMENT}"
        if lines and lines[-1] == "":
            lines[-1:] = [new_line, ""]
        else:
            lines.append(new_line)
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
    assert node == expected, f"{dotted_path} = {node!r}, expected {expected!r}"
    return True
