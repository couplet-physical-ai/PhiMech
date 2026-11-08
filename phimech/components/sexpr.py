"""A small, fast, dependency-free s-expression reader for KiCad files.

Why this exists instead of `kiutils`/`pcbnew`: `phimech/components` is stdlib-only (rule 9) so
that it imports under python 3.9, FreeCAD's bundled interpreter and the MCP venv alike, and
PhiMech only ever *reads* boards. A `.kicad_pcb` is a plain s-expression tree, so ~120 lines of
tokenizer plus four accessors is the whole dependency surface — and it stays readable when KiCad
changes its schema (KiCad 10 moved footprint placement from `(at ...)` to `(transform ...)`;
nothing here had to change).

Value mapping (deliberately lossy but predictable):
* `(` ... `)`      -> python `list`
* `"quoted text"`  -> `str`, unescaped, **never** coerced to a number (so a net named "3.3" stays
                      a string)
* bare atom        -> `float` when it parses as a number, else `str` (`F.Cu`, `yes`, `smd`)

Numbers become floats, not ints, because every dimension in a board file is a length in mm and
mixing int/float only invites `1/2 == 0`-style bugs downstream.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

SExpr = Union[str, float, List[Any]]

# One regex, one pass: the tokenizer is the hot loop, and finditer in C beats a python
# character loop by ~30x on a 4500-line board.
_TOKEN_RE = re.compile(
    r"""[\s]*(?:
          (?P<open>\()
        | (?P<close>\))
        | "(?P<str>(?:[^"\\]|\\.)*)"
        | (?P<atom>[^\s()"]+)
    )""",
    re.VERBOSE,
)

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def _unescape(s: str) -> str:
    if "\\" not in s:
        return s
    out: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _atom(text: str) -> SExpr:
    if _NUMBER_RE.match(text):
        try:
            return float(text)
        except ValueError:  # pragma: no cover - regex already guarantees this parses
            return text
    return text


def parse(text: str) -> List[SExpr]:
    """Parse `text` into a list of top-level forms.

    A `.kicad_pcb` has exactly one top-level form, so callers normally want `parse_one`.
    Raises `ValueError` with a character offset on unbalanced parentheses or stray atoms.
    """
    top: List[SExpr] = []
    stack: List[List[SExpr]] = []
    pos = 0
    n = len(text)
    match = _TOKEN_RE.match
    while pos < n:
        m = match(text, pos)
        if m is None:
            if text[pos:].strip() == "":
                break
            raise ValueError("unparsable input at offset %d: %r" % (pos, text[pos : pos + 20]))
        pos = m.end()
        if m.group("open"):
            node: List[SExpr] = []
            if stack:
                stack[-1].append(node)
            else:
                top.append(node)
            stack.append(node)
        elif m.group("close"):
            if not stack:
                raise ValueError("unbalanced ')' at offset %d" % m.start())
            stack.pop()
        else:
            g = m.group("str")
            value: SExpr = _unescape(g) if g is not None else _atom(m.group("atom"))
            if not stack:
                top.append(value)
            else:
                stack[-1].append(value)
    if stack:
        raise ValueError("unbalanced '(': %d form(s) still open at end of input" % len(stack))
    return top


def parse_one(text: str) -> List[SExpr]:
    """Parse `text` and return its single top-level list form (the whole board)."""
    forms = parse(text)
    lists = [f for f in forms if isinstance(f, list)]
    if not lists:
        raise ValueError("no s-expression form found")
    return lists[0]


def parse_file(path: Union[str, Path]) -> List[SExpr]:
    """Read and parse a file, returning its single top-level form."""
    return parse_one(Path(path).read_text(encoding="utf-8", errors="replace"))


# --------------------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------------------


def head(node: SExpr) -> Optional[str]:
    """The symbol at the head of a form, or None if `node` is not a headed list."""
    if isinstance(node, list) and node and isinstance(node[0], str):
        return node[0]
    return None


def find_all(node: SExpr, name: str) -> List[List[SExpr]]:
    """Every *immediate* child form of `node` whose head is `name`."""
    if not isinstance(node, list):
        return []
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def find(node: SExpr, name: str) -> Optional[List[SExpr]]:
    """The first immediate child form named `name`, or None."""
    for c in find_all(node, name):
        return c
    return None


def get_attr(node: SExpr, name: str, default: Any = None) -> Any:
    """Value of a one-value child form: `get_attr(general, "thickness")` -> 1.6.

    Returns `default` when the form is absent or has no value; a list of values when the form
    carries several (`(size 1.7 1.7)` -> `[1.7, 1.7]`).
    """
    sub = find(node, name)
    if sub is None or len(sub) < 2:
        return default
    if len(sub) == 2:
        return sub[1]
    return list(sub[1:])


def get_vec(node: SExpr, name: str, default: Any = None) -> Any:
    """Value of a multi-value child form as a list of floats, e.g. `(start 1 2)` -> [1.0, 2.0]."""
    sub = find(node, name)
    if sub is None or len(sub) < 2:
        return default
    out: List[float] = []
    for v in sub[1:]:
        if isinstance(v, float):
            out.append(v)
        elif isinstance(v, str):
            try:
                out.append(float(v))
            except ValueError:
                break
        else:
            break
    return out or default


def walk(node: SExpr) -> Iterator[List[SExpr]]:
    """Pre-order traversal of every list form in the tree (used to sweep for stray Edge.Cuts)."""
    if isinstance(node, list):
        yield node
        for c in node:
            if isinstance(c, list):
                for sub in walk(c):
                    yield sub


def find_deep(node: SExpr, name: str) -> List[List[SExpr]]:
    """Every form named `name` anywhere in the tree, at any depth."""
    return [f for f in walk(node) if f and f[0] == name]


def layer_of(node: SExpr) -> Optional[str]:
    """The `(layer "X")` of a graphic form, or None. Handles `(layer X)` unquoted too."""
    v = get_attr(node, "layer")
    return v if isinstance(v, str) else None


__all__ = [
    "SExpr",
    "parse",
    "parse_one",
    "parse_file",
    "head",
    "find",
    "find_all",
    "find_deep",
    "get_attr",
    "get_vec",
    "walk",
    "layer_of",
]
