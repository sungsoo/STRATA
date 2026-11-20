"""LaTeX checks that the release gates run through check_numbers.py (v21, R53).

``range_macros_in_math``: a generated macro whose value is a range ("16--18") prints
as "16 − −18" inside math mode; the v20 manuscript had one such use in Sec. 6.4.
"""
from __future__ import annotations

import re
from pathlib import Path

MACRO_DEF = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")
MATH = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.S)


def range_macros(macro_file):
    text = Path(macro_file).read_text(encoding="utf-8")
    return {name for name, value in MACRO_DEF.findall(text) if "--" in value}


def range_macros_in_math(tex_files, macro_file):
    """Every (file, macro, math snippet) where a range macro is used inside $...$."""
    ranges = range_macros(macro_file)
    hits = []
    for f in tex_files:
        text = re.sub(r"(?<!\\)%.*", "", Path(f).read_text(encoding="utf-8"))
        for m in MATH.finditer(text):
            for name in re.findall(r"\\(\w+)", m.group(1)):
                if name in ranges:
                    hits.append((str(f), name, m.group(0)[:60]))
    return hits
