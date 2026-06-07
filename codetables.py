"""
Fetch lower bounds on minimum distance from codetables.de for GF(q) codes.

Table format (Magma/plain-text):
  Row i (1-indexed) = length n=i
  Row[j] (0-indexed) = best known d for a [n, k=j+1] code over GF(q)
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

_URL = "https://codetables.de/BKLC/LowerBoundList.php?q={q}"
_CACHE_DIR = Path(".")


def fetch_table(q: int = 8, cache: bool = True) -> list[list[int]]:
    """Return the full bounds table as a list-of-rows (0-indexed by n-1)."""
    cache_file = _CACHE_DIR / f".gf{q}_bounds_cache.txt"

    if cache and cache_file.exists():
        text = cache_file.read_text()
    else:
        url = _URL.format(q=q)
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode()
        if cache:
            cache_file.write_text(text)

    return [
        [int(x) for x in m.group(1).split(",")]
        for m in re.finditer(r"\[([0-9,]+)\]", text)
    ]


def bounds_for_n(n: int, q: int = 8, cache: bool = True) -> dict[int, int]:
    """Return {k: best_known_d} for every k, for a fixed code length n."""
    table = fetch_table(q=q, cache=cache)
    if not (1 <= n <= len(table)):
        raise ValueError(f"n={n} out of table range 1..{len(table)}")
    row = table[n - 1]
    return {k + 1: row[k] for k in range(len(row))}


def best_known_d(n: int, k: int, q: int = 8, cache: bool = True) -> int:
    """Best known minimum distance for a linear [n, k] code over GF(q)."""
    return bounds_for_n(n, q=q, cache=cache)[k]
