"""Filename heuristic: natural ordering.

Cameras name clips with a running counter (DJI_0838.mov, DJI_0839.mov, …), so the
numeric suffix encodes capture order. We use it to order clips correctly (so
DJI_99 sorts before DJI_100), which feeds the preliminary timeline layout and the
fallback placement for clips that cannot be aligned by transcription.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"(\d+)")


def natural_key(name: str) -> list[tuple[int, int, str]]:
    """Sort key that orders embedded numbers numerically (DJI_99 < DJI_100).

    Each chunk is a comparable (rank, int_value, str_value) tuple so int and str
    chunks never compare against each other.
    """
    key: list[tuple[int, int, str]] = []
    for part in _NUM.split(name):
        if part.isdigit():
            key.append((0, int(part), ""))
        elif part:
            key.append((1, 0, part.lower()))
    return key
