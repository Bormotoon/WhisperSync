"""Filename heuristics: natural ordering and consecutive-sequence detection.

Cameras name clips with a running counter (DJI_0838.mov, DJI_0839.mov, …), so the
numeric suffix encodes capture order. We use it to order clips correctly (so
DJI_99 sorts before DJI_100) and to detect runs of consecutive files, which feeds
the preliminary timeline layout and the fallback placement for clips that cannot
be aligned by transcription.
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


def split_index(name: str) -> tuple[str, int | None]:
    """Split a filename (without extension is fine) into (prefix, trailing
    number). The prefix is everything before the last numeric run, so
    'DJI_0839.mov' -> ('DJI_', 839). Returns (name, None) if there is no number.
    """
    matches = list(_NUM.finditer(name))
    if not matches:
        return name, None
    last = matches[-1]
    return name[: last.start()], int(last.group())


def group_sequences(names: list[str]) -> list[int]:
    """Assign a sequence-group id to each name. Two adjacent names belong to the
    same group when they share a prefix and their numbers are consecutive
    (n, n+1). Names are expected pre-sorted in natural order.
    """
    groups: list[int] = []
    gid = -1
    prev_prefix: str | None = None
    prev_num: int | None = None
    for name in names:
        prefix, num = split_index(name)
        consecutive = (
            prev_num is not None
            and num is not None
            and prefix == prev_prefix
            and num == prev_num + 1
        )
        if not consecutive:
            gid += 1
        groups.append(gid)
        prev_prefix, prev_num = prefix, num
    return groups
