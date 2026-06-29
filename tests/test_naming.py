"""Tests for filename ordering and consecutive-sequence detection."""

from __future__ import annotations

from bormosync.engine.naming import group_sequences, natural_key, split_index


def test_natural_sort_orders_numbers_numerically() -> None:
    names = ["DJI_9.mov", "DJI_10.mov", "DJI_100.mov", "DJI_2.mov"]
    ordered = sorted(names, key=natural_key)
    assert ordered == ["DJI_2.mov", "DJI_9.mov", "DJI_10.mov", "DJI_100.mov"]


def test_split_index() -> None:
    assert split_index("DJI_0839.mov") == ("DJI_", 839)
    assert split_index("clip.mov") == ("clip.mov", None)
    # last numeric run is used
    assert split_index("C0001_take2") == ("C0001_take", 2)


def test_group_sequences_consecutive_same_group() -> None:
    names = ["DJI_0838.mov", "DJI_0839.mov", "DJI_0840.mov"]
    assert group_sequences(names) == [0, 0, 0]


def test_group_sequences_gap_starts_new_group() -> None:
    names = ["DJI_0838.mov", "DJI_0839.mov", "DJI_0845.mov", "DJI_0846.mov"]
    assert group_sequences(names) == [0, 0, 1, 1]


def test_group_sequences_prefix_change_starts_new_group() -> None:
    names = ["DJI_0001.mov", "DJI_0002.mov", "GX_0003.mov"]
    # GX_0003 has a consecutive number but a different prefix -> new group
    assert group_sequences(names) == [0, 0, 1]
