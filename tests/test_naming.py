"""Tests for filename natural ordering."""

from __future__ import annotations

from whispersync.engine.naming import natural_key


def test_natural_sort_orders_numbers_numerically() -> None:
    names = ["DJI_9.mov", "DJI_10.mov", "DJI_100.mov", "DJI_2.mov"]
    ordered = sorted(names, key=natural_key)
    assert ordered == ["DJI_2.mov", "DJI_9.mov", "DJI_10.mov", "DJI_100.mov"]
