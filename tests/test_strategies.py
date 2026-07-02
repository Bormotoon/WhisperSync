"""Tests for the strategy name/description registry.

The real per-clip planning logic lives in engine.pipeline.clip_pieces and is
tested in test_pipeline.py; this module only tests the id -> (name,
description) lookup used for display (see strategies.py's module docstring
for why the old plan_clip-based strategy classes were removed).
"""

from __future__ import annotations

import pytest

from whispersync.engine.strategies import STRATEGIES, strategy_description, strategy_name


def test_known_strategy_ids_have_names() -> None:
    for sid in (1, 2, 3):
        assert strategy_name(sid)
        assert strategy_description(sid)


def test_strategies_registry_has_exactly_three_entries() -> None:
    # The old strategy 3 ("Silence Padding") was merged into Hybrid (now id 3)
    # — see PROJECT_ANALYSIS.md §2.1. There should be no id 4 anymore.
    assert set(STRATEGIES) == {1, 2, 3}


def test_unknown_strategy_id_raises() -> None:
    with pytest.raises(ValueError):
        strategy_name(99)
    with pytest.raises(ValueError):
        strategy_description(99)
