"""Synchronization strategy registry.

The real per-clip planning logic lives in ``engine.pipeline.clip_pieces`` — it
reads ``strategy_id`` directly to control breakpoint density (see its
docstring). This module used to also hold a full parallel implementation
(``SyncStrategy`` subclasses with a ``plan_clip`` method each), but the
pipeline never called it beyond reading ``.name`` for a progress message: it
was dead code that drifted from clip_pieces' actual behaviour (most visibly,
the old strategy 3 "Silence Padding" promised zero pitch-shift while
clip_pieces time-stretched every phrase like strategy 4 — see
PROJECT_ANALYSIS.md §2.1) and was exercised only by tests of itself. Removed;
this module is now just the id -> (name, description) lookup used for display.
"""

from __future__ import annotations

STRATEGIES: dict[int, tuple[str, str]] = {
    1: ("Global Linear", "One tempo change for the whole clip. Best for linear clock drift."),
    2: (
        "Local Time-Stretch",
        "Per-segment tempo change between anchors. Handles non-linear drift.",
    ),
    3: (
        "Hybrid (Global + Silence)",
        "Each phrase is tempo-corrected by the clip's global K, then placed at "
        "its anchor position with silence absorbing the rest. Robust + near "
        "pitch-perfect. Recommended default.",
    ),
}


def strategy_name(strategy_id: int) -> str:
    entry = STRATEGIES.get(strategy_id)
    if entry is None:
        raise ValueError(f"Unknown strategy_id {strategy_id}. Valid ids: {list(STRATEGIES)}")
    return entry[0]


def strategy_description(strategy_id: int) -> str:
    entry = STRATEGIES.get(strategy_id)
    if entry is None:
        raise ValueError(f"Unknown strategy_id {strategy_id}. Valid ids: {list(STRATEGIES)}")
    return entry[1]
