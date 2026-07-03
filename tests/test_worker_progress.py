"""Tests for the GUI worker's stage-weighted overall progress (pure logic,
no Qt/PyQt6 import needed for this specific function)."""

from __future__ import annotations

from whispersync.gui.worker import _STAGE_WEIGHTS, overall_progress


def test_stage_weights_sum_to_one() -> None:
    assert abs(sum(w for _, w in _STAGE_WEIGHTS) - 1.0) < 1e-9


def test_overall_progress_starts_at_zero() -> None:
    assert overall_progress("scanning", 0.0) == 0


def test_overall_progress_ends_at_100() -> None:
    assert overall_progress("done", 1.0) == 100


def test_overall_progress_is_monotonic_across_stage_transitions() -> None:
    # Finishing one stage at 100% must not read as LOWER than starting the
    # next stage at 0% — this is exactly the "bar jumps backwards" bug that
    # motivated weighting stages in the first place.
    stages = [name for name, _ in _STAGE_WEIGHTS]
    prev = -1
    for stage in stages:
        start_val = overall_progress(stage, 0.0)
        end_val = overall_progress(stage, 1.0)
        assert start_val >= prev
        assert end_val >= start_val
        prev = end_val


def test_overall_progress_unknown_stage_falls_back_to_raw() -> None:
    assert overall_progress("some_future_stage", 0.5) == 50
