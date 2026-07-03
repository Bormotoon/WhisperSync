"""Tests for tools/verify_sync.py (pure logic — no ffmpeg)."""

from __future__ import annotations

from tools.verify_sync import LagSample, VerifyReport


def test_summary_empty_report() -> None:
    report = VerifyReport(video="v", voice="a")
    summary = report.summary()
    assert summary == {"n_confident": 0, "n_total": 0}


def test_summary_computes_median_p90_max() -> None:
    report = VerifyReport(
        video="v",
        voice="a",
        samples=[
            LagSample(t=0.0, lag_ms=10.0, sharpness=100.0),
            LagSample(t=1.0, lag_ms=-20.0, sharpness=100.0),
            LagSample(t=2.0, lag_ms=5.0, sharpness=100.0),
            LagSample(t=3.0, lag_ms=-50.0, sharpness=100.0),
        ],
    )
    summary = report.summary()
    assert summary["n_confident"] == 4
    assert summary["n_total"] == 4
    # abs lags sorted: [5, 10, 20, 50] -> median index 2 -> 20
    assert summary["median_abs_lag_ms"] == 20.0
    assert summary["max_abs_lag_ms"] == 50.0


def test_confident_property_returns_all_stored_samples() -> None:
    # measure() only appends samples that already passed the sharpness gate,
    # so .confident is simply .samples — this just locks that contract in.
    samples = [LagSample(t=0.0, lag_ms=1.0, sharpness=200.0)]
    report = VerifyReport(video="v", voice="a", samples=samples)
    assert report.confident == samples
