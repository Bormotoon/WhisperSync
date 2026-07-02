"""Tests for atempo chain decomposition."""

from __future__ import annotations

from whispersync.engine.media import build_atempo_chain
from whispersync.engine.timestretch import (
    RESAMPLE_CONFORM_MAX_DEVIATION,
    duck_filter_chain,
    edge_fade_filters,
    seam_fade_filters,
)


def _eval_chain(chain: list[str]) -> float:
    product = 1.0
    for f in chain:
        val = float(f.replace("atempo=", ""))
        product *= val
    return product


def test_atempo_chain_identity() -> None:
    chain = build_atempo_chain(1.0)
    assert len(chain) == 1
    product = _eval_chain(chain)
    assert abs(product - 1.0) < 1e-4


def test_atempo_chain_normal_range() -> None:
    for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
        chain = build_atempo_chain(factor)
        product = _eval_chain(chain)
        assert abs(product - factor) < 1e-4, f"factor={factor}, product={product}"
        for f in chain:
            val = float(f.replace("atempo=", ""))
            assert 0.5 <= val <= 2.0, f"filter {f} out of range"


def test_atempo_chain_extreme_slow() -> None:
    factor = 0.1
    chain = build_atempo_chain(factor)
    product = _eval_chain(chain)
    assert abs(product - factor) < 1e-3
    for f in chain:
        val = float(f.replace("atempo=", ""))
        assert 0.5 <= val <= 2.0


def test_atempo_chain_extreme_fast() -> None:
    factor = 8.0
    chain = build_atempo_chain(factor)
    product = _eval_chain(chain)
    assert abs(product - factor) < 1e-2
    for f in chain:
        val = float(f.replace("atempo=", ""))
        assert 0.5 <= val <= 2.0


def test_atempo_chain_typical_drift() -> None:
    for k in [0.9995, 0.999, 1.001, 1.005]:
        factor = 1.0 / k
        chain = build_atempo_chain(factor)
        product = _eval_chain(chain)
        assert abs(product - factor) < 1e-4


def test_edge_fade_disabled_when_zero() -> None:
    assert edge_fade_filters(10.0, 0) == []
    assert edge_fade_filters(0.0, 10) == []


def test_edge_fade_in_and_out() -> None:
    filters = edge_fade_filters(2.0, 10)
    assert len(filters) == 2
    assert filters[0].startswith("afade=t=in:st=0:")
    assert "afade=t=out:st=1.99" in filters[1]  # 2.0 - 0.01


def test_edge_fade_clamped_to_half_segment() -> None:
    # a 100ms fade on a 0.1s segment must clamp to <= half (0.05s)
    filters = edge_fade_filters(0.1, 100)
    assert len(filters) == 2
    in_dur = float(filters[0].split("d=")[1])
    assert in_dur <= 0.05 + 1e-9


# --- seam_fade_filters -------------------------------------------------------


def test_seam_fade_filters_both_edges() -> None:
    filters = seam_fade_filters(2.0, 10, fade_in=True, fade_out=True)
    assert len(filters) == 2
    assert filters[0].startswith("afade=t=in:")
    assert filters[1].startswith("afade=t=out:")


def test_seam_fade_filters_in_only() -> None:
    filters = seam_fade_filters(2.0, 10, fade_in=True, fade_out=False)
    assert len(filters) == 1
    assert filters[0].startswith("afade=t=in:")


def test_seam_fade_filters_out_only() -> None:
    filters = seam_fade_filters(2.0, 10, fade_in=False, fade_out=True)
    assert len(filters) == 1
    assert filters[0].startswith("afade=t=out:")


def test_seam_fade_filters_neither_edge_is_a_noop() -> None:
    # A contiguous interior seam needs no fade at all — this is what stops the
    # per-seam volume dip described in PROJECT_ANALYSIS.md §2.0.
    assert seam_fade_filters(2.0, 10, fade_in=False, fade_out=False) == []


def test_seam_fade_filters_disabled_when_zero() -> None:
    assert seam_fade_filters(10.0, 0, True, True) == []


# --- duck_filter_chain --------------------------------------------------------


def test_duck_filter_chain_none_when_no_spans() -> None:
    assert duck_filter_chain([], -18.0, 80) is None


def test_duck_filter_chain_none_when_disabled() -> None:
    assert duck_filter_chain([(1.0, 2.0)], 0.0, 80) is None


def test_duck_filter_chain_builds_volume_expr() -> None:
    chain = duck_filter_chain([(1.0, 2.0)], -18.0, 80)
    assert chain is not None
    assert chain.startswith("volume=volume=")


# --- RESAMPLE_CONFORM_MAX_DEVIATION -------------------------------------------


def test_resample_conform_threshold_covers_typical_clock_drift() -> None:
    # Real camera/recorder clock drift is well under 1%; the resample-conform
    # threshold must cover it so atempo/WSOLA artifacts are avoided for the
    # common case (PROJECT_ANALYSIS.md §2.0).
    assert RESAMPLE_CONFORM_MAX_DEVIATION >= 0.003


# --- render_piece stretch-method routing --------------------------------------


def test_render_piece_routes_small_factor_to_resample(monkeypatch) -> None:
    from pathlib import Path

    from whispersync.engine import timestretch

    calls: list[str] = []
    monkeypatch.setattr(
        timestretch,
        "resample_conform_segment",
        lambda *a, **k: calls.append("resample") or Path("out.wav"),
    )
    monkeypatch.setattr(
        timestretch, "apply_atempo_segment", lambda *a, **k: calls.append("atempo") or Path("x")
    )
    timestretch.render_piece(
        Path("in.wav"), Path("."), 0.0, 1.0, 1.002, 0, fade_ms=10, stretch_method="auto"
    )
    assert calls == ["resample"]


def test_render_piece_routes_large_factor_to_atempo(monkeypatch) -> None:
    from pathlib import Path

    from whispersync.engine import timestretch

    calls: list[str] = []
    monkeypatch.setattr(
        timestretch,
        "resample_conform_segment",
        lambda *a, **k: calls.append("resample") or Path("x"),
    )
    monkeypatch.setattr(
        timestretch,
        "apply_atempo_segment",
        lambda *a, **k: calls.append("atempo") or Path("out.wav"),
    )
    timestretch.render_piece(
        Path("in.wav"), Path("."), 0.0, 1.0, 1.3, 0, fade_ms=10, stretch_method="auto"
    )
    assert calls == ["atempo"]


def test_render_piece_unchanged_factor_is_a_plain_cut(monkeypatch) -> None:
    from pathlib import Path

    from whispersync.engine import timestretch

    calls: list[str] = []
    monkeypatch.setattr(
        timestretch, "extract_segment", lambda *a, **k: calls.append("cut") or Path("out.wav")
    )
    timestretch.render_piece(Path("in.wav"), Path("."), 0.0, 1.0, 1.0, 0, fade_ms=10)
    assert calls == ["cut"]


def test_atempo_segment_filter_locks_exact_length() -> None:
    # Regression guard: apply_atempo_segment must hard-trim each piece to its exact
    # intended output length (duration/factor). atempo otherwise outputs ~3ms short
    # per piece, which accumulates into seconds of A/V drift across the hundreds of
    # pieces the per-phrase / per-anchor strategies produce. We assert the filter
    # chain ends with apad + atrim to the exact out_duration.
    import inspect

    from whispersync.engine import timestretch

    src = inspect.getsource(timestretch.apply_atempo_segment)
    assert "apad" in src and "atrim=0:" in src, (
        "apply_atempo_segment must pad+trim each piece to an exact length so "
        "concatenated pieces do not accumulate atempo rounding drift"
    )
