"""Tests for parallel render plumbing (worker resolution + order preservation)."""

from __future__ import annotations

import os
from pathlib import Path

from whispersync.engine import pipeline


def test_resolve_workers_auto_is_cpu_count() -> None:
    assert pipeline.resolve_workers(0) == max(1, os.cpu_count() or 1)
    assert pipeline.resolve_workers(-5) == max(1, os.cpu_count() or 1)


def test_resolve_workers_explicit() -> None:
    assert pipeline.resolve_workers(1) == 1
    assert pipeline.resolve_workers(4) == 4


def test_render_pieces_preserves_order(monkeypatch) -> None:
    # Stub the actual ffmpeg render: return a path encoding the index, so we can
    # assert the returned list is ordered by piece index regardless of scheduling.
    def fake_render(input_path, output_dir, rec_start, rec_dur, factor, index, fade_ms):
        return Path(output_dir) / f"segment_{index:04d}.wav"

    monkeypatch.setattr(pipeline, "render_piece", fake_render)
    pieces = [(float(i), 1.0, 1.0) for i in range(6)]
    # workers=1 takes the sequential path (no pool); order must match input.
    out = pipeline.render_pieces(pieces, Path("rec.wav"), Path("/tmp"), 10, workers=1)
    assert [p.name for p in out] == [f"segment_{i:04d}.wav" for i in range(6)]


def test_render_pieces_empty() -> None:
    assert pipeline.render_pieces([], Path("rec.wav"), Path("/tmp"), 10, workers=4) == []
