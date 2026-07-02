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
    def fake_render(input_path, output_dir, rec_start, rec_dur, factor, index, fade_ms, **kwargs):
        return Path(output_dir) / f"segment_{index:04d}.wav"

    monkeypatch.setattr(pipeline, "render_piece", fake_render)
    pieces = [(float(i), 1.0, 1.0) for i in range(6)]
    # workers=1 takes the sequential path (no pool); order must match input.
    out = pipeline.render_pieces(pieces, Path("rec.wav"), Path("/tmp"), 10, workers=1)
    assert [p.name for p in out] == [f"segment_{i:04d}.wav" for i in range(6)]


def test_render_pieces_empty() -> None:
    assert pipeline.render_pieces([], Path("rec.wav"), Path("/tmp"), 10, workers=4) == []


# --- _piece_seam_fades -------------------------------------------------------


def test_seam_fades_contiguous_pieces_only_fade_clip_edges() -> None:
    # clip_pieces' normal output: pieces tile the recorder span with no gaps.
    # Interior seams are acoustically continuous and must NOT get a fade (a fade
    # there would carve an audible dip into otherwise-continuous audio — see
    # PROJECT_ANALYSIS.md §2.0); only the outer clip boundaries fade.
    pieces = [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)]
    flags = pipeline._piece_seam_fades(pieces)
    assert flags == [(True, False), (False, False), (False, True)]


def test_seam_fades_discontinuity_fades_both_sides() -> None:
    # After Boundary Flex nudges a boundary, a piece may no longer start exactly
    # where its neighbour ended — that seam is now a real discontinuity and both
    # sides must fade.
    pieces = [(0.0, 1.0, 1.0), (1.08, 1.0, 1.0), (2.0, 1.0, 1.0)]
    flags = pipeline._piece_seam_fades(pieces)
    assert flags[0] == (True, True)  # piece 0's end no longer meets piece 1's start
    assert flags[1] == (True, True)  # piece 1's start/end both discontinuous
    assert flags[2] == (True, True)  # piece 2's start no longer meets piece 1's end


def test_seam_fades_single_piece_fades_both_edges() -> None:
    assert pipeline._piece_seam_fades([(0.0, 1.0, 1.0)]) == [(True, True)]
