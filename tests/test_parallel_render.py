"""Tests for parallel render plumbing: worker resolution, the sequential
render path, cancel-aware future collection for the shared render pool, and
the fork-safety rules of ``_pool_context`` (PROJECT_ANALYSIS.md §3.3, §6.4)."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from whispersync.engine import pipeline


def test_resolve_workers_auto_is_cpu_count() -> None:
    assert pipeline.resolve_workers(0) == max(1, os.cpu_count() or 1)
    assert pipeline.resolve_workers(-5) == max(1, os.cpu_count() or 1)


def test_resolve_workers_explicit() -> None:
    assert pipeline.resolve_workers(1) == 1
    assert pipeline.resolve_workers(4) == 4


# --- _render_pieces_sequential ----------------------------------------------


def test_sequential_render_preserves_order(monkeypatch) -> None:
    # Stub the actual ffmpeg render: return a path encoding the index, so we can
    # assert the returned list is ordered by piece index.
    def fake_render(input_path, output_dir, rec_start, rec_dur, factor, index, fade_ms, **kwargs):
        return Path(output_dir) / f"segment_{index:04d}.wav"

    monkeypatch.setattr(pipeline, "render_piece", fake_render)
    pieces = [(float(i), 1.0, 1.0) for i in range(6)]
    out = pipeline._render_pieces_sequential(pieces, Path("rec.wav"), Path("/tmp"), 10)
    assert [p.name for p in out] == [f"segment_{i:04d}.wav" for i in range(6)]


def test_sequential_render_empty() -> None:
    assert pipeline._render_pieces_sequential([], Path("rec.wav"), Path("/tmp"), 10) == []


def test_sequential_render_checks_cancel_between_pieces(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_render(input_path, output_dir, rec_start, rec_dur, factor, index, fade_ms, **kwargs):
        calls["n"] += 1
        return Path(output_dir) / f"segment_{index:04d}.wav"

    monkeypatch.setattr(pipeline, "render_piece", fake_render)
    pieces = [(float(i), 1.0, 1.0) for i in range(10)]

    # Already-cancelled event -> the very first piece must raise before
    # rendering anything, not after finishing the whole job.
    cancel_event = threading.Event()
    cancel_event.set()
    try:
        pipeline._render_pieces_sequential(
            pieces, Path("rec.wav"), Path("/tmp"), 10, cancel_event=cancel_event
        )
        raise AssertionError("expected InterruptedError")
    except InterruptedError:
        pass
    assert calls["n"] == 0


def test_sequential_render_no_cancel_event_runs_to_completion(monkeypatch) -> None:
    def fake_render(input_path, output_dir, rec_start, rec_dur, factor, index, fade_ms, **kwargs):
        return Path(output_dir) / f"segment_{index:04d}.wav"

    monkeypatch.setattr(pipeline, "render_piece", fake_render)
    pieces = [(float(i), 1.0, 1.0) for i in range(5)]
    out = pipeline._render_pieces_sequential(pieces, Path("rec.wav"), Path("/tmp"), 10)
    assert len(out) == 5


# --- _collect_piece_futures (shared-pool path) --------------------------------


def test_collect_futures_returns_results_in_submit_order() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(lambda i=i: Path(f"segment_{i:04d}.wav")) for i in range(5)]
        out = pipeline._collect_piece_futures(futs)
    assert [p.name for p in out] == [f"segment_{i:04d}.wav" for i in range(5)]


def test_collect_futures_pre_cancelled_raises_immediately() -> None:
    cancel_event = threading.Event()
    cancel_event.set()
    fut: Future = Future()  # never resolved — collection must not wait on it
    try:
        pipeline._collect_piece_futures([fut], cancel_event)
        raise AssertionError("expected InterruptedError")
    except InterruptedError:
        pass


def test_collect_futures_cancel_mid_wait_interrupts_promptly() -> None:
    # The old per-job pool collected futures with a blocking .result() and no
    # cancel check — cancelling during a big multi-core job waited for the
    # whole job. The poll loop must notice the event within ~poll_s instead.
    cancel_event = threading.Event()
    fut: Future = Future()  # a piece that never finishes

    def cancel_soon() -> None:
        time.sleep(0.15)
        cancel_event.set()

    t = threading.Thread(target=cancel_soon)
    t.start()
    start = time.monotonic()
    try:
        pipeline._collect_piece_futures([fut], cancel_event, poll_s=0.05)
        raise AssertionError("expected InterruptedError")
    except InterruptedError:
        pass
    finally:
        t.join()
    assert time.monotonic() - start < 2.0


def test_collect_futures_propagates_worker_exception() -> None:
    fut: Future = Future()
    fut.set_exception(RuntimeError("ffmpeg failed"))
    try:
        pipeline._collect_piece_futures([fut])
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


# --- _pool_context fork safety -------------------------------------------------


def test_pool_context_never_forks_in_a_multithreaded_process() -> None:
    # This test runs under pytest with at least the main thread; spin up an
    # extra one to be certain the process is multi-threaded, mirroring the GUI
    # (Qt main thread + worker QThread). Forking such a process is the classic
    # deadlock the plan's §3.3 flagged — the chosen method must not be fork.
    stop = threading.Event()
    t = threading.Thread(target=stop.wait)
    t.start()
    try:
        assert threading.active_count() > 1
        ctx = pipeline._pool_context()
        assert ctx.get_start_method() != "fork"
    finally:
        stop.set()
        t.join()


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
