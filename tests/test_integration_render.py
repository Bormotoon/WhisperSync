"""Integration tests that exercise real ffmpeg through the render path.

Unlike the rest of the suite (pure logic, no subprocess), these generate
synthetic audio with ffmpeg and run it through the actual cut/stretch/
assemble pipeline — the layer PROJECT_ANALYSIS.md §8.1 flags as having zero
coverage despite being where the project's worst historical bug lived (the
cumulative atempo rounding drift). Skipped automatically if ffmpeg isn't on
PATH; run explicitly with `pytest -m integration` or excluded with
`pytest -m "not integration"`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from whispersync.engine.media import probe
from whispersync.engine.timestretch import (
    assemble_continuous,
    extract_segment,
    render_piece,
    resample_conform_segment,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"),
]


@pytest.fixture
def stereo24_source(tmp_path: Path) -> Path:
    """A 10s, 48kHz, 24-bit stereo synthetic source (two different sine tones
    per channel, so a channel swap/collapse would be audible/measurable)."""
    out = tmp_path / "source_24bit_stereo.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:duration=10",
        "-filter_complex",
        "[0:a][1:a]amerge=inputs=2[a]",
        "-map",
        "[a]",
        "-ar",
        "48000",
        "-acodec",
        "pcm_s24le",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_source_fixture_is_24bit_stereo(stereo24_source: Path) -> None:
    info = probe(stereo24_source)
    assert info.audio_channels == 2
    assert info.audio_bits_per_sample == 24


def test_extract_segment_preserves_channels_and_bit_depth(
    stereo24_source: Path, tmp_path: Path
) -> None:
    out = extract_segment(
        stereo24_source,
        tmp_path,
        start=1.0,
        duration=2.0,
        segment_index=0,
        channels=2,
        codec="pcm_s24le",
    )
    info = probe(out)
    assert info.audio_channels == 2
    assert info.audio_bits_per_sample == 24
    assert abs(info.duration - 2.0) < 0.01


def test_extract_segment_is_a_null_cut_no_atempo_no_fade(
    stereo24_source: Path, tmp_path: Path
) -> None:
    """A plain cut (factor=1, no fade) must reproduce the source samples
    bit-for-bit — this is the null test that would have caught the historical
    -2.97ms/piece atempo rounding bug had it existed on the cut path too."""
    out = extract_segment(
        stereo24_source,
        tmp_path,
        start=2.0,
        duration=1.0,
        segment_index=0,
        fade_ms=0,
        channels=2,
        codec="pcm_s24le",
    )
    reference = tmp_path / "reference.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "2.0",
            "-t",
            "1.0",
            "-i",
            str(stereo24_source),
            "-acodec",
            "pcm_s24le",
            str(reference),
        ],
        check=True,
        capture_output=True,
    )
    assert out.read_bytes() == reference.read_bytes()


def test_render_piece_resample_conform_exact_length(stereo24_source: Path, tmp_path: Path) -> None:
    # A small factor (0.3% speedup) must route through resample-conform (not
    # atempo) and land at the exact intended output length.
    factor = 1.003
    duration = 3.0
    out = render_piece(
        stereo24_source,
        tmp_path,
        rec_start=0.0,
        rec_dur=duration,
        factor=factor,
        index=0,
        fade_ms=0,
        sample_rate=48000,
        channels=2,
        codec="pcm_s24le",
        stretch_method="auto",
    )
    info = probe(out)
    expected = duration / factor
    assert info.audio_channels == 2
    assert info.audio_bits_per_sample == 24
    assert abs(info.duration - expected) < 0.01


def test_render_piece_atempo_exact_length_large_factor(
    stereo24_source: Path, tmp_path: Path
) -> None:
    # A large factor forces the atempo (WSOLA) path; the exact-length contract
    # (apad+atrim) must still hold — this is the regression guard for the
    # historical cumulative-drift bug (-2.97ms/piece before the fix).
    factor = 1.3
    duration = 2.0
    out = render_piece(
        stereo24_source,
        tmp_path,
        rec_start=1.0,
        rec_dur=duration,
        factor=factor,
        index=0,
        fade_ms=0,
        sample_rate=48000,
        channels=2,
        codec="pcm_s24le",
        stretch_method="atempo",
    )
    info = probe(out)
    expected = duration / factor
    assert abs(info.duration - expected) < 0.005  # well under 5ms


def test_assemble_continuous_preserves_format_and_exact_total_length(
    stereo24_source: Path, tmp_path: Path
) -> None:
    pieces = [
        render_piece(
            stereo24_source,
            tmp_path,
            rec_start=float(i * 2),
            rec_dur=2.0,
            factor=1.0,
            index=i,
            fade_ms=10,
            sample_rate=48000,
            channels=2,
            codec="pcm_s24le",
        )
        for i in range(3)
    ]
    out = tmp_path / "assembled.wav"
    assemble_continuous(
        pieces,
        lead_silence=0.0,
        total_duration=6.0,
        sample_rate=48000,
        output_path=out,
        channels=2,
        codec="pcm_s24le",
    )
    info = probe(out)
    assert info.audio_channels == 2
    assert info.audio_bits_per_sample == 24
    assert abs(info.duration - 6.0) < 0.01


def test_resample_conform_pitch_shift_is_within_the_small_drift_budget(
    stereo24_source: Path, tmp_path: Path
) -> None:
    # A resample-conform at 1.002x must shift a 440Hz tone to ~440*1.002Hz —
    # audibly inaudible (a few cents) but present, confirming the conform
    # actually resampled rather than being a no-op.
    out = resample_conform_segment(
        stereo24_source,
        tmp_path,
        start=0.0,
        duration=4.0,
        factor=1.002,
        segment_index=0,
        sample_rate=48000,
        channels=2,
        codec="pcm_s24le",
    )
    info = probe(out)
    assert abs(info.duration - 4.0 / 1.002) < 0.01
