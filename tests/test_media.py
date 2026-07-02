"""Tests for media utilities."""

from __future__ import annotations

from pathlib import Path

from whispersync.engine.media import build_atempo_chain, path_to_file_uri, pcm_codec_for_bit_depth


def test_pcm_codec_16bit_source() -> None:
    assert pcm_codec_for_bit_depth(16) == "pcm_s16le"


def test_pcm_codec_8bit_source_still_16bit_floor() -> None:
    # Nothing below 16-bit is worth preserving as-is; floor at 16-bit PCM.
    assert pcm_codec_for_bit_depth(8) == "pcm_s16le"


def test_pcm_codec_24bit_source() -> None:
    assert pcm_codec_for_bit_depth(24) == "pcm_s24le"


def test_pcm_codec_32bit_source() -> None:
    assert pcm_codec_for_bit_depth(32) == "pcm_s32le"


def test_pcm_codec_unknown_depth_defaults_to_24bit() -> None:
    # Unknown depth (e.g. a lossy source codec) defaults to 24-bit, which covers
    # the vast majority of professional recorders without truncation.
    assert pcm_codec_for_bit_depth(None) == "pcm_s24le"


def test_path_to_file_uri() -> None:
    p = Path("/home/user/videos/test file.mp4")
    uri = path_to_file_uri(p)
    assert uri.startswith("file://")
    assert "test%20file.mp4" in uri


def test_path_to_file_uri_no_spaces() -> None:
    p = Path("/home/user/videos/test.mp4")
    uri = path_to_file_uri(p)
    assert uri == f"file://{p.resolve()}"


def test_build_atempo_chain_in_range() -> None:
    chain = build_atempo_chain(1.5)
    assert len(chain) == 1
    assert "1.5" in chain[0]


def test_build_atempo_chain_needs_decomposition() -> None:
    chain = build_atempo_chain(4.0)
    assert len(chain) >= 2
    product = 1.0
    for f in chain:
        val = float(f.replace("atempo=", ""))
        assert 0.5 <= val <= 2.0
        product *= val
    assert abs(product - 4.0) < 0.01
