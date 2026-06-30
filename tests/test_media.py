"""Tests for media utilities."""

from __future__ import annotations

from pathlib import Path

from whispersync.engine.media import build_atempo_chain, path_to_file_uri


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
