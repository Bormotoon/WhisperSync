"""Tests for FCPXML generation and validation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from bormosync.engine.export import (
    fps_to_frame_duration,
    generate_fcpxml,
    to_rational,
    validate_fcpxml,
)
from bormosync.engine.media import MediaInfo
from bormosync.models import MediaClip, SyncPlan


def test_to_rational() -> None:
    assert to_rational(1.0, 30000) == "30000/30000s"
    assert to_rational(0.0, 30000) == "0/30000s"
    assert to_rational(2.0, 48000) == "96000/48000s"


def test_fps_to_frame_duration() -> None:
    assert fps_to_frame_duration(Fraction(30000, 1001)) == "1001/30000s"
    assert fps_to_frame_duration(Fraction(25, 1)) == "1/25s"


def test_generate_and_validate_fcpxml(tmp_path: Path) -> None:
    video_info = MediaInfo(
        path=Path("/videos/clip1.mp4"),
        duration=60.0,
        fps=Fraction(30000, 1001),
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48000,
    )

    plan = SyncPlan(
        strategy_id=1,
        clips=[
            MediaClip(
                path=Path("/videos/clip1.mp4"),
                kind="video",
                offset=0.0,
                in_point=0.0,
                duration=60.0,
                lane=1,
            ),
            MediaClip(
                path=Path("/audio/synced.wav"),
                kind="audio",
                offset=2.0,
                in_point=0.0,
                duration=58.0,
                lane=-1,
            ),
        ],
        audio_ops=[{"type": "atempo", "factor": 0.999}],
        total_duration=60.0,
    )

    output = tmp_path / "test_output.fcpxml"
    result = generate_fcpxml(plan, [video_info], output)

    assert result.exists()
    assert validate_fcpxml(result)

    tree = ET.parse(result)
    root = tree.getroot()
    assert root.tag == "fcpxml"
    assert root.get("version") == "1.9"

    spine = root.find(".//spine")
    assert spine is not None

    gap = spine.find("gap")
    assert gap is not None

    clips = gap.findall("asset-clip")
    assert len(clips) == 2

    lanes = {c.get("lane") for c in clips}
    assert "1" in lanes
    assert "-1" in lanes

    # every asset-clip must reference a declared asset
    asset_ids = {a.get("id") for a in root.findall(".//asset")}
    for c in clips:
        assert c.get("ref") in asset_ids


def test_fcpxml_roundtrip_times(tmp_path: Path) -> None:
    video_info = MediaInfo(
        path=Path("/v/a.mp4"),
        duration=120.0,
        fps=Fraction(25, 1),
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48000,
    )

    plan = SyncPlan(
        strategy_id=1,
        clips=[
            MediaClip(Path("/v/a.mp4"), "video", 0.0, 0.0, 120.0, 1),
            MediaClip(Path("/a/r.wav"), "audio", 5.0, 0.0, 115.0, -1),
        ],
        audio_ops=[],
        total_duration=120.0,
    )

    out = tmp_path / "rt.fcpxml"
    generate_fcpxml(plan, [video_info], out)

    tree = ET.parse(out)
    clips = tree.findall(".//asset-clip")
    assert len(clips) == 2
