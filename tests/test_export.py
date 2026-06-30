"""Tests for FCPXML generation and validation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from whispersync.engine.export import (
    fps_to_frame_duration,
    generate_fcpxml,
    to_rational,
    validate_fcpxml,
)
from whispersync.engine.media import MediaInfo
from whispersync.models import MediaClip, SyncPlan


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

    # New layout: the video clip is the primary-storyline element (directly in the
    # spine, no lane), and the audio is a connected clip nested under it on lane -1.
    spine_video = spine.findall("asset-clip")
    assert len(spine_video) == 1, "the video clip should sit directly in the spine"
    video_clip = spine_video[0]
    assert video_clip.get("lane") is None, "primary-storyline clip carries no lane"

    connected = video_clip.findall("asset-clip")
    assert len(connected) == 1, "the audio should be connected to the video clip"
    assert connected[0].get("lane") == "-1"

    # the audio's offset is relative to the parent's start (2s here)
    clips = [video_clip, connected[0]]

    # every asset-clip must reference a declared asset
    asset_ids = {a.get("id") for a in root.findall(".//asset")}
    for c in clips:
        assert c.get("ref") in asset_ids

    # FCPXML 1.9+ DTD: the file reference must live on <media-rep src=...>, never
    # as a `src` attribute on <asset> (Final Cut rejects the latter on import).
    for asset in root.findall(".//asset"):
        assert asset.get("src") is None, "asset must not carry a src attribute"
        rep = asset.find("media-rep")
        assert rep is not None and rep.get("src"), "asset needs a <media-rep src=...>"


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


def test_spine_times_are_frame_aligned(tmp_path: Path) -> None:
    """FCP rejects spine offsets/durations that are not on a frame boundary, and
    warns about a custom format name. Both must be avoided."""
    video_info = MediaInfo(
        path=Path("/v/DJI_0829.mp4"),
        duration=60.04,
        fps=Fraction(30000, 1001),  # 29.97 fps -> one frame = 1001/30000 s
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48000,
    )
    # deliberately non-frame-aligned offsets/durations
    plan = SyncPlan(
        strategy_id=1,
        clips=[
            MediaClip(Path("/v/DJI_0829.mp4"), "video", 2.9106329, 0.0, 60.04, 1),
            MediaClip(Path("/a/synced_000.wav"), "audio", 2.9106329, 0.0, 60.04, -1),
        ],
        audio_ops=[],
        total_duration=62.95,
    )
    out = tmp_path / "fa.fcpxml"
    generate_fcpxml(plan, [video_info], out, audio_sample_rate=48000)

    root = ET.parse(out).getroot()

    fmt = root.find(".//format")
    assert fmt is not None and "name" not in fmt.attrib  # custom name -> FCP warns

    for ac in root.findall(".//asset-clip"):
        for attr in ("offset", "duration", "start"):
            ticks_s, den_s = ac.get(attr, "0/30000s")[:-1].split("/")
            assert int(den_s) == 30000
            assert int(ticks_s) % 1001 == 0, f"{attr} not on a frame boundary: {ac.get(attr)}"


def test_mixed_fps_and_relative_src(tmp_path: Path) -> None:
    """Cameras at different rates each get a format at their native fps, and media
    living next to the FCPXML is referenced by a relative path."""
    (tmp_path / "audio_synced").mkdir()
    synced = tmp_path / "audio_synced" / "synced_000.wav"
    synced.write_bytes(b"RIFF")

    a = MediaInfo(
        Path("/v/A.mp4"), 10.0, Fraction(30000, 1001), 1920, 1080, "h264", "aac", 2, 48000
    )
    b = MediaInfo(Path("/v/B.mp4"), 10.0, Fraction(30, 1), 3840, 2160, "h264", "aac", 2, 48000)
    plan = SyncPlan(
        strategy_id=1,
        clips=[
            MediaClip(Path("/v/A.mp4"), "video", 0.0, 0.0, 10.0, 1),
            MediaClip(Path("/v/B.mp4"), "video", 12.0, 0.0, 10.0, 2),
            MediaClip(synced, "audio", 0.0, 0.0, 10.0, -1),
        ],
        audio_ops=[],
        total_duration=22.0,
    )
    out = tmp_path / "sync_output.fcpxml"
    generate_fcpxml(plan, [a, b], out, audio_sample_rate=48000)
    root = ET.parse(out).getroot()

    frame_durs = {f.get("frameDuration") for f in root.findall(".//format")}
    assert "1001/30000s" in frame_durs and "1/30s" in frame_durs  # both native rates

    # co-located synced audio -> relative; source videos -> absolute file://
    srcs = {a.get("name"): a.find("media-rep").get("src") for a in root.findall(".//asset")}
    assert srcs["synced_000"] == "audio_synced/synced_000.wav"
    assert srcs["A"].startswith("file://")
