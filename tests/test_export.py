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
from whispersync.models import MediaClip, SubClip, SyncPlan


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


def _vinfo(path: str, dur: float) -> MediaInfo:
    return MediaInfo(
        path=Path(path),
        duration=dur,
        fps=Fraction(30000, 1001),
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48000,
    )


def test_display_names_and_roles(tmp_path: Path) -> None:
    clips = [
        MediaClip(
            path=Path("/v/DJI_0830.MOV"),
            kind="video",
            offset=0.0,
            in_point=0.0,
            duration=10.0,
            lane=1,
            role="Video",
        ),
        MediaClip(
            path=Path("/a/synced_001.wav"),
            kind="audio",
            offset=0.0,
            in_point=0.0,
            duration=10.0,
            lane=-1,
            display_name="DJI_0830_voice",
            role="Dialogue",
        ),
        MediaClip(
            path=Path("/a/blah_(Instrumental)_melband.wav"),
            kind="audio",
            offset=0.0,
            in_point=0.0,
            duration=10.0,
            lane=-2,
            display_name="DJI_0830_ambience",
            role="Effects",
        ),
    ]
    plan = SyncPlan(strategy_id=4, clips=clips, audio_ops=[], total_duration=10.0)
    out = tmp_path / "roles.fcpxml"
    generate_fcpxml(plan, [_vinfo("/v/DJI_0830.MOV", 10.0)], out, audio_sample_rate=48000)
    root = ET.parse(out).getroot()

    by_name = {c.get("name"): c for c in root.findall(".//asset-clip")}
    assert set(by_name) == {"DJI_0830", "DJI_0830_voice", "DJI_0830_ambience"}
    # roles land on the right attribute (videoRole vs audioRole)
    assert by_name["DJI_0830"].get("videoRole") == "Video"
    assert by_name["DJI_0830_voice"].get("audioRole") == "Dialogue"
    assert by_name["DJI_0830_ambience"].get("audioRole") == "Effects"
    # friendly names also propagate to the <asset> entries
    asset_names = {a.get("name") for a in root.findall(".//asset")}
    assert "DJI_0830_voice" in asset_names and "DJI_0830_ambience" in asset_names


def test_no_role_omits_attribute(tmp_path: Path) -> None:
    clips = [
        MediaClip(
            path=Path("/v/a.mov"),
            kind="video",
            offset=0.0,
            in_point=0.0,
            duration=5.0,
            lane=1,
        )
    ]
    plan = SyncPlan(strategy_id=1, clips=clips, audio_ops=[], total_duration=5.0)
    out = tmp_path / "norole.fcpxml"
    generate_fcpxml(plan, [_vinfo("/v/a.mov", 5.0)], out)
    clip = ET.parse(out).getroot().find(".//asset-clip")
    assert clip is not None
    assert clip.get("videoRole") is None and clip.get("audioRole") is None
    assert clip.get("name") == "a"  # falls back to stem


def test_compound_clip_emits_media_and_ref_clip(tmp_path: Path) -> None:
    """audio_compound: pieces become their own assets inside a <media>'s nested
    sequence/spine (sized to the video), referenced from the main spine by a
    <ref-clip> instead of a plain <asset-clip>."""
    pieces_dir = tmp_path / "audio_synced" / "DJI_0830_voice_pieces"
    pieces_dir.mkdir(parents=True)
    p0 = pieces_dir / "segment_0000.wav"
    p1 = pieces_dir / "segment_0001.wav"
    p0.write_bytes(b"RIFF")
    p1.write_bytes(b"RIFF")

    clips = [
        MediaClip(
            path=Path("/v/DJI_0830.MOV"),
            kind="video",
            offset=0.0,
            in_point=0.0,
            duration=10.0,
            lane=1,
            role="Video",
        ),
        MediaClip(
            path=tmp_path / "audio_synced" / "DJI_0830_voice.wav",  # nominal, never written
            kind="audio",
            offset=0.0,
            in_point=0.0,
            duration=10.0,
            lane=-1,
            display_name="DJI_0830_voice",
            role="Dialogue",
            subclips=[
                SubClip(path=p0, offset=0.5, in_point=2.0, duration=3.0),
                SubClip(path=p1, offset=6.0, in_point=8.0, duration=2.0),
            ],
        ),
    ]
    plan = SyncPlan(strategy_id=1, clips=clips, audio_ops=[], total_duration=10.0)
    out = tmp_path / "compound.fcpxml"
    generate_fcpxml(plan, [_vinfo("/v/DJI_0830.MOV", 10.0)], out, audio_sample_rate=48000)
    assert validate_fcpxml(out)

    root = ET.parse(out).getroot()

    # One <media> resource, named after the clip, holding a nested sequence/spine
    # sized to the full (video) duration with the two pieces + gaps between them.
    media_els = root.findall("./resources/media")
    assert len(media_els) == 1
    media = media_els[0]
    assert media.get("name") == "DJI_0830_voice"
    inner_seq = media.find("sequence")
    assert inner_seq is not None
    inner_clips = inner_seq.findall(".//asset-clip")
    assert len(inner_clips) == 2
    inner_gaps = inner_seq.findall(".//gap")
    assert len(inner_gaps) == 2  # before piece 0 (0->0.5) and between pieces (3.5->6.0)

    # Each piece is its own <asset> (not folded into one file).
    piece_asset_names = {a.get("name") for a in root.findall("./resources/asset")}
    assert {"segment_0000", "segment_0001"} <= piece_asset_names

    # The main project spine references the compound via <ref-clip>, not <asset-clip>.
    library = root.find("library")
    assert library is not None
    project_spine = library.find(".//spine")
    assert project_spine is not None
    assert project_spine.findall(".//ref-clip"), "compound clip must appear as ref-clip"
    ref_clip = project_spine.find(".//ref-clip")
    assert ref_clip is not None
    assert ref_clip.get("ref") == media.get("id")
    assert ref_clip.get("name") == "DJI_0830_voice"
    assert ref_clip.get("lane") == "-1"
    # sized to the video clip, not to the sum of the pieces
    ticks, den = ref_clip.get("duration", "0/1s")[:-1].split("/")
    assert abs(int(ticks) / int(den) - 10.0) < 0.05
    # audioRole/videoRole are NOT declared for ref-clip in the FCPXML DTD — Final
    # Cut refuses to import a document with either attribute set here (even though
    # clip.role == "Dialogue" above), so they must be omitted despite the role
    # being set on the MediaClip.
    assert ref_clip.get("audioRole") is None
    assert ref_clip.get("videoRole") is None

    # No stray <asset-clip> was created for the compound clip's own nominal path.
    top_level_asset_srcs = {
        a.find("media-rep").get("src") for a in root.findall("./resources/asset")
    }
    assert not any("DJI_0830_voice.wav" in src for src in top_level_asset_srcs)


def test_validate_fcpxml_ignores_nested_compound_spine(tmp_path: Path) -> None:
    """The document's own (project) spine must be validated, never a compound
    clip's nested one living under <resources> earlier in document order."""
    piece = tmp_path / "segment_0000.wav"
    piece.write_bytes(b"RIFF")
    clips = [
        MediaClip(Path("/v/a.mov"), "video", 0.0, 0.0, 5.0, 1),
        MediaClip(
            tmp_path / "a_voice.wav",
            "audio",
            0.0,
            0.0,
            5.0,
            -1,
            display_name="a_voice",
            subclips=[SubClip(path=piece, offset=0.0, in_point=0.0, duration=5.0)],
        ),
    ]
    plan = SyncPlan(strategy_id=1, clips=clips, audio_ops=[], total_duration=5.0)
    out = tmp_path / "nested.fcpxml"
    generate_fcpxml(plan, [_vinfo("/v/a.mov", 5.0)], out)
    assert validate_fcpxml(out)


def test_validate_fcpxml_rejects_role_attrs_on_ref_clip(tmp_path: Path) -> None:
    """Defense in depth: even if some future change reintroduces audioRole/
    videoRole on a <ref-clip> (which generate_fcpxml itself no longer does),
    validate_fcpxml must catch it — this is exactly the attribute Final Cut's DTD
    rejects with 'No declaration for attribute audioRole of element ref-clip'."""
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n"
        '<fcpxml version="1.9">'
        "<resources>"
        '<media id="r1" name="voice"><sequence format="r0" duration="1/1s">'
        "<spine/>"
        "</sequence></media>"
        "</resources>"
        '<library><event><project><sequence format="r0" duration="1/1s">'
        '<spine><asset-clip ref="r2" name="v" duration="1/1s">'
        '<ref-clip ref="r1" name="voice" lane="-1" duration="1/1s" audioRole="Dialogue"/>'
        "</asset-clip></spine>"
        "</sequence></project></event></library>"
        "</fcpxml>"
    )
    out = tmp_path / "bad_role.fcpxml"
    out.write_text(doc)
    assert not validate_fcpxml(out)
