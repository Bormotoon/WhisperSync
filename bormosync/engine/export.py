"""FCPXML generator for Final Cut Pro / DaVinci Resolve."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from bormosync.engine.media import MediaInfo, path_to_file_uri
from bormosync.models import SyncPlan

logger = logging.getLogger(__name__)


def to_rational(seconds: float, timebase: int) -> str:
    ticks = round(seconds * timebase)
    return f"{ticks}/{timebase}s"


def fps_to_frame_duration(fps: Fraction) -> str:
    return f"{fps.denominator}/{fps.numerator}s"


def _frame_rational(seconds: float, fps: Fraction, mode: str = "round") -> str:
    """Express ``seconds`` on the sequence timebase, snapped to a whole frame.

    Final Cut requires spine offsets/durations to land on an edit-frame boundary
    (a multiple of the frame duration), otherwise it warns and re-quantises. We
    round offsets, floor clip durations (never claim more media than exists) and
    ceil the sequence/gap duration (so it covers every clip).
    """
    frames_f = seconds * float(fps)
    if mode == "floor":
        frames = int(frames_f)
    elif mode == "ceil":
        frames = -int(-frames_f // 1)
    else:
        frames = round(frames_f)
    ticks = frames * fps.denominator
    return f"{ticks}/{fps.numerator}s"


def generate_fcpxml(
    plan: SyncPlan,
    video_infos: list[MediaInfo],
    output_path: Path,
    fcpxml_version: str = "1.9",
    project_name: str = "BormoSync",
    audio_sample_rate: int | None = None,
) -> Path:
    ref_video = video_infos[0] if video_infos else None
    fps: Fraction = (ref_video.fps or Fraction(25, 1)) if ref_video else Fraction(25, 1)
    width: int = ref_video.width or 1920 if ref_video else 1920
    height: int = ref_video.height or 1080 if ref_video else 1080
    # Audio timebase: caller override (e.g. recorder rate) wins, else camera's.
    if audio_sample_rate:
        sample_rate: int = audio_sample_rate
    else:
        sample_rate = (ref_video.audio_sample_rate or 48000) if ref_video else 48000
    timebase = fps.numerator

    frame_dur = fps_to_frame_duration(fps)

    root = ET.Element("fcpxml", version=fcpxml_version)

    resources = ET.SubElement(root, "resources")

    fmt_id = "r1"
    # No custom `name` (a non-standard FFVideoFormat name makes Final Cut warn);
    # colorSpace is declared so the sequence format resolves cleanly.
    ET.SubElement(
        resources,
        "format",
        id=fmt_id,
        frameDuration=frame_dur,
        width=str(width or 1920),
        height=str(height or 1080),
        colorSpace="1-1-1 (Rec. 709)",
    )

    asset_map: dict[str, str] = {}
    asset_counter = 2

    seen_paths: set[str] = set()
    for clip in plan.clips:
        path_str = str(clip.path.resolve())
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)

        asset_id = f"r{asset_counter}"
        asset_counter += 1
        asset_map[path_str] = asset_id

        file_uri = path_to_file_uri(clip.path)
        is_video = clip.kind == "video"
        # Asset duration is expressed on the asset's own timebase: the video
        # frame grid for video, the audio sample rate for audio.
        asset_tb = timebase if is_video else sample_rate
        # NOTE: in FCPXML 1.9+ the file reference lives on the <media-rep> child,
        # NOT as a `src` attribute on <asset> (the DTD has no such attribute and
        # Final Cut rejects it). Keep the asset attributes to the declared set.
        asset_attrs = {
            "id": asset_id,
            "name": clip.path.stem,
            "start": "0s",
            "duration": to_rational(clip.duration + clip.in_point, asset_tb),
            "hasVideo": "1" if is_video else "0",
            "hasAudio": "1",
        }
        if is_video:
            asset_attrs["format"] = fmt_id
        else:
            asset_attrs["audioSources"] = "1"
            asset_attrs["audioChannels"] = "1"
            asset_attrs["audioRate"] = str(sample_rate)

        asset_el = ET.SubElement(resources, "asset", asset_attrs)
        ET.SubElement(asset_el, "media-rep", kind="original-media", src=file_uri)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="BormoSync")
    project = ET.SubElement(event, "project", name=project_name)

    seq_dur = _frame_rational(plan.total_duration, fps, "ceil")
    seq = ET.SubElement(
        project,
        "sequence",
        format=fmt_id,
        tcStart="0s",
        tcFormat="NDF",
        duration=seq_dur,
    )

    spine = ET.SubElement(seq, "spine")

    gap = ET.SubElement(
        spine,
        "gap",
        name="Gap",
        offset="0s",
        start="0s",
        duration=seq_dur,
    )

    for clip in plan.clips:
        path_str = str(clip.path.resolve())
        ref_id = asset_map.get(path_str, "r2")

        # All spine times are snapped to the sequence frame grid (offset rounded,
        # duration floored so we never reference more media than the file holds).
        ET.SubElement(
            gap,
            "asset-clip",
            ref=ref_id,
            lane=str(clip.lane),
            name=clip.path.stem,
            offset=_frame_rational(clip.offset, fps, "round"),
            start=_frame_rational(clip.in_point, fps, "round"),
            duration=_frame_rational(clip.duration, fps, "floor"),
        )

    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ET.indent(tree, space="  ")

    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(b"<!DOCTYPE fcpxml>\n")
        tree.write(f, xml_declaration=False, encoding="UTF-8")

    logger.info("FCPXML written to %s", output_path)
    return output_path


def validate_fcpxml(path: Path) -> bool:
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        if root.tag != "fcpxml":
            logger.error("Root tag is '%s', expected 'fcpxml'", root.tag)
            return False

        spine = root.find(".//spine")
        if spine is None:
            logger.error("No <spine> found")
            return False

        gap = spine.find("gap")
        if gap is None:
            logger.error("No <gap> found in spine")
            return False

        clips = gap.findall("asset-clip")
        logger.info("FCPXML valid: %d asset-clips in spine", len(clips))
        return True

    except ET.ParseError as e:
        logger.error("FCPXML parse error: %s", e)
        return False
