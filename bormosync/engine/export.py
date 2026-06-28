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


def generate_fcpxml(
    plan: SyncPlan,
    video_infos: list[MediaInfo],
    output_path: Path,
    fcpxml_version: str = "1.9",
    project_name: str = "BormoSync",
) -> Path:
    ref_video = video_infos[0] if video_infos else None
    fps: Fraction = (ref_video.fps or Fraction(25, 1)) if ref_video else Fraction(25, 1)
    width: int = ref_video.width or 1920 if ref_video else 1920
    height: int = ref_video.height or 1080 if ref_video else 1080
    sample_rate: int = ref_video.audio_sample_rate or 48000 if ref_video else 48000
    timebase = fps.numerator

    frame_dur = fps_to_frame_duration(fps)

    root = ET.Element("fcpxml", version=fcpxml_version)

    resources = ET.SubElement(root, "resources")

    fmt_id = "r1"
    fmt_name = f"FFVideoFormat{width}x{height}p{float(fps):.2f}"
    ET.SubElement(
        resources,
        "format",
        id=fmt_id,
        name=fmt_name,
        frameDuration=frame_dur,
        width=str(width or 1920),
        height=str(height or 1080),
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
        has_video = "1" if clip.kind == "video" else "0"
        has_audio = "1"

        asset_el = ET.SubElement(
            resources,
            "asset",
            id=asset_id,
            name=clip.path.stem,
            src=file_uri,
            start="0s",
            duration=to_rational(clip.duration + clip.in_point, timebase),
            hasVideo=has_video,
            hasAudio=has_audio,
            format=fmt_id,
        )
        ET.SubElement(asset_el, "media-rep", kind="original-media", src=file_uri)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="BormoSync")
    project = ET.SubElement(event, "project", name=project_name)

    seq = ET.SubElement(
        project,
        "sequence",
        format=fmt_id,
        tcStart="0s",
        tcFormat="NDF",
        duration=to_rational(plan.total_duration, timebase),
    )

    spine = ET.SubElement(seq, "spine")

    gap = ET.SubElement(
        spine,
        "gap",
        name="Gap",
        offset="0s",
        start="0s",
        duration=to_rational(plan.total_duration, timebase),
    )

    for clip in plan.clips:
        path_str = str(clip.path.resolve())
        ref_id = asset_map.get(path_str, "r2")

        clip_tb = timebase if clip.kind == "video" else (sample_rate or 48000)

        ET.SubElement(
            gap,
            "clip",
            lane=str(clip.lane),
            name=clip.path.stem,
            offset=to_rational(clip.offset, clip_tb),
            start=to_rational(clip.in_point, clip_tb),
            duration=to_rational(clip.duration, clip_tb),
            ref=ref_id,
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

        clips = gap.findall("clip")
        logger.info("FCPXML valid: %d clips in spine", len(clips))
        return True

    except ET.ParseError as e:
        logger.error("FCPXML parse error: %s", e)
        return False
