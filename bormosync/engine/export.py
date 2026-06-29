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


def _media_src(path: Path, base_dir: Path) -> str:
    """media-rep ``src`` for a clip's file.

    Media that lives under the FCPXML's own folder (the rendered synced audio) is
    referenced by a path relative to the document, so the project stays portable
    and Final Cut resolves it right next to the XML. Anything outside (the source
    videos) keeps an absolute ``file://`` URL.
    """
    try:
        return str(path.resolve().relative_to(base_dir))
    except ValueError:
        return path_to_file_uri(path)


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
    # Sequence (timeline) rate — every spine position is snapped to this grid.
    seq_fps: Fraction = (ref_video.fps or Fraction(25, 1)) if ref_video else Fraction(25, 1)
    seq_w: int = (ref_video.width or 1920) if ref_video else 1920
    seq_h: int = (ref_video.height or 1080) if ref_video else 1080
    # Audio timebase: caller override (e.g. recorder rate) wins, else camera's.
    if audio_sample_rate:
        sample_rate: int = audio_sample_rate
    else:
        sample_rate = (ref_video.audio_sample_rate or 48000) if ref_video else 48000

    info_by_path = {str(i.path.resolve()): i for i in video_infos}

    root = ET.Element("fcpxml", version=fcpxml_version)
    resources = ET.SubElement(root, "resources")

    # One resource-id counter shared by formats and assets.
    _rid = [1]

    def next_rid() -> str:
        rid = f"r{_rid[0]}"
        _rid[0] += 1
        return rid

    # A distinct <format> per (fps, width, height) so cameras at different rates
    # (e.g. 29.97 vs 30) are each declared honestly and Final Cut conforms them.
    formats: dict[tuple[int, int, int, int], str] = {}

    def _format_for(f: Fraction, w: int, h: int) -> str:
        key = (f.numerator, f.denominator, w or 1920, h or 1080)
        fid = formats.get(key)
        if fid is None:
            fid = next_rid()
            ET.SubElement(
                resources,
                "format",
                id=fid,
                frameDuration=fps_to_frame_duration(f),
                width=str(w or 1920),
                height=str(h or 1080),
                colorSpace="1-1-1 (Rec. 709)",
            )
            formats[key] = fid
        return fid

    seq_fmt = _format_for(seq_fps, seq_w, seq_h)  # r1

    asset_map: dict[str, str] = {}
    base_dir = output_path.parent.resolve()

    seen_paths: set[str] = set()
    for clip in plan.clips:
        path_str = str(clip.path.resolve())
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)

        asset_id = next_rid()
        asset_map[path_str] = asset_id
        file_uri = _media_src(clip.path, base_dir)
        # NOTE: in FCPXML 1.9+ the file reference lives on the <media-rep> child,
        # NOT as a `src` attribute on <asset>. Durations use each asset's own grid:
        # the video's native fps, or the audio sample rate.
        if clip.kind == "video":
            info = info_by_path.get(path_str)
            cfps = info.fps if info and info.fps else seq_fps
            cw = info.width if info and info.width else seq_w
            ch = info.height if info and info.height else seq_h
            asset_attrs = {
                "id": asset_id,
                "name": clip.path.stem,
                "start": "0s",
                "duration": _frame_rational(clip.duration + clip.in_point, cfps, "round"),
                "hasVideo": "1",
                "hasAudio": "1",
                "format": _format_for(cfps, cw, ch),
            }
        else:
            asset_attrs = {
                "id": asset_id,
                "name": clip.path.stem,
                "start": "0s",
                "duration": to_rational(clip.duration + clip.in_point, sample_rate),
                "hasVideo": "0",
                "hasAudio": "1",
                "audioSources": "1",
                "audioChannels": "1",
                "audioRate": str(sample_rate),
            }

        asset_el = ET.SubElement(resources, "asset", asset_attrs)
        ET.SubElement(asset_el, "media-rep", kind="original-media", src=file_uri)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="BormoSync")
    project = ET.SubElement(event, "project", name=project_name)

    seq_dur = _frame_rational(plan.total_duration, seq_fps, "ceil")
    seq = ET.SubElement(
        project,
        "sequence",
        format=seq_fmt,
        tcStart="0s",
        tcFormat="NDF",  # linear timecode (TC == real elapsed); avoids drop-frame ambiguity
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
            offset=_frame_rational(clip.offset, seq_fps, "round"),
            start=_frame_rational(clip.in_point, seq_fps, "round"),
            duration=_frame_rational(clip.duration, seq_fps, "floor"),
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
