"""FCPXML generator for Final Cut Pro / DaVinci Resolve."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from whispersync.engine.media import MediaInfo, path_to_file_uri, probe
from whispersync.models import MediaClip, SyncPlan

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
    project_name: str = "WhisperSync",
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
        asset_name = clip.display_name or clip.path.stem
        if clip.kind == "video":
            info = info_by_path.get(path_str)
            cfps = info.fps if info and info.fps else seq_fps
            cw = info.width if info and info.width else seq_w
            ch = info.height if info and info.height else seq_h
            asset_attrs = {
                "id": asset_id,
                "name": asset_name,
                "start": "0s",
                "duration": _frame_rational(clip.duration + clip.in_point, cfps, "round"),
                "hasVideo": "1",
                "hasAudio": "1",
                "format": _format_for(cfps, cw, ch),
            }
        else:
            # Report the rendered file's real channel count/rate (it now preserves
            # the recorder's native channels — see PROJECT_ANALYSIS.md §2.0) rather
            # than a hard-coded mono assumption; falls back to the sequence default
            # if the file can't be probed (e.g. in unit tests with fake paths).
            audio_channels = 1
            audio_rate = sample_rate
            try:
                audio_info = probe(clip.path)
                audio_channels = audio_info.audio_channels or 1
                audio_rate = audio_info.audio_sample_rate or sample_rate
            except (RuntimeError, OSError):
                pass
            asset_attrs = {
                "id": asset_id,
                "name": asset_name,
                "start": "0s",
                "duration": to_rational(clip.duration + clip.in_point, audio_rate),
                "hasVideo": "0",
                "hasAudio": "1",
                "audioSources": "1",
                "audioChannels": str(audio_channels),
                "audioRate": str(audio_rate),
            }

        asset_el = ET.SubElement(resources, "asset", asset_attrs)
        ET.SubElement(asset_el, "media-rep", kind="original-media", src=file_uri)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="WhisperSync")
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

    # Video clips form the PRIMARY STORYLINE: they sit directly in the spine, in
    # timeline order, with <gap> elements filling the time before the first clip and
    # between non-contiguous clips (so each clip keeps its real synced position).
    # Audio clips are CONNECTED clips, attached to the spine video clip that covers
    # their start, with an offset/lane relative to that parent. This drops the old
    # full-timeline gap placeholder so video lands on the main track on import.
    video_clips = sorted((c for c in plan.clips if c.kind == "video"), key=lambda c: c.offset)
    audio_clips = sorted((c for c in plan.clips if c.kind != "video"), key=lambda c: c.offset)

    # Frame duration (seconds) of the sequence grid, used to align spine offsets.
    frame_s = float(seq_fps.denominator) / float(seq_fps.numerator)

    def _clip_name(clip: MediaClip) -> str:
        return clip.display_name or clip.path.stem

    def _set_role(el: ET.Element, clip: MediaClip) -> None:
        # FCPX colours/groups clips by role: videoRole on video, audioRole on audio.
        if clip.role:
            el.set("videoRole" if clip.kind == "video" else "audioRole", clip.role)

    def _spine_clip(clip: MediaClip, offset_str: str) -> ET.Element:
        el = ET.SubElement(
            spine,
            "asset-clip",
            ref=asset_map.get(str(clip.path.resolve()), "r2"),
            name=_clip_name(clip),
            offset=offset_str,
            start=_frame_rational(clip.in_point, seq_fps, "round"),
            duration=_frame_rational(clip.duration, seq_fps, "floor"),
        )
        _set_role(el, clip)
        return el

    # Lay the video clips end-to-end with gaps for the holes. The spine's own clock
    # ("offset") is contiguous; each element's offset is where it begins on it.
    # Track (timeline_start, timeline_end, parent_in_point, element) so connected
    # clips can be positioned on the PARENT's local clock (which starts at in_point).
    spine_elems: list[tuple[float, float, float, ET.Element]] = []
    cursor = 0.0
    for clip in video_clips:
        if clip.offset > cursor + frame_s / 2:
            gap_dur = clip.offset - cursor
            ET.SubElement(
                spine,
                "gap",
                name="Gap",
                offset=_frame_rational(cursor, seq_fps, "round"),
                start="0s",
                duration=_frame_rational(gap_dur, seq_fps, "round"),
            )
            cursor = clip.offset
        el = _spine_clip(clip, _frame_rational(cursor, seq_fps, "round"))
        end = cursor + clip.duration
        spine_elems.append((cursor, end, clip.in_point, el))
        cursor = end

    if not spine_elems:
        # No video (audio-only) — fall back to a single gap holding the audio so the
        # document is still valid.
        gap = ET.SubElement(spine, "gap", name="Gap", offset="0s", start="0s", duration=seq_dur)
        for clip in audio_clips:
            el = ET.SubElement(
                gap,
                "asset-clip",
                ref=asset_map.get(str(clip.path.resolve()), "r2"),
                lane=str(clip.lane),
                name=_clip_name(clip),
                offset=_frame_rational(clip.offset, seq_fps, "round"),
                start=_frame_rational(clip.in_point, seq_fps, "round"),
                duration=_frame_rational(clip.duration, seq_fps, "floor"),
            )
            _set_role(el, clip)
    else:
        # Attach each audio clip to the spine video clip covering its start.
        for clip in audio_clips:
            parent = None
            for start_s, end_s, in_pt, el in spine_elems:
                if start_s - frame_s <= clip.offset < end_s:
                    parent = (start_s, in_pt, el)
                    break
            if parent is None:
                # Before the first / after the last video — clamp to the nearest.
                s0, _e0, ip0, el0 = spine_elems[0]
                parent = (s0, ip0, el0)
            parent_tl_start, parent_in_pt, parent_el = parent
            rel = max(0.0, clip.offset - parent_tl_start)
            # A connected clip's `offset` is on the PARENT's OWN clock, which starts
            # at the parent's `start` value (its in_point), NOT at the parent's spine
            # position. So offset = parent_in_point + rel (was mistakenly the spine
            # offset + rel, which pushed the audio far to the right).
            el = ET.SubElement(
                parent_el,
                "asset-clip",
                ref=asset_map.get(str(clip.path.resolve()), "r2"),
                lane=str(clip.lane),
                name=_clip_name(clip),
                offset=_frame_rational(parent_in_pt + rel, seq_fps, "round"),
                start=_frame_rational(clip.in_point, seq_fps, "round"),
                duration=_frame_rational(clip.duration, seq_fps, "floor"),
            )
            _set_role(el, clip)

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

        # Video clips now sit directly in the spine (primary storyline); audio is
        # connected to them. A valid document has at least one asset-clip somewhere
        # in the spine (directly, or nested under a clip / gap).
        clips = spine.findall(".//asset-clip")
        if not clips:
            logger.error("No <asset-clip> found in spine")
            return False

        logger.info("FCPXML valid: %d asset-clip(s) in spine", len(clips))
        return True

    except ET.ParseError as e:
        logger.error("FCPXML parse error: %s", e)
        return False
