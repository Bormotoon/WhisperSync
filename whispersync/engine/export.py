"""FCPXML generator for Final Cut Pro / DaVinci Resolve."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from whispersync.engine.media import MediaInfo, path_to_file_uri
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
    # Frame duration (seconds) of the sequence grid, used to align spine offsets —
    # including inside a compound clip's own nested spine (built below).
    frame_s = float(seq_fps.denominator) / float(seq_fps.numerator)

    asset_map: dict[str, str] = {}
    base_dir = output_path.parent.resolve()
    # Compound audio clips (clip.subclips set) get a <media> resource instead of a
    # plain <asset>; keyed by object identity since these MediaClips only live for
    # the duration of this call.
    media_map: dict[int, str] = {}

    seen_paths: set[str] = set()
    for clip in plan.clips:
        if clip.kind == "audio" and clip.subclips:
            # COMPOUND CLIP: each speech piece stays its own asset, placed at its own
            # offset inside a nested <media><sequence><spine>, sized to the clip's
            # full (video) duration. The main sequence later references this <media>
            # via a <ref-clip> instead of an <asset-clip>, so every piece remains an
            # individually editable clip in Final Cut (for hand crossfades/nudges).
            subs = sorted(clip.subclips, key=lambda s: s.offset)
            sub_asset_ids: list[str] = []
            for sub in subs:
                sub_path_str = str(sub.path.resolve())
                sub_asset_id = asset_map.get(sub_path_str)
                if sub_asset_id is None:
                    sub_asset_id = next_rid()
                    asset_map[sub_path_str] = sub_asset_id
                    sub_asset_attrs = {
                        "id": sub_asset_id,
                        "name": sub.path.stem,
                        "start": "0s",
                        "duration": to_rational(sub.duration + sub.in_point, sample_rate),
                        "hasVideo": "0",
                        "hasAudio": "1",
                        "audioSources": "1",
                        "audioChannels": "1",
                        "audioRate": str(sample_rate),
                    }
                    sub_asset_el = ET.SubElement(resources, "asset", sub_asset_attrs)
                    ET.SubElement(
                        sub_asset_el,
                        "media-rep",
                        kind="original-media",
                        src=_media_src(sub.path, base_dir),
                    )
                sub_asset_ids.append(sub_asset_id)

            media_id = next_rid()
            media_name = clip.display_name or clip.path.stem
            media_el = ET.SubElement(resources, "media", id=media_id, name=media_name)
            inner_seq = ET.SubElement(
                media_el,
                "sequence",
                format=seq_fmt,
                duration=_frame_rational(clip.duration, seq_fps, "ceil"),
            )
            inner_spine = ET.SubElement(inner_seq, "spine")
            cursor = 0.0
            for idx, (sub, sub_asset_id) in enumerate(zip(subs, sub_asset_ids, strict=True)):
                if sub.offset > cursor + frame_s / 2:
                    ET.SubElement(
                        inner_spine,
                        "gap",
                        name="Gap",
                        offset=_frame_rational(cursor, seq_fps, "round"),
                        start="0s",
                        duration=_frame_rational(sub.offset - cursor, seq_fps, "round"),
                    )
                    cursor = sub.offset
                ET.SubElement(
                    inner_spine,
                    "asset-clip",
                    ref=sub_asset_id,
                    name=f"{media_name}_piece{idx:03d}",
                    offset=_frame_rational(cursor, seq_fps, "round"),
                    start=_frame_rational(sub.in_point, seq_fps, "round"),
                    duration=_frame_rational(sub.duration, seq_fps, "floor"),
                )
                cursor += sub.duration
            media_map[id(clip)] = media_id
            continue

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
            asset_attrs = {
                "id": asset_id,
                "name": asset_name,
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

    def _clip_name(clip: MediaClip) -> str:
        return clip.display_name or clip.path.stem

    def _set_role(el: ET.Element, clip: MediaClip) -> None:
        # FCPX colours/groups clips by role: videoRole on video, audioRole on audio.
        # NOTE: ref-clip (compound clips) does NOT declare these attributes in the
        # FCPXML DTD — only asset-clip/clip/gap/etc. do. Setting audioRole/videoRole
        # on a ref-clip fails Final Cut's DTD validation on import ("No declaration
        # for attribute audioRole of element ref-clip"), so skip it there. (Proper
        # role assignment for a compound clip would need a nested
        # <audio-role-source> child instead of an attribute.)
        if clip.role and el.tag != "ref-clip":
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
            offset_str = _frame_rational(clip.offset, seq_fps, "round")
            duration_str = _frame_rational(clip.duration, seq_fps, "floor")
            if clip.subclips:
                el = ET.SubElement(
                    gap,
                    "ref-clip",
                    ref=media_map[id(clip)],
                    lane=str(clip.lane),
                    name=_clip_name(clip),
                    offset=offset_str,
                    start="0s",
                    duration=duration_str,
                )
            else:
                el = ET.SubElement(
                    gap,
                    "asset-clip",
                    ref=asset_map.get(str(clip.path.resolve()), "r2"),
                    lane=str(clip.lane),
                    name=_clip_name(clip),
                    offset=offset_str,
                    start=_frame_rational(clip.in_point, seq_fps, "round"),
                    duration=duration_str,
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
            offset_str = _frame_rational(parent_in_pt + rel, seq_fps, "round")
            duration_str = _frame_rational(clip.duration, seq_fps, "floor")
            if clip.subclips:
                # Compound clip: reference the <media> resource built above. Its own
                # nested sequence always starts at 0, so `start` here is just "0s".
                el = ET.SubElement(
                    parent_el,
                    "ref-clip",
                    ref=media_map[id(clip)],
                    lane=str(clip.lane),
                    name=_clip_name(clip),
                    offset=offset_str,
                    start="0s",
                    duration=duration_str,
                )
            else:
                el = ET.SubElement(
                    parent_el,
                    "asset-clip",
                    ref=asset_map.get(str(clip.path.resolve()), "r2"),
                    lane=str(clip.lane),
                    name=_clip_name(clip),
                    offset=offset_str,
                    start=_frame_rational(clip.in_point, seq_fps, "round"),
                    duration=duration_str,
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

        # Defense in depth: per the FCPXML DTD, <ref-clip> (compound/multicam
        # references) does NOT declare audioRole/videoRole — only <asset-clip> does.
        # Final Cut rejects the WHOLE import with "No declaration for attribute
        # audioRole of element ref-clip" if either sneaks in, so catch it here
        # ourselves rather than let the user find out from Final Cut. Checked
        # document-wide (not just the main spine) since a compound clip's own
        # nested spine could theoretically hold one too.
        bad_role_clips = [
            rc
            for rc in root.iter("ref-clip")
            if rc.get("audioRole") is not None or rc.get("videoRole") is not None
        ]
        if bad_role_clips:
            logger.error(
                "%d <ref-clip> element(s) carry audioRole/videoRole — Final Cut's "
                "DTD does not declare these attributes for ref-clip and will refuse "
                "the whole import",
                len(bad_role_clips),
            )
            return False

        # Scope the search to the project's own spine under <library> — a compound
        # clip's <media> resource (under <resources>, earlier in document order) has
        # its own nested <spine> that must NOT be mistaken for the real one.
        library = root.find("library")
        spine = library.find(".//spine") if library is not None else None
        if spine is None:
            logger.error("No <spine> found")
            return False

        # Video clips now sit directly in the spine (primary storyline); audio is
        # connected to them (either a plain asset-clip, or a ref-clip for a compound
        # clip). A valid document has at least one clip somewhere in the spine
        # (directly, or nested under a clip / gap).
        clips = spine.findall(".//asset-clip") + spine.findall(".//ref-clip")
        if not clips:
            logger.error("No <asset-clip>/<ref-clip> found in spine")
            return False

        logger.info("FCPXML valid: %d clip(s) in spine", len(clips))
        return True

    except ET.ParseError as e:
        logger.error("FCPXML parse error: %s", e)
        return False
