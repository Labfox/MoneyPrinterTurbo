"""
Semantic placement of video material along the audio/subtitle timeline.

Goal: instead of dropping stock clips onto the timeline in random order, play
each clip at the moment the narration is talking about something related to it.

How it works:
1. The subtitle (.srt) gives us the spoken timeline: a list of (start, end, text)
   segments — i.e. *what is being said and when*.
2. Each downloaded clip carries a short description (the search term it was
   fetched for) — i.e. *what is on screen*.
3. We embed both with a local sentence-transformers model and, for each window
   of the timeline, pick the clip whose description is most similar to the text
   being spoken during that window.

Everything here is best-effort: if sentence-transformers is not installed, or no
descriptions/subtitles are available, callers should fall back to the existing
random/sequential ordering. Nothing in this module raises on the happy path of
"feature unavailable" — it returns None instead.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from loguru import logger

from app.config import config

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,\.](\d+)")


@dataclass
class SemanticPlacement:
    """A single timeline window resolved to a source video clip."""

    source_path: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def is_available() -> bool:
    """Return True if the optional embedding backend can be imported."""
    try:
        import sentence_transformers  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _load_model():
    """
    Lazily load and cache the sentence-transformers model.

    The model is a few tens of MB and is downloaded on first use, so we never
    import or instantiate it unless the semantic feature is actually requested.
    """
    from sentence_transformers import SentenceTransformer

    model_name = str(config.app.get("semantic_match_model", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()
    logger.info(f"loading semantic match model: {model_name}")
    return SentenceTransformer(model_name)


def _parse_srt_timestamp(value: str) -> Optional[float]:
    match = _SRT_TIME.search(value)
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    # millis may be 1-3 digits depending on the writer; normalise to seconds.
    millis = (millis + "000")[:3]
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def parse_subtitle_segments(subtitle_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse an .srt file into (start_seconds, end_seconds, text) tuples.

    Reuses the project's tolerant block parser so we stay consistent with how
    subtitles are rendered elsewhere.
    """
    from app.services import subtitle as subtitle_service

    segments: List[Tuple[float, float, str]] = []
    for _, times_line, text in subtitle_service.file_to_subtitles(subtitle_path):
        if "-->" not in times_line:
            continue
        start_raw, end_raw = times_line.split("-->", 1)
        start = _parse_srt_timestamp(start_raw)
        end = _parse_srt_timestamp(end_raw)
        if start is None or end is None or end <= start:
            continue
        clean_text = text.replace("\n", " ").strip()
        if clean_text:
            segments.append((start, end, clean_text))
    return segments


def _build_windows(
    segments: List[Tuple[float, float, str]],
    audio_duration: float,
    max_clip_duration: float,
) -> List[Tuple[float, float, str]]:
    """
    Slice the timeline into fixed windows of up to max_clip_duration and attach
    the subtitle text spoken during each window.

    Fixed windows (rather than per-subtitle) guarantee we cover the whole
    [0, audio_duration] range with predictable clip lengths, including any tail
    that extends past the last subtitle line.
    """
    if max_clip_duration <= 0:
        max_clip_duration = 5.0

    windows: List[Tuple[float, float, str]] = []
    start = 0.0
    while start < audio_duration:
        end = min(start + max_clip_duration, audio_duration)
        # Collect every subtitle segment that overlaps this window.
        spoken = [
            text
            for seg_start, seg_end, text in segments
            if seg_end > start and seg_start < end
        ]
        windows.append((start, end, " ".join(spoken).strip()))
        start = end
    return windows


def _cosine_best_indices(window_vecs, clip_vecs):
    """Return, per window, the clip indices ranked by descending cosine sim."""
    import numpy as np

    # Vectors are L2-normalised at encode time, so a dot product is the cosine.
    sims = window_vecs @ clip_vecs.T  # (n_windows, n_clips)
    ranked = np.argsort(-sims, axis=1)
    return sims, ranked


def build_semantic_placements(
    subtitle_path: str,
    audio_duration: float,
    clip_descriptions: Dict[str, str],
    clip_durations: Dict[str, float],
    max_clip_duration: float,
    rng=None,
) -> Optional[List[SemanticPlacement]]:
    """
    Resolve the audio timeline to an ordered list of source-clip placements.

    Returns None (signalling the caller to fall back to random ordering) when the
    feature cannot be applied: backend missing, no descriptions, no usable
    subtitle text, or any unexpected embedding failure.
    """
    import random as _random

    rng = rng or _random.Random()

    if not clip_descriptions or audio_duration <= 0:
        return None
    if not is_available():
        logger.warning(
            "semantic match requested but sentence-transformers is not installed; "
            "install the 'semantic' extra (pip install sentence-transformers). "
            "falling back to random ordering."
        )
        return None

    segments = parse_subtitle_segments(subtitle_path)
    if not segments:
        logger.warning(
            "semantic match requested but no usable subtitle text was found; "
            "falling back to random ordering."
        )
        return None

    # Stable ordering of the clip pool so embeddings line up with paths.
    clip_paths = [p for p in clip_descriptions if clip_descriptions.get(p)]
    if not clip_paths:
        return None
    descriptions = [clip_descriptions[p] for p in clip_paths]

    windows = _build_windows(segments, audio_duration, max_clip_duration)
    if not windows:
        return None

    try:
        import numpy as np  # noqa: F401

        model = _load_model()
        window_texts = [text if text else "" for _, _, text in windows]
        window_vecs = model.encode(
            window_texts, normalize_embeddings=True, show_progress_bar=False
        )
        clip_vecs = model.encode(
            descriptions, normalize_embeddings=True, show_progress_bar=False
        )
        sims, ranked = _cosine_best_indices(window_vecs, clip_vecs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"semantic match failed during embedding, falling back: {exc}")
        return None

    placements: List[SemanticPlacement] = []
    prev_idx = -1
    for w_index, (start, end, text) in enumerate(windows):
        window_dur = end - start
        order = list(ranked[w_index])

        # If this window has no spoken text, or no description is meaningfully
        # related, just rotate through clips to keep visual variety.
        if not text:
            chosen = order[w_index % len(order)]
        else:
            chosen = order[0]
            # Avoid showing the exact same source twice in a row when a close
            # runner-up exists, so matched-but-repetitive clips don't stack up.
            if chosen == prev_idx and len(order) > 1:
                runner_up = order[1]
                if sims[w_index][chosen] - sims[w_index][runner_up] < 0.05:
                    chosen = runner_up
        prev_idx = chosen

        source_path = clip_paths[chosen]
        source_dur = float(clip_durations.get(source_path, 0.0)) or window_dur
        clip_len = min(window_dur, source_dur)

        # Pick where inside the source clip to start; randomise when there is
        # slack so repeated picks of the same source don't always show the head.
        slack = max(0.0, source_dur - clip_len)
        clip_start = rng.uniform(0, slack) if slack > 0 else 0.0
        placements.append(
            SemanticPlacement(
                source_path=source_path,
                start_time=clip_start,
                end_time=clip_start + clip_len,
            )
        )

    logger.info(
        f"semantic match: placed {len(placements)} windows across "
        f"{len(clip_paths)} source clips over {audio_duration:.1f}s"
    )
    return placements
