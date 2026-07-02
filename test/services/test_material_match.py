import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import material_match


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:03,000
A cute cat plays on the sofa at home.

2
00:00:03,000 --> 00:00:06,000
Then the dog runs fast across the park.

3
00:00:06,000 --> 00:00:09,000
Finally we talk about money and the stock market.
"""


def _write_srt(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class _TopicModel:
    """
    Deterministic stand-in for a sentence-transformers model.

    Each text is embedded as a one-hot over a fixed topic vocabulary based on the
    keywords it contains, so cosine similarity is fully predictable without
    needing torch or a real model download.
    """

    TOPICS = ("cat", "dog", "money")

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for text in texts:
            lowered = text.lower()
            vec = np.array(
                [1.0 if topic in lowered else 0.0 for topic in self.TOPICS]
            )
            norm = np.linalg.norm(vec)
            if norm == 0:
                # Unmatched text gets a uniform vector; still valid/normalised.
                vec = np.ones(len(self.TOPICS))
                norm = np.linalg.norm(vec)
            vecs.append(vec / norm)
        return np.array(vecs)


class _FixedSimModel:
    """Returns preset vectors so we can drive exact similarity scores."""

    def __init__(self, window_vecs, clip_vecs, clip_texts):
        self._window_vecs = np.array(window_vecs)
        self._clip_vecs = np.array(clip_vecs)
        self._clip_texts = set(clip_texts)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if set(texts) <= self._clip_texts:
            return self._clip_vecs
        return self._window_vecs


class TestTimestampParsing(unittest.TestCase):
    def test_parse_comma_and_dot_milliseconds(self):
        self.assertAlmostEqual(
            material_match._parse_srt_timestamp("00:01:02,500 --> x"), 62.5
        )
        self.assertAlmostEqual(
            material_match._parse_srt_timestamp("01:00:00.250"), 3600.25
        )

    def test_parse_invalid_returns_none(self):
        self.assertIsNone(material_match._parse_srt_timestamp("not a time"))


class TestSubtitleSegments(unittest.TestCase):
    def setUp(self):
        self.srt_path = _write_srt(SAMPLE_SRT)

    def tearDown(self):
        os.remove(self.srt_path)

    def test_parses_segments_with_times_and_text(self):
        segments = material_match.parse_subtitle_segments(self.srt_path)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][:2], (0.0, 3.0))
        self.assertIn("cat", segments[0][2].lower())
        self.assertEqual(segments[2][:2], (6.0, 9.0))

    def test_missing_file_returns_empty(self):
        self.assertEqual(
            material_match.parse_subtitle_segments("/no/such/file.srt"), []
        )


class TestBuildWindows(unittest.TestCase):
    def test_windows_cover_full_duration(self):
        segments = [(0.0, 2.0, "cat"), (2.0, 4.0, "dog"), (4.0, 7.0, "money")]
        windows = material_match._build_windows(segments, audio_duration=8.0, max_clip_duration=5.0)
        # 0-5 and 5-8
        self.assertEqual([(w[0], w[1]) for w in windows], [(0.0, 5.0), (5.0, 8.0)])
        # full coverage to the audio end even past the last subtitle
        self.assertAlmostEqual(windows[-1][1], 8.0)

    def test_text_from_overlapping_segments_only(self):
        segments = [(0.0, 2.0, "cat"), (2.0, 4.0, "dog"), (4.0, 7.0, "money")]
        windows = material_match._build_windows(segments, audio_duration=7.0, max_clip_duration=3.0)
        # first window [0,3) overlaps cat (0-2) and dog (2-4)
        self.assertIn("cat", windows[0][2])
        self.assertIn("dog", windows[0][2])
        self.assertNotIn("money", windows[0][2])

    def test_zero_max_duration_defaults_safely(self):
        windows = material_match._build_windows([], audio_duration=4.0, max_clip_duration=0)
        self.assertTrue(windows)
        self.assertLessEqual(windows[0][1] - windows[0][0], 5.0)


class TestBuildSemanticPlacementsFallback(unittest.TestCase):
    """All the conditions under which the feature must return None (fall back)."""

    def setUp(self):
        self.srt_path = _write_srt(SAMPLE_SRT)
        self.descriptions = {"/v/cat.mp4": "kitten cat"}
        self.durations = {"/v/cat.mp4": 30.0}

    def tearDown(self):
        os.remove(self.srt_path)

    def test_no_descriptions(self):
        self.assertIsNone(
            material_match.build_semantic_placements(
                self.srt_path, 9.0, {}, {}, 3
            )
        )

    def test_non_positive_duration(self):
        self.assertIsNone(
            material_match.build_semantic_placements(
                self.srt_path, 0.0, self.descriptions, self.durations, 3
            )
        )

    def test_backend_unavailable(self):
        with patch.object(material_match, "is_available", return_value=False):
            self.assertIsNone(
                material_match.build_semantic_placements(
                    self.srt_path, 9.0, self.descriptions, self.durations, 3
                )
            )

    def test_no_subtitle_segments(self):
        with patch.object(material_match, "is_available", return_value=True):
            self.assertIsNone(
                material_match.build_semantic_placements(
                    "/no/such/file.srt", 9.0, self.descriptions, self.durations, 3
                )
            )

    def test_embedding_error_falls_back(self):
        with patch.object(material_match, "is_available", return_value=True), patch.object(
            material_match, "_load_model", side_effect=RuntimeError("boom")
        ):
            self.assertIsNone(
                material_match.build_semantic_placements(
                    self.srt_path, 9.0, self.descriptions, self.durations, 3
                )
            )


class TestBuildSemanticPlacements(unittest.TestCase):
    def setUp(self):
        self.srt_path = _write_srt(SAMPLE_SRT)

    def tearDown(self):
        os.remove(self.srt_path)

    def test_each_window_matches_related_clip(self):
        descriptions = {
            "/v/cat.mp4": "an adorable cat",
            "/v/dog.mp4": "a dog in the park",
            "/v/money.mp4": "money and stock market finance",
        }
        durations = {p: 30.0 for p in descriptions}

        with patch.object(material_match, "is_available", return_value=True), patch.object(
            material_match, "_load_model", return_value=_TopicModel()
        ):
            placements = material_match.build_semantic_placements(
                self.srt_path,
                audio_duration=9.0,
                clip_descriptions=descriptions,
                clip_durations=durations,
                max_clip_duration=3,
                rng=random.Random(0),
            )

        self.assertIsNotNone(placements)
        self.assertEqual(
            [p.source_path for p in placements],
            ["/v/cat.mp4", "/v/dog.mp4", "/v/money.mp4"],
        )

    def test_placement_durations_and_bounds(self):
        descriptions = {"/v/cat.mp4": "cat", "/v/money.mp4": "money"}
        durations = {"/v/cat.mp4": 2.0, "/v/money.mp4": 30.0}

        with patch.object(material_match, "is_available", return_value=True), patch.object(
            material_match, "_load_model", return_value=_TopicModel()
        ):
            placements = material_match.build_semantic_placements(
                self.srt_path,
                audio_duration=9.0,
                clip_descriptions=descriptions,
                clip_durations=durations,
                max_clip_duration=3,
                rng=random.Random(1),
            )

        for p in placements:
            # never read past the end of the source clip
            self.assertLessEqual(p.end_time, durations[p.source_path] + 1e-6)
            self.assertGreaterEqual(p.start_time, 0.0)
            self.assertGreater(p.duration, 0.0)
            # window length is capped at max_clip_duration
            self.assertLessEqual(p.duration, 3.0 + 1e-6)

    def test_clip_goes_to_the_window_it_matches_best(self):
        # Both windows prefer clip A, but the second window matches it far more
        # strongly. A must go to the second window; the first takes its own
        # runner-up (B) instead of stealing A just by coming earlier.
        srt = (
            "1\n00:00:00,000 --> 00:00:03,000\ntopic\n\n"
            "2\n00:00:03,000 --> 00:00:06,000\ntopic\n"
        )
        path = _write_srt(srt)
        try:
            descriptions = {"/v/a.mp4": "A", "/v/b.mp4": "B"}
            durations = {"/v/a.mp4": 30.0, "/v/b.mp4": 30.0}
            model = _FixedSimModel(
                window_vecs=[[0.90, 0.80], [0.95, 0.30]],
                clip_vecs=[[1.0, 0.0], [0.0, 1.0]],
                clip_texts=["A", "B"],
            )
            with patch.object(material_match, "is_available", return_value=True), patch.object(
                material_match, "_load_model", return_value=model
            ):
                placements = material_match.build_semantic_placements(
                    path,
                    audio_duration=6.0,
                    clip_descriptions=descriptions,
                    clip_durations=durations,
                    max_clip_duration=3,
                    rng=random.Random(0),
                )
            self.assertEqual(
                [p.source_path for p in placements], ["/v/b.mp4", "/v/a.mp4"]
            )
        finally:
            os.remove(path)

    def test_each_clip_used_at_most_once(self):
        # Every window prefers clip A; without-replacement selection must still
        # spread across the pool instead of looping the same source.
        srt = "".join(
            f"{i + 1}\n00:00:0{i * 3},000 --> 00:00:0{i * 3 + 3},000\ntopic\n\n"
            for i in range(3)
        )
        path = _write_srt(srt)
        try:
            descriptions = {"/v/a.mp4": "A", "/v/b.mp4": "B", "/v/c.mp4": "C"}
            durations = {p: 30.0 for p in descriptions}
            model = _FixedSimModel(
                window_vecs=[[0.9, 0.5, 0.1]] * 3,
                clip_vecs=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                clip_texts=["A", "B", "C"],
            )
            with patch.object(material_match, "is_available", return_value=True), patch.object(
                material_match, "_load_model", return_value=model
            ):
                placements = material_match.build_semantic_placements(
                    path,
                    audio_duration=9.0,
                    clip_descriptions=descriptions,
                    clip_durations=durations,
                    max_clip_duration=3,
                    rng=random.Random(0),
                )
            sources = [p.source_path for p in placements]
            self.assertEqual(len(sources), len(set(sources)))
            self.assertEqual(sources[0], "/v/a.mp4")
        finally:
            os.remove(path)

    def test_reuse_only_after_pool_exhausted(self):
        # More windows than clips: reuse is unavoidable, but only after every
        # clip has been shown once, and never twice in a row.
        srt = "".join(
            f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i + 1},000\ntopic\n\n"
            for i in range(5)
        )
        path = _write_srt(srt)
        try:
            descriptions = {"/v/a.mp4": "A", "/v/b.mp4": "B"}
            durations = {p: 30.0 for p in descriptions}
            model = _FixedSimModel(
                window_vecs=[[0.9, 0.5]] * 5,
                clip_vecs=[[1.0, 0.0], [0.0, 1.0]],
                clip_texts=["A", "B"],
            )
            with patch.object(material_match, "is_available", return_value=True), patch.object(
                material_match, "_load_model", return_value=model
            ):
                placements = material_match.build_semantic_placements(
                    path,
                    audio_duration=5.0,
                    clip_descriptions=descriptions,
                    clip_durations=durations,
                    max_clip_duration=1,
                    rng=random.Random(0),
                )
            sources = [p.source_path for p in placements]
            self.assertEqual(set(sources[:2]), {"/v/a.mp4", "/v/b.mp4"})
            for prev, cur in zip(sources, sources[1:]):
                self.assertNotEqual(prev, cur)
        finally:
            os.remove(path)

    def test_avoids_immediate_repeat_when_runner_up_is_close(self):
        # Two identical windows that both prefer clip A, with B only 0.03 behind.
        srt = (
            "1\n00:00:00,000 --> 00:00:03,000\ntopic\n\n"
            "2\n00:00:03,000 --> 00:00:06,000\ntopic\n"
        )
        path = _write_srt(srt)
        try:
            descriptions = {"/v/a.mp4": "A", "/v/b.mp4": "B"}
            durations = {"/v/a.mp4": 30.0, "/v/b.mp4": 30.0}
            model = _FixedSimModel(
                window_vecs=[[0.90, 0.87], [0.90, 0.87]],
                clip_vecs=[[1.0, 0.0], [0.0, 1.0]],
                clip_texts=["A", "B"],
            )
            with patch.object(material_match, "is_available", return_value=True), patch.object(
                material_match, "_load_model", return_value=model
            ):
                placements = material_match.build_semantic_placements(
                    path,
                    audio_duration=6.0,
                    clip_descriptions=descriptions,
                    clip_durations=durations,
                    max_clip_duration=3,
                    rng=random.Random(0),
                )
            self.assertEqual(
                [p.source_path for p in placements], ["/v/a.mp4", "/v/b.mp4"]
            )
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
