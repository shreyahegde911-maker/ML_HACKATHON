import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.candidates import CandidateConfig, build_segments, generate_candidates
from pathsnap.features import build_feature_table, attach_reference_labels, reference_candidate_coverage


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame({"node_id": [0, 1], "x_m": [0., 100.], "y_m": [0., 0.]})
        self.arcs = pd.DataFrame({"arc_id": [0], "from_node_id": [0], "to_node_id": [1]})
        self.segments = build_segments(self.nodes, self.arcs)
        self.obs = pd.DataFrame({"observation_id": [0, 1, 2], "timestamp": [4., 3., 100.], "x_m": [20., 30., 40.], "y_m": [0., 0., 0.]})
        self.ref = pd.DataFrame({"trip_id": ["00000001"] * 3, "observation_id": [0, 1, 2], "reference_arc_id": [0] * 3, "alignment_status": ["aligned"] * 3})

    def features(self):
        return build_feature_table("00000001", self.obs, generate_candidates(self.obs, self.segments), self.segments)

    def test_backwards_timestamp_keeps_source_sequence(self):
        f = self.features().set_index("observation_id")
        self.assertEqual(f.time_gap_s.tolist(), [0., 0., 97.])
        self.assertEqual(f.speed_mps.iloc[1], 0.)
        self.assertAlmostEqual(f.speed_mps.iloc[2], 10 / 97)

    def test_no_candidates_retains_all_reference_misses(self):
        f = self.features().iloc[:0]
        self.assertTrue(attach_reference_labels(f, self.ref).empty)
        coverage = reference_candidate_coverage(f, self.ref)
        self.assertEqual(len(coverage), 3)
        self.assertTrue(coverage.reference_arc_missing.all())
        self.assertFalse(coverage.has_candidates.any())

    def test_unreliable_labels_stay_null_and_candidates_unchanged(self):
        f = self.features()
        labels = attach_reference_labels(f, self.ref.assign(alignment_status="too_far"))
        self.assertTrue(labels.label.isna().all())
        self.assertEqual(len(labels), len(f))

    def test_long_segment_fallback_and_integer_ids(self):
        obs = self.obs.iloc[:1].copy()
        obs["x_m"], obs["y_m"] = 10., 200.
        c = generate_candidates(obs, self.segments)
        self.assertEqual(c.arc_id.tolist(), [0])
        self.assertTrue(c.search_expanded.all())
        self.assertTrue(pd.api.types.is_integer_dtype(c.arc_id))
        obs["y_m"] = 500.
        self.assertTrue(generate_candidates(obs, self.segments).empty)

    def test_missing_reference_arc_is_not_inserted(self):
        f = self.features()
        coverage = reference_candidate_coverage(f, self.ref.assign(reference_arc_id=99))
        self.assertTrue(coverage.reference_arc_missing.all())
        labelled = attach_reference_labels(f, self.ref.assign(reference_arc_id=99))
        self.assertTrue(labelled.label.eq(0).all())
        self.assertEqual(len(labelled), len(f))


if __name__ == "__main__":
    unittest.main()
