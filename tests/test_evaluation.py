import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.evaluation import evaluate_classifier
from pathsnap.models import predict_candidates
from pathsnap.features.engineering import FEATURES


class EvaluationTests(unittest.TestCase):
    def test_prediction_handles_no_candidates(self):
        self.assertEqual(
            len(predict_candidates(None, pd.DataFrame(columns=FEATURES))), 0
        )

    def test_candidate_misses_remain_in_top1_denominator(self):
        features = pd.DataFrame(
            {
                "trip_id": ["a", "a"],
                "observation_id": [0, 0],
                "arc_id": [5, 6],
                "label": [1, 0],
            }
        )
        coverage = pd.DataFrame(
            {
                "trip_id": ["a"] * 3,
                "observation_id": [0, 1, 2],
                "reference_arc_id": [5, 7, 8],
                "label_reliable": [True, True, False],
                "reference_arc_missing": [False, True, False],
            }
        )
        metrics, predictions, _ = evaluate_classifier(
            features, coverage, np.array([0.9, 0.1])
        )
        self.assertEqual(metrics["eligible_observations"], 2)
        self.assertEqual(metrics["observation_top1_accuracy"], 0.5)
        self.assertEqual(metrics["candidate_miss_rate"], 0.5)
        self.assertEqual(metrics["observations_without_candidates"], 2)
        self.assertEqual(metrics["candidate_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
