import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from pathsnap.decoding import decode, sequence_metrics


class DecoderTests(unittest.TestCase):
    def trip(self):
        return {
            "trip_id": "00000001",
            "nodes": pd.DataFrame(
                {"x_m": [0, 100, 200, 0, 100, 200], "y_m": [0, 0, 0, 20, 20, 20]}
            ),
            "arcs": pd.DataFrame(
                {"from_node_id": [0, 1, 3, 4], "to_node_id": [1, 2, 4, 5]}
            ),
            "observations": pd.DataFrame(
                {
                    "observation_id": [0, 1, 2],
                    "x_m": [10.0, 80.0, 120.0],
                    "y_m": [0.0, 0.0, 0.0],
                    "timestamp": [1.0, 2.0, 3.0],
                }
            ),
        }

    def test_viterbi_rejects_disconnected_pointwise_jump(self):
        trip = self.trip()
        c = pd.DataFrame(
            {"observation_id": [0, 0, 1, 1, 2, 2], "arc_id": [0, 2, 0, 2, 1, 3]}
        )
        out = decode(trip, c, np.array([0.9, 0.1, 0.45, 0.55, 0.9, 0.1]))
        self.assertEqual(out["predictions"].predicted_arc_id.tolist(), [0, 0, 1])
        self.assertEqual(out["route_runs"], [[0, 1]])

    def test_gap_starts_new_route_instead_of_false_bridge(self):
        trip = self.trip()
        trip["observations"].loc[1, "timestamp"] = 100
        trip["observations"].loc[2, "timestamp"] = 101
        c = pd.DataFrame({"observation_id": [0, 1, 2], "arc_id": [0, 2, 3]})
        out = decode(trip, c, np.ones(3))
        self.assertEqual(len(out["route_runs"]), 2)
        self.assertEqual(out["breaks"][0]["reason"], "timestamp_gap")

    def test_no_candidates_preserves_missing_prediction(self):
        trip = self.trip()
        c = pd.DataFrame({"observation_id": [0, 2], "arc_id": [0, 1]})
        out = decode(trip, c, np.ones(2))
        self.assertEqual(len(out["predictions"]), 3)
        self.assertTrue(pd.isna(out["predictions"].predicted_arc_id.iloc[1]))

    def test_directed_connectors_and_unreachable_reverse(self):
        trip = {
            "trip_id": "00000001",
            "nodes": pd.DataFrame({"x_m": [0, 100, 200, 300], "y_m": [0, 0, 0, 0]}),
            "arcs": pd.DataFrame({"from_node_id": [0, 1, 2], "to_node_id": [1, 2, 3]}),
            "observations": pd.DataFrame(
                {
                    "observation_id": [0, 1],
                    "x_m": [10.0, 290.0],
                    "y_m": [0.0, 0.0],
                    "timestamp": [1.0, 11.0],
                }
            ),
        }
        c = pd.DataFrame({"observation_id": [0, 1], "arc_id": [0, 2]})
        out = decode(trip, c, np.ones(2))
        self.assertEqual(out["route_runs"], [[0, 1, 2]])
        trip["observations"]["x_m"] = [290.0, 10.0]
        c["arc_id"] = [2, 0]
        out = decode(trip, c, np.ones(2))
        self.assertEqual(out["route_runs"], [[2], [0]])
        self.assertEqual(
            out["breaks"][0]["reason"], "disconnected_or_implausible_transition"
        )

    def test_revisiting_same_arc_keeps_loop_connectors(self):
        trip = {
            "trip_id": "00000001",
            "nodes": pd.DataFrame({"x_m": [0, 100, 100], "y_m": [0, 0, 50]}),
            "arcs": pd.DataFrame({"from_node_id": [0, 1, 2], "to_node_id": [1, 2, 0]}),
            "observations": pd.DataFrame(
                {
                    "observation_id": [0, 1],
                    "x_m": [90.0, 10.0],
                    "y_m": [0.0, 0.0],
                    "timestamp": [1.0, 11.0],
                }
            ),
        }
        candidates = pd.DataFrame({"observation_id": [0, 1], "arc_id": [0, 0]})
        decoded = decode(trip, candidates, np.ones(2))
        self.assertEqual(decoded["predictions"].predicted_arc_id.tolist(), [0, 0])
        self.assertEqual(decoded["route_runs"], [[0, 1, 2, 0]])

    def test_ui_payload_includes_connector_geometry(self):
        from pathsnap.api.service import payload

        trip = {
            "trip_id": "00000001",
            "crs": {"projected": "EPSG:32632"},
            "nodes": pd.DataFrame({"x_m": [0, 100, 200, 300], "y_m": [0, 0, 0, 0]}),
            "arcs": pd.DataFrame({"from_node_id": [0, 1, 2], "to_node_id": [1, 2, 3]}),
            "observations": pd.DataFrame(
                {
                    "observation_id": [0, 1],
                    "x_m": [10.0, 290.0],
                    "y_m": [0.0, 0.0],
                    "timestamp": [1.0, 11.0],
                }
            ),
        }
        candidates = pd.DataFrame({"observation_id": [0, 1], "arc_id": [0, 2]})
        decoded = decode(trip, candidates, np.ones(2))
        result = payload(trip, decoded["predictions"], decoded, None, 0.0, "test")
        self.assertEqual(len(result["classifier_segments"]), 2)
        self.assertEqual(len(result["decoded_segments"]), 3)
        self.assertEqual(len(result["map_geojson"]["decoded_segments"]["coordinates"]), 3)


if __name__ == "__main__":
    unittest.main()
