"""Focused invariants for raw input safety and provisional supervision."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.data import load_trip, load_reference_route, align_reference_route, create_split_manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "train" / "00000001"
        self.directory.mkdir(parents=True)
        self.write("nodes", "8 50\n8.001 50\n8.002 50\n")
        self.write("arcs", "0 1\n1 2\n")
        self.write("track", "8.0002 50 1\n8.0004 50 2\n8.0012 50 3\n")
        self.write("route", "0\n1\n")

    def write(self, extension, content):
        (self.directory / f"00000001.{extension}").write_text(content)

    def test_loader_requires_no_reference_and_preserves_ids(self):
        (self.directory / "00000001.route").unlink()
        trip = load_trip(self.root, "train", 1)
        self.assertEqual(trip["observations"]["observation_id"].tolist(), [0, 1, 2])
        self.assertEqual(trip["arcs"]["arc_id"].tolist(), [0, 1])
        self.assertEqual(trip["crs"]["units"], "metre")
        length = np.linalg.norm(trip["nodes"].loc[1, ["x_m", "y_m"]].to_numpy(dtype=float) - trip["nodes"].loc[0, ["x_m", "y_m"]].to_numpy(dtype=float))
        self.assertTrue(70 < length < 73)

    def test_source_order_and_time_anomalies_retained(self):
        self.write("track", "8.0002 50 4\n8.0004 50 3\n8.0012 50 100\n")
        trip = load_trip(self.root, "train", 1)
        self.assertEqual(trip["observations"]["timestamp"].tolist(), [4, 3, 100])
        self.assertEqual(trip["validation"]["nonpositive_time_observation_ids"], [1])
        self.assertEqual(trip["validation"]["large_gap_observation_ids"], [2])

    def test_malformed_shapes_coordinates_and_ids_rejected(self):
        for text in ["8 50\n", "nan 50 1\n", "8 91 1\n", "8 50 1 4\n", "\n"]:
            with self.subTest(text=text):
                self.write("track", text)
                with self.assertRaises(ValueError):
                    load_trip(self.root, "train", 1)
        self.write("track", "8 50 1\n")
        for text in ["0 3\n", "0 -1\n", "0 1.5\n", "0 99999999999999999999999999\n"]:
            with self.subTest(text=text):
                self.write("arcs", text)
                with self.assertRaises(ValueError):
                    load_trip(self.root, "train", 1)

    def test_reference_ids_rejected_and_occurrences_preserved(self):
        trip = load_trip(self.root, "train", 1)
        self.write("route", "2\n")
        with self.assertRaises(ValueError):
            load_reference_route(self.root, "train", 1, arcs=trip["arcs"])
        self.write("route", "0\n1\n0\n")
        ref = load_reference_route(self.root, "train", 1, arcs=trip["arcs"])
        self.assertEqual(ref["arc_id"].tolist(), [0, 1, 0])
        self.assertEqual(ref["section_id"].tolist(), [0, 0, 1])

    def test_ordered_alignment_and_ambiguity(self):
        trip = load_trip(self.root, "train", 1)
        ref = load_reference_route(self.root, "train", 1, arcs=trip["arcs"])
        labels = align_reference_route(trip, ref)
        self.assertEqual(labels["reference_arc_id"].tolist(), [0, 0, 1])
        self.assertTrue((np.diff(labels["route_position"]) >= 0).all())
        self.assertEqual(labels["alignment_status"].tolist(), ["aligned"] * 3)
        self.write("track", "8.001 50 1\n8.001 50 100\n")
        trip = load_trip(self.root, "train", 1)
        labels = align_reference_route(trip, ref)
        self.assertTrue(labels["alignment_status"].str.contains("ambiguous_geometry").all())
        self.assertTrue(labels["alignment_status"].str.contains("gap_boundary").all())

    def test_repeated_occurrences_and_discontinuities_flagged(self):
        self.write("route", "0\n1\n0\n")
        trip = load_trip(self.root, "train", 1)
        ref = load_reference_route(self.root, "train", 1, arcs=trip["arcs"])
        labels = align_reference_route(trip, ref)
        repeated = labels["reference_arc_id"] == 0
        self.assertTrue(labels.loc[repeated, "alignment_status"].str.contains("repeated_arc_occurrence").all())
        self.assertTrue((np.diff(labels["route_position"]) >= 0).all())

    def test_distant_and_zero_length_labels_are_ineligible(self):
        self.write("track", "8 50.01 1\n")
        trip = load_trip(self.root, "train", 1)
        ref = load_reference_route(self.root, "train", 1, arcs=trip["arcs"])
        labels = align_reference_route(trip, ref)
        self.assertIn("too_far", labels.alignment_status.iloc[0])
        self.write("arcs", "0 0\n")
        self.write("route", "0\n")
        self.write("track", "8 50 1\n")
        trip = load_trip(self.root, "train", 1)
        ref = load_reference_route(self.root, "train", 1, arcs=trip["arcs"])
        labels = align_reference_route(trip, ref)
        self.assertIn("zero_length_segment", labels.alignment_status.iloc[0])

    def test_reserved_partition_and_path_traversal_rejected(self):
        for partition, tid in [("evaluator_test", "1"), ("train", "../1")]:
            with self.assertRaises(ValueError):
                load_trip(self.root, partition, tid)

    def test_manifest_reproducible_disjoint_and_complete(self):
        for i in range(80):
            (self.root / "train" / f"{i:08}").mkdir(exist_ok=True)
        for i in range(80, 90):
            (self.root / "student_test" / f"{i:08}").mkdir(parents=True)
        first = create_split_manifest(self.root)
        self.assertEqual(json.dumps(first), json.dumps(create_split_manifest(self.root)))
        self.assertEqual([len(first[k]) for k in ("train", "validation", "student_test")], [64, 16, 10])
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse((set(first["train"]) | set(first["validation"])) & set(first["student_test"]))


if __name__ == "__main__":
    unittest.main()
