import sys, unittest, tempfile, os
from unittest.mock import patch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from pathsnap.api.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"PATHSNAP_LOCAL_CONFIG": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = TestClient(app)

    def test_catalog_and_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(len(self.client.get("/trips").json()["trips"]), 10)
        self.assertEqual(self.client.get("/vendor/leaflet/leaflet.js").status_code, 200)

    def test_unknown_and_invalid_trips(self):
        self.assertEqual(
            self.client.post("/predict", json={"trip_id": "99999999"}).status_code, 404
        )
        self.assertEqual(
            self.client.post("/predict", json={"trip_id": "../../etc"}).status_code, 422
        )

    def test_separate_ui_origin(self):
        r = self.client.options(
            "/predict",
            headers={
                "Origin": "http://127.0.0.1:8001",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], "*")

    def test_external_evaluator_root_predicts_without_reference(self):
        class Model:
            def predict_proba(self, x):
                return np.tile([0.3, 0.7], (len(x), 1))

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "00000100"
            directory.mkdir()
            for suffix, content in {
                "nodes": "8 50\n8.001 50\n",
                "arcs": "0 1\n",
                "track": "8.0001 50 1\n8.0002 50 2\n",
            }.items():
                (directory / f"00000100.{suffix}").write_text(content)
            with patch.dict(os.environ, {"PATHSNAP_EVALUATOR_ROOT": temp}), patch(
                "pathsnap.api.main.bundle", return_value={"model": Model()}
            ):
                response = self.client.post("/predict", json={"trip_id": "00000100"})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIsNone(data["metrics"])
                self.assertEqual(data["mode"], "reserved-evaluator")
                self.assertEqual(len(data["predictions"]), 2)
                self.assertNotIn("reference_arc_id", str(data))
                geometry = data["map_geojson"]
                self.assertEqual(geometry["gps"]["type"], "MultiPoint")
                self.assertEqual(geometry["gps"]["coordinates"][0], [8.0001, 50.0])
                self.assertEqual(geometry["roads"]["coordinates"][0][0], [8.0, 50.0])
                self.assertEqual(geometry["decoded_segments"]["type"], "MultiLineString")
                with patch("pathsnap.api.main.bundle", return_value={"model": Model(), "features": ["distance_m", "heading_difference", "speed_mps", "time_gap_s"]}):
                    health = self.client.get("/health").json()
                self.assertEqual(health["trip_count"], 1)
                self.assertEqual(health["status"], "ok")
                self.assertEqual(self.client.get("/trips").json()["trip_count"], 1)

    def test_reference_is_opened_only_after_prediction(self):
        import pathsnap.api.main as api
        class Model:
            def predict_proba(self, x):
                return np.tile([0.3, 0.7], (len(x), 1))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "123"
            directory.mkdir()
            for suffix, content in {"nodes": "8 50\n8.001 50\n", "arcs": "0 1\n", "track": "8.0001 50 1\n8.0002 50 2\n", "route": "0\n"}.items():
                (directory / f"123.{suffix}").write_text(content)
            actual_predict, actual_open = api.predict, Path.open
            stage = {"prediction_complete": False, "reference_opened": False}
            def guarded_open(path, *args, **kwargs):
                if path.suffix == ".route":
                    self.assertTrue(stage["prediction_complete"])
                    stage["reference_opened"] = True
                return actual_open(path, *args, **kwargs)
            def prediction(*args, **kwargs):
                output = actual_predict(*args, **kwargs)
                stage["prediction_complete"] = True
                return output
            with patch.dict(os.environ, {"PATHSNAP_EVALUATOR_ROOT": temp}), patch.object(api, "bundle", return_value={"model": Model()}), patch.object(api, "predict", side_effect=prediction), patch.object(Path, "open", guarded_open):
                response = self.client.post("/predict", json={"trip_id": "123"})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(stage["reference_opened"])
                self.assertNotIn("reference_arc_id", response.text)

    def test_bad_optional_reference_does_not_discard_prediction(self):
        class Model:
            def predict_proba(self, x):
                return np.tile([0.3, 0.7], (len(x), 1))

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "00000100"
            directory.mkdir()
            for suffix, content in {
                "nodes": "8 50\n8.001 50\n",
                "arcs": "0 1\n",
                "track": "8.0001 50 1\n8.0002 50 2\n",
                "route": "999999\n",
            }.items():
                (directory / f"00000100.{suffix}").write_text(content)
            with patch.dict(os.environ, {"PATHSNAP_EVALUATOR_ROOT": temp}), patch(
                "pathsnap.api.main.bundle", return_value={"model": Model()}
            ):
                response = self.client.post("/predict", json={"trip_id": "00000100"})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(len(data["predictions"]), 2)
                self.assertIsNone(data["metrics"])
                self.assertIn("evaluation_warning", data)
                self.assertEqual({p["arc_id"] for p in data["predictions"]}, {0})
                self.assertNotIn("reference_arc_id", data)
                self.assertNotIn("999999", data["evaluation_warning"])

    def test_health_checks_model_contents_not_just_file_existence(self):
        with patch(
            "pathsnap.api.main.bundle", side_effect=ValueError("corrupt artifact")
        ):
            health = self.client.get("/health").json()
            self.assertFalse(health["model_ready"])
            self.assertEqual(health["status"], "not_ready")

    def test_model_cache_reloads_after_artifact_replacement(self):
        import pathsnap.api.main as api

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "artifacts/final/selected.joblib"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"a")
            try:
                with patch.object(api, "ROOT", root), patch.object(
                    api.joblib, "load", side_effect=[{"version": 1}, {"version": 2}]
                ) as load:
                    self.assertEqual(api.bundle()["version"], 1)
                    self.assertEqual(api.bundle()["version"], 1)
                    path.write_bytes(b"new")
                    self.assertEqual(api.bundle()["version"], 2)
                    self.assertEqual(load.call_count, 2)
            finally:
                api._load_bundle.cache_clear()


if __name__ == "__main__":
    unittest.main()
