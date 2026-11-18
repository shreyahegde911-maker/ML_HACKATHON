import sys, unittest, tempfile, os
from unittest.mock import patch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from pathsnap.api.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_catalog_and_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(len(self.client.get("/trips").json()["trips"]), 10)

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


if __name__ == "__main__":
    unittest.main()
