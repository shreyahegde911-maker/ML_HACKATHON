"""Exercise all ten configured trips over real HTTP; save evidence, not labels."""

import argparse, json, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    a = p.parse_args()
    records = []
    with httpx.Client(base_url=a.url, timeout=180) as client:
        health = client.get("/health")
        health.raise_for_status()
        catalog = client.get("/trips").json()
        assert len(catalog["trips"]) == 10
        assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
        assert client.post("/predict", json={"trip_id": "../x"}).status_code == 422
        for tid in catalog["trips"]:
            tick = time.perf_counter()
            r = client.post("/predict", json={"trip_id": tid})
            r.raise_for_status()
            data = r.json()
            assert (
                data["trip_id"] == tid
                and len(data["predictions"]) == data["observation_count"]
            )

            def inspect(value):
                if isinstance(value, dict):
                    assert not (
                        set(value)
                        & {
                            "reference_arc_id",
                            "alignment_status",
                            "reference_route",
                            "ground_truth",
                        }
                    )
                    for v in value.values():
                        inspect(v)
                elif isinstance(value, list):
                    for v in value:
                        inspect(v)

            inspect(data)
            records.append(
                {
                    "trip_id": tid,
                    "status": r.status_code,
                    "observations": data["observation_count"],
                    "elapsed_s": time.perf_counter() - tick,
                    "metrics": data["metrics"],
                }
            )
            print(json.dumps(records[-1]), flush=True)
    out = ROOT / "results/final/api_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "health": health.json(),
                "trips": records,
                "unknown_trip_status": 404,
                "invalid_id_status": 422,
                "reference_payload_leak_check": "passed",
                "scope": "local HTTP; a second physical laptop was not accessed",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
