#!/usr/bin/env python3
"""Run either saved classifier on raw GPS/network inputs without references."""

import argparse
from pathlib import Path
import sys
import time
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import load_trip
from pathsnap.candidates import build_segments, generate_candidates
from pathsnap.features import build_feature_table
from pathsnap.models import predict_candidates
from pathsnap.evaluation import top_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default="random_forest",
    )
    parser.add_argument(
        "--partition", choices=["train", "student_test"], default="student_test"
    )
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Student folder")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/predictions.csv")
    args = parser.parse_args()
    tick = time.perf_counter()
    bundle = joblib.load(ROOT / "artifacts/models" / f"{args.model}.joblib")
    trip = load_trip(args.data_root, args.partition, args.trip_id)
    segments = build_segments(trip["nodes"], trip["arcs"])
    candidates = generate_candidates(trip["observations"], segments)
    features = build_feature_table(
        trip["trip_id"], trip["observations"], candidates, segments
    )
    predictions = top_candidates(
        features, predict_candidates(bundle["model"], features)
    )
    observations = trip["observations"][["observation_id"]].copy()
    observations.insert(0, "trip_id", trip["trip_id"])
    output = observations.merge(
        predictions, on=["trip_id", "observation_id"], how="left", validate="one_to_one"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        f"{args.model}: {len(output)} observations, {len(features)} candidates, {time.perf_counter()-tick:.3f}s end-to-end; saved {args.output}"
    )


if __name__ == "__main__":
    main()
