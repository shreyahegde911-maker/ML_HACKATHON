#!/usr/bin/env python3
"""Train Parts 3/4 on the shared trip split and record actual experiments."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from pathsnap.features.engineering import FEATURES
from pathsnap.models import make_models, predict_candidates
from pathsnap.evaluation import evaluate_classifier


def read_split(directory, partition):
    paths = [directory / f"{partition}_supervised.csv", directory / f"{partition}_supervised.csv.gz"]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError(paths)
    frame = pd.read_csv(path, dtype={"trip_id": str})
    cpaths = [directory / f"{partition}_reference_coverage.csv", directory / f"{partition}_reference_coverage.csv.gz"]
    coverage = pd.read_csv(next(p for p in cpaths if p.exists()), dtype={"trip_id": str})
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError(f"{partition}: nonfinite features")
    if frame.duplicated(["trip_id", "observation_id", "arc_id"]).any():
        raise ValueError(f"{partition}: duplicate candidates")
    return frame, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "cache/candidate_features")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baselines")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/models")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/part1/split_manifest.json")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    data = {p: read_split(args.features, p) for p in ("train", "validation", "student_test")}
    for partition, (features, coverage) in data.items():
        if set(coverage.trip_id) != set(manifest[partition]) or not set(features.trip_id).issubset(manifest[partition]):
            raise ValueError(f"{partition}: manifest mismatch")
    train = data["train"][0]
    labelled = train.loc[train.label.notna()].reset_index(drop=True)
    x, y, groups = labelled[FEATURES], labelled.label.to_numpy(dtype=int), labelled.trip_id
    if len(np.unique(y)) != 2:
        raise ValueError("Training must contain both candidate classes")
    cv_results, metrics, configs = [], [], {}
    from sklearn.metrics import f1_score
    for name, model in make_models(jobs=args.jobs).items():
        configs[name] = {k: v for k, v in model.get_params(deep=True).items() if isinstance(v, (str, int, float, bool, type(None)))}
        print(f"{name}: {len(labelled)} labelled candidates across {groups.nunique()} training trips", flush=True)
        for fold, (fit, valid) in enumerate(GroupKFold(n_splits=args.cv_folds).split(x, y, groups), 1):
            fold_model = clone(model)
            tick = time.perf_counter()
            fold_model.fit(x.iloc[fit], y[fit])
            f1 = float(f1_score(y[valid], fold_model.predict(x.iloc[valid]), average="macro", zero_division=0))
            cv_results.append({"model": name, "fold": fold, "candidate_macro_f1": f1,
                               "elapsed_s": time.perf_counter() - tick,
                               "fit_trips": sorted(groups.iloc[fit].unique()), "validation_trips": sorted(groups.iloc[valid].unique())})
            print(f"  group CV fold {fold}: Macro F1={f1:.4f}", flush=True)
        tick = time.perf_counter()
        model.fit(x, y)
        training_s = time.perf_counter() - tick
        model_path = args.artifacts / f"{name}.joblib"
        joblib.dump({"model": model, "features": FEATURES, "label_source": "provisional_ordered_reference_alignment", "training_trip_ids": sorted(groups.unique()), "manifest_training_trip_ids": manifest["train"]}, model_path, compress=3)
        restored = joblib.load(model_path)["model"]
        np.testing.assert_allclose(predict_candidates(model, train.head(100)), predict_candidates(restored, train.head(100)))
        for partition, (features, coverage) in data.items():
            tick = time.perf_counter()
            probabilities = predict_candidates(model, features)
            inference_s = time.perf_counter() - tick
            result, observations, per_trip = evaluate_classifier(features, coverage, probabilities)
            result.update({"model": name, "partition": partition, "training_s": training_s, "inference_s": inference_s,
                           "cv_macro_f1_mean": float(np.mean([r["candidate_macro_f1"] for r in cv_results if r["model"] == name])),
                           "cv_macro_f1_std": float(np.std([r["candidate_macro_f1"] for r in cv_results if r["model"] == name]))})
            metrics.append(result)
            observations.to_csv(args.output / f"{name}_{partition}_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            per_trip.to_csv(args.output / f"{name}_{partition}_per_trip.csv", index=False)
            print(f"  {partition}: Top-1={result['observation_top1_accuracy']:.4f}, candidate Macro F1={result['candidate_macro_f1']:.4f}", flush=True)
        pd.DataFrame(metrics).drop(columns="candidate_confusion_matrix").to_csv(args.output / "comparison.csv", index=False)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (args.output / "cross_validation.json").write_text(json.dumps(cv_results, indent=2) + "\n")
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyproj", "folium", "matplotlib"]}
    record = {"seed": 42, "features": FEATURES, "cv_folds": args.cv_folds, "models": configs,
              "effective_training_trip_ids": sorted(groups.unique()),
              "training_trips_without_eligible_candidate_labels": sorted(set(manifest["train"]) - set(groups.unique())),
              "versions": versions, "python": sys.version,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "label_source": "provisional_ordered_reference_alignment; not official ground truth",
              "selection": "Compare validation evidence; student_test is reported separately, evaluator data excluded.",
              "command": ["python", "scripts/train_baselines.py", *sys.argv[1:]]}
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.DataFrame(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ["observation_top1_accuracy", "candidate_macro_f1", "candidate_log_loss"], ["Observation Top-1", "Candidate Macro F1", "Candidate log loss"]):
        table.loc[table.partition.ne("train")].pivot(index="model", columns="partition", values=metric).plot.bar(ax=ax, rot=15)
        ax.set_title(title); ax.set_xlabel("")
    fig.suptitle("Two-model baseline: provisional reference labels")
    fig.tight_layout(); fig.savefig(args.output / "comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
