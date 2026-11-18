"""Five-model comparison, TPE Bayesian search and final artifact export."""

import os, sys, json, time, hashlib, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import joblib, numpy as np, pandas as pd, optuna
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from pathsnap.models import make_models, predict_candidates
from pathsnap.features.engineering import FEATURES
from pathsnap.evaluation import evaluate_classifier
from pathsnap.data.readers import load_trip_directory
from pathsnap.decoding import sequence_metrics

OUT = ROOT / "results/final"
ART = ROOT / "artifacts/final"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def save_model(value, path):
    temporary = path.with_suffix(".joblib.tmp")
    try:
        joblib.dump(value, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/features")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Student folder")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "results/part1/split_manifest.json").read_text())
    input_files = []

    def read(p, s):
        options = [args.features / f"{p}_{s}.csv", args.features / f"{p}_{s}.csv.gz"]
        path = next((candidate for candidate in options if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing shared table: {options}")
        input_files.append(path)
        frame = pd.read_csv(path, dtype={"trip_id": str})
        if not set(frame.trip_id).issubset(manifest[p]):
            raise ValueError(f"{p}: trip IDs do not match the shared manifest")
        return frame

    train = pd.concat(
        [read(p, "supervised") for p in ["train", "validation"]], ignore_index=True
    )
    dev = read("student_test", "supervised")
    coverage = read("student_test", "reference_coverage")
    if set(train.trip_id) & set(dev.trip_id):
        raise ValueError("Training and student-development trips overlap")
    for name, frame in [("training", train), ("development", dev)]:
        if not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError(f"{name}: nonfinite model features")
        if not frame.label.dropna().isin([0, 1]).all():
            raise ValueError(f"{name}: labels must be binary or unset")
    train = train.loc[train.label.notna()].reset_index(drop=True)
    X = train[FEATURES]
    y = train.label.to_numpy(dtype=int)
    groups = train.trip_id
    folds = list(GroupKFold(3).split(X, y, groups))
    development_trips = {
        tid: load_trip_directory(args.data_root / "student_test" / tid)
        for tid in sorted(coverage.trip_id.unique())
    }
    dump(
        OUT / "folds.json",
        [
            {
                "fold": i + 1,
                "fit_trips": sorted(groups.iloc[a].unique()),
                "validation_trips": sorted(groups.iloc[b].unique()),
            }
            for i, (a, b) in enumerate(folds)
        ],
    )
    models = make_models(jobs=2)
    models.update(
        {
            "decision_tree": DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
            ),
            "gaussian_naive_bayes": GaussianNB(),
        }
    )
    results = []
    cv_rows = []
    configs = {}
    selected_scores = {}

    def cross(model, indices=None):
        scores = []
        for i, (a, b) in enumerate(folds):
            if indices is not None:
                a = np.intersect1d(a, indices)
                b = np.intersect1d(b, indices)
            m = clone(model)
            m.fit(X.iloc[a], y[a])
            scores.append(
                float(
                    f1_score(
                        y[b], m.predict(X.iloc[b]), average="macro", zero_division=0
                    )
                )
            )
        return scores

    def evaluate(name, model, cv):
        tick = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - tick
        save_model(
            {
                "model": model,
                "features": FEATURES,
                "training_trip_ids": sorted(groups.unique()),
                "label_source": "provisional_ordered_reference_alignment",
            },
            ART / f"{name}.joblib",
        )
        tick = time.perf_counter()
        p = predict_candidates(model, dev)
        infer_s = time.perf_counter() - tick
        metrics, predictions, per_trip = evaluate_classifier(dev, coverage, p)
        metrics.update(
            model=name,
            cv_macro_f1=float(np.mean(cv)),
            cv_std=float(np.std(cv)),
            fit_s=fit_s,
            inference_s=infer_s,
        )
        # Training score is a measured overfit diagnostic, never a selection objective.
        metrics["training_candidate_macro_f1"] = float(
            f1_score(y, model.predict(X), average="macro", zero_division=0)
        )
        continuity = [
            sequence_metrics(
                trip,
                predictions.loc[predictions.trip_id.eq(tid)].sort_values(
                    "observation_id"
                ),
            )
            for tid, trip in development_trips.items()
        ]
        pairs = sum(row["continuity_pairs"] for row in continuity)
        metrics["direct_step_continuity"] = (
            sum(
                (row["direct_step_continuity"] or 0.0) * row["continuity_pairs"]
                for row in continuity
            )
            / pairs
            if pairs
            else None
        )
        results.append(metrics)
        predictions.to_csv(
            OUT / f"{name}_predictions.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        per_trip.to_csv(OUT / f"{name}_per_trip.csv", index=False)
        dump(OUT / "metrics.json", results)
        pd.DataFrame(results).drop(columns="candidate_confusion_matrix").to_csv(
            OUT / "comparison.csv", index=False
        )
        print(
            name,
            "CV",
            metrics["cv_macro_f1"],
            "development Top1",
            metrics["observation_top1_accuracy"],
            flush=True,
        )

    for name, model in models.items():
        configs[name] = {
            k: v
            for k, v in model.get_params(deep=True).items()
            if isinstance(v, (str, float, int, bool, type(None)))
        }
        tick = time.perf_counter()
        cv = cross(model)
        cv_rows.append(
            {
                "model": name,
                "fold_macro_f1": cv,
                "elapsed_s": time.perf_counter() - tick,
            }
        )
        dump(OUT / "cross_validation.json", cv_rows)
        selected_scores[name] = float(np.mean(cv))
        evaluate(name, model, cv)
    winner = max(selected_scores, key=selected_scores.get)
    base = models[winner]
    # Fixed per-trip search sample limits cost without using development data.
    rng = np.random.default_rng(42)
    sample = np.sort(
        np.concatenate(
            [
                rng.choice(g.index, min(len(g), 1200), replace=False)
                for _, g in train.groupby("trip_id")
            ]
        )
    )

    def parameters(trial):
        if winner == "random_forest":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 40, 100, step=20),
                max_depth=trial.suggest_int("max_depth", 8, 18),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 80),
                max_features=trial.suggest_float("max_features", 0.5, 1.0),
            )
        if winner == "hist_gradient_boosting":
            return dict(
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 31),
                max_iter=trial.suggest_int("max_iter", 60, 160, step=20),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.03, 0.18, log=True
                ),
                l2_regularization=trial.suggest_float(
                    "l2_regularization", 0.001, 10.0, log=True
                ),
            )
        if winner == "decision_tree":
            return dict(
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100),
            )
        if winner == "logistic_regression":
            return {
                "logisticregression__C": trial.suggest_float("C", 0.01, 100, log=True)
            }
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-4, log=True)
        }

    def objective(trial):
        m = clone(base).set_params(**parameters(trial))
        return float(np.mean(cross(m, sample)))

    # A study belongs to exact data/model/folds/code, not just a fixed name.
    context = {
        "winner": winner,
        "model_configuration": configs[winner],
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_files
        },
        "folds": json.loads((OUT / "folds.json").read_text()),
        "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()
    context["fingerprint"] = fingerprint
    dump(OUT / "trial_context.json", context)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3),
        study_name=f"pathsnap_{fingerprint[:16]}",
        storage=f'sqlite:///{OUT/"bayesian_search.sqlite3"}',
        load_if_exists=True,
    )
    study.set_user_attr("context_fingerprint", fingerprint)
    search_start = time.perf_counter()
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, 10 - completed))
    # Trial names differ from sklearn's nested LR parameter name.
    best = study.best_params.copy()
    if winner == "logistic_regression":
        best = {"logisticregression__C": best["C"]}
    tuned = clone(base).set_params(**best)
    tuned_cv = cross(tuned)
    cv_rows.append({"model": winner + "_tuned", "fold_macro_f1": tuned_cv})
    dump(OUT / "cross_validation.json", cv_rows)
    evaluate(winner + "_tuned", tuned, tuned_cv)
    study.trials_dataframe().to_csv(OUT / "bayesian_trials.csv", index=False)
    save_model(
        {
            "model": tuned,
            "features": FEATURES,
            "training_trip_ids": sorted(groups.unique()),
            "label_source": "provisional_ordered_reference_alignment",
            "name": winner + "_tuned",
        },
        ART / "selected.joblib",
    )
    dump(
        OUT / "selection.json",
        {
            "baseline_winner": winner,
            "selection_metric": "3-fold training-only candidate Macro F1 on all available labelled training trips",
            "best_params": best,
            "search_objective": study.best_value,
            "search_sample_rows": len(sample),
            "search_cap_per_trip": 1200,
            "search_trials": 10,
            "study_name": study.study_name,
            "context_fingerprint": fingerprint,
            "search_elapsed_s": time.perf_counter() - search_start,
            "untuned_cv": selected_scores[winner],
            "tuned_full_cv": float(np.mean(tuned_cv)),
            "fit_candidate_rows": len(train),
            "fit_trips": sorted(groups.unique()),
            "excluded_no_label_trip": "00000011",
            "tuned_model_exported": True,
            "label_warning": "provisional derived supervision, not official pointwise ground truth",
        },
    )
    dump(OUT / "model_configurations.json", configs)
    dump(
        OUT / "artifact_checksums.json",
        [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(ART.glob("*.joblib"))
        ],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, col, title in zip(
        axes,
        [
            "cv_macro_f1",
            "observation_top1_accuracy",
            "candidate_macro_f1",
            "candidate_log_loss",
        ],
        [
            "Trip-grouped CV Macro F1",
            "Development Top-1",
            "Development candidate F1",
            "Development log loss",
        ],
    ):
        ax.bar(table.model, table[col], color=["#2563eb"] * 5 + ["#7c3aed"])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=75)
    fig.suptitle("Measured five-model comparison and Bayesian-tuned winner")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=140)
    plt.close(fig)
    print("FINAL_MODEL_READY", winner, flush=True)


if __name__ == "__main__":
    main()
