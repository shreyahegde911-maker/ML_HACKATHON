"""Separate binary candidate metrics and observation-level ranking metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def top_candidates(features, probabilities):
    if len(features) != len(probabilities):
        raise ValueError("One probability is required per candidate row")
    if features.empty:
        return pd.DataFrame(
            columns=["trip_id", "observation_id", "predicted_arc_id", "score"]
        )
    scored = features[["trip_id", "observation_id", "arc_id"]].copy()
    scored["score"] = probabilities
    # Stable input ordering resolves ties deterministically without references.
    best = scored.groupby(["trip_id", "observation_id"], sort=False).score.idxmax()
    return (
        scored.loc[best]
        .rename(columns={"arc_id": "predicted_arc_id"})
        .reset_index(drop=True)
    )


def evaluate_classifier(features, coverage, probabilities):
    eligible = features.label.notna()
    y = features.loc[eligible, "label"].to_numpy(dtype=int)
    p = np.asarray(probabilities)[eligible.to_numpy()]
    predicted = p >= 0.5
    candidate = (
        {
            "candidate_accuracy": float(accuracy_score(y, predicted)),
            "candidate_macro_f1": float(
                f1_score(y, predicted, average="macro", zero_division=0)
            ),
            "candidate_macro_precision": float(
                precision_score(y, predicted, average="macro", zero_division=0)
            ),
            "candidate_macro_recall": float(
                recall_score(y, predicted, average="macro", zero_division=0)
            ),
            "candidate_log_loss": float(
                log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])
            ),
            "candidate_confusion_matrix": confusion_matrix(
                y, predicted, labels=[0, 1]
            ).tolist(),
            "labelled_candidate_rows": len(y),
        }
        if len(y)
        else {"labelled_candidate_rows": 0}
    )
    winners = top_candidates(features, probabilities)
    observations = coverage.merge(
        winners, on=["trip_id", "observation_id"], how="left", validate="one_to_one"
    )
    reliable = observations.label_reliable.astype(bool)
    correct = observations.predicted_arc_id.eq(observations.reference_arc_id) & reliable
    observations["correct_on_provisional_reference"] = correct
    n = int(reliable.sum())
    candidate.update(
        {
            "observations": len(observations),
            "eligible_observations": n,
            "alignment_coverage": n / len(observations) if len(observations) else None,
            "observation_top1_accuracy": float(correct.sum() / n) if n else None,
            "candidate_miss_rate": (
                float(observations.loc[reliable, "reference_arc_missing"].mean())
                if n
                else None
            ),
            "observations_without_candidates": int(
                observations.predicted_arc_id.isna().sum()
            ),
        }
    )
    per_trip = []
    for trip_id, group in observations.groupby("trip_id", sort=True):
        valid = group.label_reliable.astype(bool)
        per_trip.append(
            {
                "trip_id": trip_id,
                "observations": len(group),
                "eligible_observations": int(valid.sum()),
                "top1_accuracy": (
                    float(group.loc[valid, "correct_on_provisional_reference"].mean())
                    if valid.any()
                    else None
                ),
            }
        )
    return candidate, observations, pd.DataFrame(per_trip)
