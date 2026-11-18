"""Shared LR/RF features; identifiers and status columns are metadata."""

import numpy as np
import pandas as pd

FEATURES = ["distance_m", "heading_difference", "speed_mps", "time_gap_s"]


def _same_id(left, right):
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def build_feature_table(trip_id, observations, candidates, segments):
    """Heading difference is absolute, wrapped degrees in [0,180]."""
    # Part 1 preserves source order. Sorting would conceal backwards timestamps
    # and calculate movement between observations that were not consecutive.
    obs = observations.copy()
    dt = pd.to_numeric(obs.timestamp, errors="coerce").diff()
    dx, dy = obs.x_m.diff(), obs.y_m.diff()
    displacement = np.hypot(dx, dy)
    valid_dt = dt.where(np.isfinite(dt) & (dt > 0))
    speed = (displacement / valid_dt).where(valid_dt.notna())
    moving = (displacement > 1e-6) & valid_dt.notna()
    move_heading = np.degrees(np.arctan2(dx, dy)) % 360.0  # clockwise from north
    obs["_speed"] = speed.where(moving, 0.0).fillna(0.0)
    obs["_dt"] = valid_dt.fillna(0.0)
    obs["_move_heading"] = move_heading.where(moving)
    out = candidates.merge(
        obs[["observation_id", "_speed", "_dt", "_move_heading"]],
        on="observation_id",
        how="left",
    )
    s = segments[["arc_id", "dx", "dy", "length_m"]].drop_duplicates("arc_id")
    out = out.merge(s, on="arc_id", how="left")
    arc_heading = np.degrees(np.arctan2(out.dx, out.dy)) % 360.0
    raw = ((out._move_heading - arc_heading + 180.0) % 360.0) - 180.0
    out["heading_difference"] = (
        np.abs(raw).where(out.length_m > 1e-9).where(out._move_heading.notna(), 0.0)
    )
    out["distance_m"] = out.distance_m.astype(float)
    out["speed_mps"] = out._speed.astype(float)
    out["time_gap_s"] = out._dt.astype(float)
    out.insert(0, "trip_id", str(trip_id))
    return (
        out[["trip_id", "observation_id", "arc_id", *FEATURES, "search_expanded"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(
            {
                "distance_m": 0.0,
                "heading_difference": 0.0,
                "speed_mps": 0.0,
                "time_gap_s": 0.0,
            }
        )
    )


def attach_reference_labels(
    features, references, reliable_statuses=("aligned", "reliable", "ok")
):
    """Join optional observation labels; never adds candidates for missing arcs."""
    ref = references[
        ["trip_id", "observation_id", "reference_arc_id", "alignment_status"]
    ].copy()
    out = features.merge(
        ref, on=["trip_id", "observation_id"], how="left", validate="many_to_one"
    )
    reliable_values = {str(s).lower() for s in reliable_statuses}
    reliable = (
        out.alignment_status.astype("string").str.lower().isin(reliable_values)
        & out.reference_arc_id.notna()
    )
    out["label"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[reliable, "label"] = (
        out.loc[reliable, "arc_id"].to_numpy()
        == out.loc[reliable, "reference_arc_id"].to_numpy()
    ).astype("int64")
    out["label_reliable"] = reliable
    matches = out["label"].eq(1).fillna(False)
    has_match = matches.groupby([out.trip_id, out.observation_id]).transform("any")
    out["reference_arc_missing"] = reliable & ~has_match
    return out


def reference_candidate_coverage(
    features, references, reliable_statuses=("aligned", "reliable", "ok")
):
    """One row per reference observation, retaining total candidate misses."""
    ref = references[
        ["trip_id", "observation_id", "reference_arc_id", "alignment_status"]
    ].copy()
    reliable_values = {str(s).lower() for s in reliable_statuses}
    reliable = (
        ref.alignment_status.astype("string").str.lower().isin(reliable_values)
        & ref.reference_arc_id.notna()
    )
    ref["label_reliable"] = reliable.to_numpy()
    if ref.duplicated(["trip_id", "observation_id"]).any():
        raise ValueError("Reference observation keys must be unique")
    counts = (
        features.groupby(["trip_id", "observation_id"]).size().rename("candidate_count")
    )
    out = ref.join(counts, on=["trip_id", "observation_id"])
    out["candidate_count"] = out.candidate_count.fillna(0).astype(int)
    matched = (
        features[["trip_id", "observation_id", "arc_id"]]
        .drop_duplicates()
        .rename(columns={"arc_id": "reference_arc_id"})
    )
    matched["_matched"] = True
    out = out.merge(
        matched,
        on=["trip_id", "observation_id", "reference_arc_id"],
        how="left",
        validate="one_to_one",
    )
    out["reference_arc_missing"] = out.label_reliable & ~out["_matched"].eq(True)
    out["has_candidates"] = out.candidate_count > 0
    out = out.drop(columns="_matched")
    return out
