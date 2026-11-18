"""Inference and server-only evaluation boundary shared by API and checks."""

import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from pathsnap.data.readers import load_trip_directory, _read
from pathsnap.data import align_reference_route
from pathsnap.candidates import build_segments, generate_candidates
from pathsnap.features import build_feature_table
from pathsnap.models import predict_candidates
from pathsnap.evaluation import top_candidates
from pathsnap.decoding import decode, sequence_metrics

ROOT = Path(__file__).resolve().parents[3]


def reference_for_evaluation(trip, directory):
    """Private labels: never included in UI payloads or inference features."""
    path = Path(directory) / f'{trip["trip_id"]}.route'
    if not path.exists():
        return None
    ids = _read(path, 1, integer=True).ravel()
    edges = trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    if ((ids < 0) | (ids >= len(edges))).any():
        raise ValueError("Reference IDs out of bounds")
    endpoints = edges[ids]
    breaks = np.r_[False, endpoints[:-1, 1] != endpoints[1:, 0]]
    ref = pd.DataFrame(
        {
            "route_position": np.arange(len(ids)),
            "arc_id": ids,
            "section_id": np.cumsum(breaks),
            "discontinuity_before": breaks,
        }
    )
    return align_reference_route(trip, ref)


def predict(trip, model, settings=None):
    segments = build_segments(trip["nodes"], trip["arcs"])
    candidates = generate_candidates(trip["observations"], segments)
    features = build_feature_table(
        trip["trip_id"], trip["observations"], candidates, segments
    )
    probabilities = predict_candidates(model, features)
    pointwise = top_candidates(features, probabilities)
    full = trip["observations"][["observation_id"]].copy()
    full.insert(0, "trip_id", trip["trip_id"])
    pointwise = full.merge(
        pointwise, on=["trip_id", "observation_id"], how="left", validate="one_to_one"
    )
    decoded = decode(trip, candidates, probabilities, **(settings or {}))
    return pointwise, decoded, features


def payload(trip, pointwise, decoded, metrics, elapsed, mode):
    obs = trip["observations"]
    nodes = trip["nodes"]
    edges = trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    xy = nodes[["x_m", "y_m"]].to_numpy()
    gps = obs[["x_m", "y_m"]].to_numpy()
    low = gps.min(axis=0) - 150
    high = gps.max(axis=0) + 150
    lines = xy[edges]
    mask = ((lines.max(axis=1) >= low) & (lines.min(axis=1) <= high)).all(axis=1)
    # Display thinning never changes the actual inference or returned arc IDs.
    roads = lines[mask]
    stride = max(1, (len(roads) + 7999) // 8000)
    roads = roads[::stride]

    def selected(frame):
        ids = frame.predicted_arc_id.dropna().astype(int).unique()
        return lines[ids].round(3).tolist()

    route_ids = sorted({arc for run in decoded["route_runs"] for arc in run})

    return {
        "trip_id": trip["trip_id"],
        "mode": mode,
        "label_basis": (
            "provisional_ordered_reference_alignment"
            if metrics
            else "reference unavailable"
        ),
        "observation_count": len(obs),
        "inference_s": elapsed,
        "metrics": metrics,
        "gps": gps.round(3).tolist(),
        "roads": roads.round(3).tolist(),
        "classifier_segments": selected(pointwise),
        "decoded_segments": lines[np.asarray(route_ids, dtype=int)].round(3).tolist(),
        "route_runs": decoded["route_runs"],
        "breaks": decoded["breaks"],
        "decoder_settings": decoded["settings"],
        "predictions": [
            {
                "observation_id": int(row.observation_id),
                "arc_id": (
                    None if pd.isna(row.predicted_arc_id) else int(row.predicted_arc_id)
                ),
            }
            for row in decoded["predictions"].itertuples()
        ],
        "crs": trip["crs"]["projected"],
    }
