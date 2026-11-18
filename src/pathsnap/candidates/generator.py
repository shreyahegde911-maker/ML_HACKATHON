"""Candidate road segments from observations, without using reference routes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CandidateConfig:
    search_radius_m: float = 100.0
    max_candidates: int = 8
    expanded_search_radius_m: float = 300.0

    def __post_init__(self):
        if (
            self.search_radius_m <= 0
            or self.expanded_search_radius_m < self.search_radius_m
        ):
            raise ValueError(
                "radii must be positive and expanded radius >= search radius"
            )
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least one")


def build_segments(nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    """Join endpoints into metric segments, retaining arc IDs and direction."""
    n = nodes.set_index("node_id")
    a = arcs.copy()
    a["_order"] = np.arange(len(a))
    start = n.reindex(a.from_node_id)
    end = n.reindex(a.to_node_id)
    good = (
        np.isfinite(start.x_m.to_numpy(dtype=float))
        & np.isfinite(start.y_m.to_numpy(dtype=float))
        & np.isfinite(end.x_m.to_numpy(dtype=float))
        & np.isfinite(end.y_m.to_numpy(dtype=float))
    )
    out = pd.DataFrame(
        {
            "arc_id": a.arc_id.to_numpy()[good],
            "from_node_id": a.from_node_id.to_numpy()[good],
            "to_node_id": a.to_node_id.to_numpy()[good],
            "x1": start.x_m.to_numpy()[good],
            "y1": start.y_m.to_numpy()[good],
            "x2": end.x_m.to_numpy()[good],
            "y2": end.y_m.to_numpy()[good],
        }
    )
    out["dx"] = out.x2 - out.x1
    out["dy"] = out.y2 - out.y1
    out["length_m"] = np.hypot(out.dx, out.dy)
    return out.reset_index(drop=True)


def _distances(px, py, seg):
    dx, dy = seg.dx.to_numpy(), seg.dy.to_numpy()
    denom = dx * dx + dy * dy
    t = np.divide(
        (px - seg.x1.to_numpy()) * dx + (py - seg.y1.to_numpy()) * dy,
        denom,
        out=np.zeros(len(seg), dtype=float),
        where=denom > 0,
    )
    t = np.clip(t, 0, 1)
    return np.hypot(
        px - (seg.x1.to_numpy() + t * dx), py - (seg.y1.to_numpy() + t * dy)
    )


def generate_candidates(
    observations: pd.DataFrame,
    segments: pd.DataFrame,
    config: CandidateConfig = CandidateConfig(),
) -> pd.DataFrame:
    """Return one row per point/arc within the radius, sorted nearest-first.

    A midpoint KD-tree queried at radius + half the longest segment is a
    conservative spatial index: every segment within the requested point
    distance is retrieved, then exact point-to-segment distance is applied.
    If no segment is within the main radius, expand to the configured fallback
    radius and mark ``search_expanded``. No candidate is emitted if that also
    finds none; observations with zero rows remain visible in observation-level
    coverage summaries. Candidate selection only sees points and network.
    """
    cols = ["observation_id", "arc_id", "distance_m", "search_expanded"]
    if observations.empty or segments.empty:
        return pd.DataFrame(columns=cols)
    seg = segments.reset_index(drop=True)
    # Length-stratified midpoint trees avoid letting one unusually long arc
    # inflate every query's radius while preserving a conservative bound.
    positive = seg.length_m[seg.length_m > 0]
    cuts = (
        np.unique(np.quantile(positive, [0.25, 0.5, 0.75]))
        if len(positive)
        else np.array([])
    )
    groups = np.searchsorted(cuts, seg.length_m.to_numpy(), side="left")
    indexes = []
    for group_id in np.unique(groups):
        ids = np.flatnonzero(groups == group_id)
        subset = seg.iloc[ids]
        centers = np.column_stack(
            ((subset.x1 + subset.x2) / 2, (subset.y1 + subset.y2) / 2)
        )
        indexes.append((ids, cKDTree(centers), float(subset.length_m.max()) / 2))
    rows = []
    arc_ids = seg.arc_id.to_numpy(dtype=np.int64)
    x1, y1 = seg.x1.to_numpy(), seg.y1.to_numpy()
    dx, dy = seg.dx.to_numpy(), seg.dy.to_numpy()
    denominator = dx * dx + dy * dy
    for p in observations.itertuples(index=False):
        px, py = float(p.x_m), float(p.y_m)
        if not np.isfinite(px) or not np.isfinite(py):
            continue
        retrieved = set()
        for global_ids, tree, half_max in indexes:
            local = tree.query_ball_point(
                [px, py], config.expanded_search_radius_m + half_max
            )
            retrieved.update(global_ids[i] for i in local)
        ids = np.asarray(sorted(retrieved), dtype=np.int64)
        if not len(ids):
            continue
        t = np.divide(
            (px - x1[ids]) * dx[ids] + (py - y1[ids]) * dy[ids],
            denominator[ids],
            out=np.zeros(len(ids)),
            where=denominator[ids] > 0,
        )
        t = np.clip(t, 0, 1)
        d = np.hypot(px - x1[ids] - t * dx[ids], py - y1[ids] - t * dy[ids])
        keep = d <= config.search_radius_m
        expanded = False
        if not keep.any():
            expanded = True
            keep = d <= config.expanded_search_radius_m
        chosen = np.flatnonzero(keep)
        chosen = chosen[
            np.lexsort((arc_ids[ids[chosen]], d[chosen]))[: config.max_candidates]
        ]
        for local in chosen:
            rows.append(
                (
                    int(p.observation_id),
                    int(arc_ids[ids[local]]),
                    float(d[local]),
                    expanded,
                )
            )
    return pd.DataFrame(rows, columns=cols)
