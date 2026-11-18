"""Provisional reference supervision, NOT the production HMM/decoder.

This routine consumes the supplied ordered reference route. It must never be
called by candidate generation or the prediction pipeline.
"""

import numpy as np
import pandas as pd


def align_reference_route(
    trip,
    reference,
    *,
    max_distance_m=75.0,
    ambiguity_margin_m=5.0,
    gap_threshold_s=30.0,
    max_progress_speed_mps=60.0,
    distance_scale_m=25.0,
    progress_penalty=0.05,
    batch_size=256
):
    """Monotone dynamic-programming alignment to reference arc occurrences.

    Starts/ends may be anywhere along the reference (semiglobal). State indices
    never decrease; repeated arc occurrences stay distinct. Forward movement
    has a small distance/time penalty, relaxed across long/invalid time gaps.
    Alignment status 'aligned' means eligible provisional supervision only.
    Other statuses must be excluded from supervised fitting/scoring.
    """
    if (
        max_distance_m <= 0
        or ambiguity_margin_m < 0
        or gap_threshold_s <= 0
        or distance_scale_m <= 0
        or max_progress_speed_mps <= 0
        or progress_penalty < 0
        or batch_size < 1
    ):
        raise ValueError("Invalid alignment configuration")
    obs, nodes, arcs = trip["observations"], trip["nodes"], trip["arcs"]
    if not len(obs) or not len(reference):
        raise ValueError("Alignment requires nonempty observations and reference")
    positions = reference["route_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(len(reference))):
        raise ValueError("Reference must retain ordered route occurrences")
    ids = reference["arc_id"].to_numpy(dtype=np.int64)
    if ((ids < 0) | (ids >= len(arcs))).any():
        raise ValueError("Reference arc ID out of bounds")
    endpoints = arcs.iloc[ids][["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    xy = nodes[["x_m", "y_m"]].to_numpy()
    starts, ends = xy[endpoints[:, 0]], xy[endpoints[:, 1]]
    vectors = ends - starts
    length2 = (vectors**2).sum(axis=1)
    lengths = np.sqrt(length2)
    progress = np.r_[0.0, np.cumsum(lengths)[:-1]]
    points = obs[["x_m", "y_m"]].to_numpy()
    timestamps = obs["timestamp"].to_numpy()
    n, m = len(points), len(ids)
    # Disk outputs are small; memory is O(observations * reference occurrences)
    # for backpointers, plus a bounded geometric-distance batch.
    back = np.empty((n, m), dtype=np.int32)
    previous = np.zeros(m)
    state_indices = np.arange(m, dtype=np.int32)

    def geometry(chunk):
        offset = chunk[:, None, :] - starts[None, :, :]
        fraction = np.clip(
            (offset * vectors).sum(axis=2) / np.where(length2 > 0, length2, 1), 0, 1
        )
        distance = np.linalg.norm(offset - fraction[:, :, None] * vectors, axis=2)
        return distance, fraction

    for base in range(0, n, batch_size):
        distance, _ = geometry(points[base : base + batch_size])
        for local, distances in enumerate(distance):
            i = base + local
            if i == 0:
                back[i] = state_indices
                current = np.minimum((distances / distance_scale_m) ** 2, 10000)
            else:
                dt = timestamps[i] - timestamps[i - 1]
                coefficient = (
                    progress_penalty / (max_progress_speed_mps * dt)
                    if 0 < dt <= gap_threshold_s
                    else 0.0
                )
                transformed = previous - coefficient * progress
                best = np.minimum.accumulate(transformed)
                source = np.maximum.accumulate(
                    np.where(transformed == best, state_indices, -1)
                )
                back[i] = source
                current = (
                    np.minimum((distances / distance_scale_m) ** 2, 10000)
                    + best
                    + coefficient * progress
                )
            previous = current - current.min()
    selected = np.empty(n, dtype=np.int32)
    selected[-1] = np.argmin(previous)
    for i in range(n - 1, 0, -1):
        selected[i - 1] = back[i, selected[i]]
    del back

    distances_out = np.empty(n)
    fractions_out = np.empty(n)
    ambiguous = np.zeros(n, dtype=bool)
    repeated = pd.Series(ids).duplicated(keep=False).to_numpy()
    for base in range(0, n, batch_size):
        distance, fraction = geometry(points[base : base + batch_size])
        stop = base + len(distance)
        chosen = selected[base:stop]
        rows = np.arange(len(distance))
        chosen_distance = distance[rows, chosen]
        distances_out[base:stop] = chosen_distance
        fractions_out[base:stop] = fraction[rows, chosen]
        different_arc = ids[None, :] != ids[chosen, None]
        alternative = np.min(np.where(different_arc, distance, np.inf), axis=1)
        ambiguous[base:stop] = alternative <= chosen_distance + ambiguity_margin_m

    flags = [[] for _ in range(n)]

    def flag(mask, name):
        for index in np.flatnonzero(mask):
            flags[index].append(name)

    flag(ambiguous, "ambiguous_geometry")
    flag(repeated[selected], "repeated_arc_occurrence")
    flag(distances_out > max_distance_m, "too_far")
    flag(lengths[selected] == 0, "zero_length_segment")
    dt = np.diff(timestamps)
    boundary = np.zeros(n, dtype=bool)
    invalid = np.zeros(n, dtype=bool)
    for step in np.flatnonzero(dt > gap_threshold_s):
        boundary[step : step + 2] = True
    for step in np.flatnonzero(dt <= 0):
        invalid[step : step + 2] = True
    flag(boundary, "gap_boundary")
    flag(invalid, "invalid_timestamp")
    sections = reference["section_id"].to_numpy()[selected]
    discontinuity = np.zeros(n, dtype=bool)
    for step in np.flatnonzero(np.diff(sections) != 0):
        discontinuity[step : step + 2] = True
    # Also exclude any chosen segment immediately bordering a reference break.
    route_breaks = np.flatnonzero(reference["discontinuity_before"].to_numpy())
    if len(route_breaks):
        discontinuity |= np.isin(selected, np.r_[route_breaks, route_breaks - 1])
    flag(discontinuity, "route_discontinuity")
    along = progress[selected] + fractions_out * lengths[selected]
    implausible = np.zeros(n, dtype=bool)
    bad = (
        (dt > 0)
        & (dt <= gap_threshold_s)
        & ((np.diff(along) > max_progress_speed_mps * dt + 25) | (np.diff(along) < -25))
    )
    for step in np.flatnonzero(bad):
        implausible[step : step + 2] = True
    flag(implausible, "implausible_progress")
    return pd.DataFrame(
        {
            "trip_id": trip["trip_id"],
            "observation_id": obs["observation_id"].to_numpy(),
            "reference_arc_id": ids[selected],
            "alignment_status": [
                "|".join(reasons) if reasons else "aligned" for reasons in flags
            ],
            "alignment_distance_m": distances_out,
            "route_position": selected,
            "reference_section_id": sections,
            "label_source": "provisional_ordered_reference_alignment",
        }
    )
