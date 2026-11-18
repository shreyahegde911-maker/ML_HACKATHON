"""Classifier-emission HMM with directed road distances and Viterbi decoding."""

from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class RoadDistances:
    def __init__(self, trip, limit=1900.0, cache_size=64):
        self.edges = trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int)
        self.xy = trip["nodes"][["x_m", "y_m"]].to_numpy()
        self.vectors = self.xy[self.edges[:, 1]] - self.xy[self.edges[:, 0]]
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        frame = pd.DataFrame(
            {
                "u": self.edges[:, 0],
                "v": self.edges[:, 1],
                "w": np.maximum(self.lengths, 1e-6),
                "arc": np.arange(len(self.edges)),
            }
        )
        unique = frame.sort_values("w", kind="stable").drop_duplicates(["u", "v"])
        self.graph = csr_matrix(
            (unique.w, (unique.u, unique.v)), shape=(len(self.xy), len(self.xy))
        )
        self.arc_lookup = {
            (int(r.u), int(r.v)): int(r.arc) for r in unique.itertuples()
        }
        self.cache = OrderedDict()
        self.limit = limit
        self.cache_size = cache_size

    def source(self, node):
        node = int(node)
        if node not in self.cache:
            self.cache[node] = dijkstra(
                self.graph,
                directed=True,
                indices=node,
                return_predecessors=True,
                limit=self.limit,
            )
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(node)
        return self.cache[node]

    def path(self, source, target):
        source, target = int(source), int(target)
        if source == target:
            return []
        distances, predecessors = self.source(source)
        if not np.isfinite(distances[target]):
            return None
        nodes = [target]
        while nodes[-1] != source:
            node = int(predecessors[nodes[-1]])
            if node < 0 or len(nodes) > len(self.xy):
                return None
            nodes.append(node)
        nodes.reverse()
        return [self.arc_lookup[(u, v)] for u, v in zip(nodes[:-1], nodes[1:])]

    def fractions(self, point, ids):
        v = self.vectors[ids]
        den = (v * v).sum(axis=1)
        return np.clip(
            np.divide(
                ((point - self.xy[self.edges[ids, 0]]) * v).sum(axis=1),
                den,
                out=np.zeros(len(ids)),
                where=den > 0,
            ),
            0,
            1,
        )

    def travel(self, previous, current, previous_fraction, current_fraction):
        p = np.asarray(previous, dtype=int)
        c = np.asarray(current, dtype=int)
        result = np.empty((len(p), len(c)))
        for i, arc in enumerate(p):
            distance = self.source(self.edges[arc, 1])[0][self.edges[c, 0]]
            result[i] = (
                (1 - previous_fraction[i]) * self.lengths[arc]
                + distance
                + current_fraction * self.lengths[c]
            )
            same = c == arc
            advance = (current_fraction - previous_fraction[i]) * self.lengths[arc]
            result[i, same & (advance >= -25)] = np.maximum(
                0, advance[same & (advance >= -25)]
            )
        return result


def decode(
    trip, candidates, probabilities, *, beta_m=30.0, gap_s=30.0, max_speed_mps=60.0
):
    if beta_m <= 0 or gap_s <= 0 or max_speed_mps <= 0:
        raise ValueError("Decoder parameters must be positive")
    if len(candidates) != len(probabilities):
        raise ValueError("Probability/candidate length mismatch")
    roads = RoadDistances(trip, limit=max_speed_mps * gap_s + 100)
    obs = trip["observations"]
    points = obs[["x_m", "y_m"]].to_numpy()
    times = obs.timestamp.to_numpy()
    grouped = {
        int(k): g.index.to_numpy()
        for k, g in candidates.reset_index(drop=True).groupby(
            "observation_id", sort=False
        )
    }
    arcs = candidates.arc_id.to_numpy(dtype=int)
    probabilities = np.clip(np.asarray(probabilities), 1e-12, 1.0)
    predicted = np.full(len(obs), -1, dtype=int)
    breaks = []
    chain = []
    backs = []
    scores = None
    previous_ids = None
    previous_fraction = None

    def flush():
        if not chain:
            return
        state = int(np.argmax(scores))
        for i in range(len(chain) - 1, -1, -1):
            observation, indices = chain[i]
            predicted[observation] = arcs[indices[state]]
            if i:
                state = int(backs[i][state])

    for i in range(len(obs)):
        index = grouped.get(int(obs.observation_id.iloc[i]), np.array([], dtype=int))
        if not len(index):
            flush()
            chain = []
            backs = []
            scores = None
            breaks.append(
                {
                    "observation_id": int(obs.observation_id.iloc[i]),
                    "reason": "no_candidates",
                }
            )
            continue
        ids = arcs[index]
        fraction = roads.fractions(points[i], ids)
        emission = np.log(probabilities[index] / probabilities[index].sum())
        reset = None
        if scores is not None:
            dt = times[i] - times[i - 1]
            if dt <= 0 or dt > gap_s:
                reset = "timestamp_gap"
            else:
                travel = roads.travel(previous_ids, ids, previous_fraction, fraction)
                plausible = np.isfinite(travel) & (travel <= max_speed_mps * dt + 25)
                transitions = (
                    -np.abs(travel - np.linalg.norm(points[i] - points[i - 1])) / beta_m
                )
                transitions[~plausible] = -np.inf
                total = scores[:, None] + transitions
                parents = np.argmax(total, axis=0)
                new = emission + total[parents, np.arange(len(ids))]
                if not np.isfinite(new).any():
                    reset = "disconnected_or_implausible_transition"
        if scores is None or reset:
            if reset:
                flush()
                breaks.append(
                    {"observation_id": int(obs.observation_id.iloc[i]), "reason": reset}
                )
            chain = []
            backs = []
            scores = emission
            parents = np.zeros(len(ids), dtype=int)
        else:
            scores = new
        scores -= np.max(scores)
        chain.append((i, index))
        backs.append(parents)
        previous_ids, previous_fraction = ids, fraction
    flush()
    # Expand graph connectors; never draw a bridge over a reset or unreachable gap.
    boundary = {b["observation_id"] for b in breaks}
    runs = []
    run = []
    last = None
    for i, arc in enumerate(predicted):
        if arc < 0 or int(obs.observation_id.iloc[i]) in boundary:
            if run:
                runs.append(run)
            run = []
            last = None
        if arc < 0:
            continue
        if last is not None and arc != last:
            connector = roads.path(roads.edges[last, 1], roads.edges[arc, 0])
            if connector is None:
                if run:
                    runs.append(run)
                run = []
            else:
                run.extend(connector)
        if not run or run[-1] != arc:
            run.append(int(arc))
        last = arc
    if run:
        runs.append(run)
    frame = pd.DataFrame(
        {
            "trip_id": trip["trip_id"],
            "observation_id": obs.observation_id,
            "predicted_arc_id": pd.array(
                np.where(predicted < 0, None, predicted), dtype="Int64"
            ),
        }
    )
    return {
        "predictions": frame,
        "route_runs": runs,
        "breaks": breaks,
        "settings": {"beta_m": beta_m, "gap_s": gap_s, "max_speed_mps": max_speed_mps},
    }


def sequence_metrics(trip, predictions, reference=None):
    ids = predictions.predicted_arc_id.to_numpy(dtype=float, na_value=np.nan)
    edges = trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    valid = np.isfinite(ids[:-1]) & np.isfinite(ids[1:])
    valid &= (np.diff(trip["observations"].timestamp) > 0) & (
        np.diff(trip["observations"].timestamp) <= 30
    )
    a, b = ids[:-1][valid].astype(int), ids[1:][valid].astype(int)
    continuity = (
        float(((a == b) | (edges[a, 1] == edges[b, 0])).mean()) if len(a) else None
    )
    result = {"direct_step_continuity": continuity, "continuity_pairs": len(a)}
    if reference is not None:
        merged = reference.merge(
            predictions,
            on=["trip_id", "observation_id"],
            how="left",
            validate="one_to_one",
        )
        eligible = merged.alignment_status.eq("aligned")
        result.update(
            eligible_observations=int(eligible.sum()),
            top1_accuracy=(
                float(
                    merged.loc[eligible, "predicted_arc_id"]
                    .eq(merged.loc[eligible, "reference_arc_id"])
                    .fillna(False)
                    .mean()
                )
                if eligible.any()
                else None
            ),
        )
        # Set overlap complements pointwise accuracy; it is not an order metric.
        true = set(merged.loc[eligible, "reference_arc_id"].dropna().astype(int))
        guess = set(merged.loc[eligible, "predicted_arc_id"].dropna().astype(int))
        intersection = len(true & guess)
        result["arc_set_f1"] = (
            2 * intersection / (len(true) + len(guess)) if true or guess else None
        )
    return result
