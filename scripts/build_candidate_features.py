"""Build network-only candidates and shared LR/RF feature table.

Run from repo root: python scripts/build_candidate_features.py --partition train
The Part 1 JSON split manifest is used by default when present. A raw-file
adapter is retained only for checkouts where Part 1's loader is unavailable.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.candidates import CandidateConfig, build_segments, generate_candidates
from pathsnap.features import build_feature_table, attach_reference_labels, reference_candidate_coverage

try:
    from pathsnap.data import load_trip as part1_load_trip
except ImportError:
    try:
        from pathsnap.data.loader import load_trip as part1_load_trip
    except ImportError:
        part1_load_trip = None


def read_trip(trip_dir, trip_id):
    """Temporary raw adapter honoring the documented Part 1 dataframe schema."""
    p = Path(trip_dir)
    track = pd.read_csv(p / f"{trip_id}.track", sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
    track.insert(0, "observation_id", np.arange(len(track), dtype="int64"))
    nodes = pd.read_csv(p / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["longitude", "latitude"])
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype="int64"))
    arcs = pd.read_csv(p / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype="int64"))
    # Local equirectangular projection around the network centroid; coordinates
    # are suitable in metres for these compact single-trip German road graphs.
    lon0, lat0 = nodes.longitude.mean(), nodes.latitude.mean()
    xscale = 111320.0 * np.cos(np.deg2rad(lat0))
    yscale = 110574.0
    for frame in (track, nodes):
        frame["x_m"] = (frame.longitude - lon0) * xscale
        frame["y_m"] = (frame.latitude - lat0) * yscale
    return track, nodes, arcs


def trip_token(value):
    value = str(value)
    return value.zfill(8) if value.isdigit() else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train", choices=["train", "validation", "student_test"])
    ap.add_argument("--manifest", type=Path, default=ROOT / "results" / "part1" / "split_manifest.json")
    ap.add_argument("--references", type=Path, help="Part 1 alignment CSV; never used for candidate selection")
    ap.add_argument("--reliable-status", nargs="+", default=["aligned", "reliable", "ok"],
                    help="alignment_status values eligible for labels (case-insensitive)")
    ap.add_argument("--trip-id")
    ap.add_argument("--data-root", type=Path, default=ROOT / "Student folder")
    ap.add_argument("--output", type=Path, default=ROOT / "cache" / "candidate_features")
    ap.add_argument("--radius-m", type=float, default=100.0)
    ap.add_argument("--expanded-radius-m", type=float, default=300.0)
    ap.add_argument("--max-candidates", type=int, default=8)
    args = ap.parse_args()

    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.partition not in manifest or "source_partitions" not in manifest:
            raise ValueError("split manifest must contain the requested trip list and source_partitions")
        trip_ids = [trip_token(x) for x in manifest[args.partition]]
        source_partition = manifest["source_partitions"][args.partition]
    else:
        if args.partition == "validation":
            raise FileNotFoundError("validation generation requires Part 1's split manifest")
        source_partition = args.partition
        trip_root = args.data_root / source_partition
        # Partition directory itself enforces trip-level separation; no row split.
        trip_ids = sorted(p.name for p in trip_root.iterdir() if p.is_dir())
    if args.trip_id:
        chosen = trip_token(args.trip_id)
        if chosen not in trip_ids:
            raise ValueError(f"trip {chosen} is not assigned to manifest partition {args.partition}")
        trip_ids = [chosen]
    config = CandidateConfig(args.radius_m, args.max_candidates, args.expanded_radius_m)
    features, summaries = [], []
    start = time.perf_counter()
    for trip_id in trip_ids:
        t0 = time.perf_counter()
        if part1_load_trip is not None:
            trip = part1_load_trip(args.data_root, source_partition, trip_id)
            observations, nodes, arcs = trip["observations"], trip["nodes"], trip["arcs"]
        else:
            observations, nodes, arcs = read_trip(args.data_root / source_partition / trip_id, trip_id)
        segments = build_segments(nodes, arcs)
        candidates = generate_candidates(observations, segments, config)
        ft = build_feature_table(trip_id, observations, candidates, segments)
        features.append(ft)
        cand_obs = set(candidates.observation_id)
        summaries.append({
            "trip_id": trip_id, "partition": args.partition, "source_partition": source_partition,
            "observation_count": len(observations), "node_count": len(nodes), "arc_count": len(arcs),
            "segment_count": len(segments), "candidate_rows": len(candidates),
            "observations_with_candidates": len(cand_obs),
            "observations_without_candidates": len(observations) - len(cand_obs),
            "mean_candidates_per_observation": len(candidates) / max(len(observations), 1),
            "expanded_search_observations": int(candidates.loc[candidates.search_expanded, "observation_id"].nunique()) if len(candidates) else 0,
            "reference_label_observations": 0, "reference_candidate_misses": 0,
            "candidate_miss_rate": np.nan, "runtime_s": time.perf_counter() - t0,
        })
        print(f"[{len(summaries)}/{len(trip_ids)}] {trip_id}: {len(observations)} observations, {len(candidates)} candidates", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_table = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    if args.references and len(feature_table):
        references = pd.read_csv(args.references, dtype={"trip_id": str})
        references["trip_id"] = references.trip_id.map(trip_token)
        supervised = attach_reference_labels(feature_table, references, args.reliable_status)
        coverage = reference_candidate_coverage(feature_table, references, args.reliable_status)
        for trip_id, group in coverage.groupby("trip_id"):
            m = summary.trip_id.astype(str) == str(trip_id)
            reliable_count = int(group.label_reliable.sum())
            misses = int(group.reference_arc_missing.sum())
            summary.loc[m, "reference_label_observations"] = reliable_count
            summary.loc[m, "reference_candidate_misses"] = misses
            summary.loc[m, "candidate_miss_rate"] = misses / reliable_count if reliable_count else np.nan
        coverage.to_csv(args.output / f"{args.partition}_reference_coverage.csv", index=False)
        supervised.to_csv(args.output / f"{args.partition}_supervised.csv", index=False)
    feature_table.to_csv(args.output / f"{args.partition}_features.csv", index=False)
    summary.to_csv(args.output / f"{args.partition}_coverage.csv", index=False)
    print(f"trips={len(trip_ids)} observations={summary.observation_count.sum() if len(summary) else 0} "
          f"candidate_rows={len(feature_table)} runtime_s={time.perf_counter()-start:.2f}")
    print(f"features: {args.output / (args.partition + '_features.csv')}")
    print(f"coverage: {args.output / (args.partition + '_coverage.csv')}")
    if summary.reference_label_observations.sum() == 0:
        message = "Part 1 reference alignments not present" if not args.references else "no reliable alignment labels in supplied file"
        print(f"candidate miss rate: unavailable ({message})")


if __name__ == "__main__":
    main()
