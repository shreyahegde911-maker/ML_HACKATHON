#!/usr/bin/env python3
"""Inspect permitted partitions and optionally derive provisional labels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from pathsnap.data import (
    load_trip,
    load_reference_route,
    align_reference_route,
    create_split_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Student folder"))
    parser.add_argument(
        "--partition", choices=["train", "student_test", "all"], default="train"
    )
    parser.add_argument(
        "--trip", help="Trip ID; omit to inspect all trips in selected partitions"
    )
    parser.add_argument(
        "--align", action="store_true", help="Derive provisional labels separately"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/part1"))
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("data/derived/reference_labels")
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    for destination in (args.output_dir.resolve(), args.labels_dir.resolve()):
        if destination == data_root or data_root in destination.parents:
            parser.error("Outputs must be outside the raw data root")
    if args.trip and args.partition == "all":
        parser.error("Use one partition with --trip")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = create_split_manifest(data_root)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    partitions = (
        ["train", "student_test"] if args.partition == "all" else [args.partition]
    )
    inventory, quality, sources = [], {}, []
    start = time.perf_counter()
    for partition in partitions:
        trip_ids = (
            [str(args.trip).zfill(8)]
            if args.trip
            else sorted(
                p.name
                for p in (data_root / partition).iterdir()
                if p.is_dir() and p.name.isdigit()
            )
        )
        for trip_id in trip_ids:
            tick = time.perf_counter()
            raw = data_root / partition / trip_id
            file_info = {}
            for suffix in ("track", "nodes", "arcs", "route"):
                path = raw / f"{trip_id}.{suffix}"
                file_info[suffix] = {
                    "exists": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                }
                if path.is_file():
                    sources.append(
                        {
                            "path": str(path.relative_to(data_root)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
            trip = load_trip(data_root, partition, trip_id)
            reference = (
                load_reference_route(data_root, partition, trip_id, arcs=trip["arcs"])
                if file_info["route"]["exists"]
                else None
            )
            details = {
                **trip["validation"],
                "crs": {k: v for k, v in trip["crs"].items() if k != "wkt"},
                "files": file_info,
                "reference_discontinuity_positions": (
                    reference.loc[
                        reference["discontinuity_before"], "route_position"
                    ].tolist()
                    if reference is not None
                    else []
                ),
            }
            row = {
                "partition": partition,
                "trip_id": trip_id,
                "observations": len(trip["observations"]),
                "nodes": len(trip["nodes"]),
                "arcs": len(trip["arcs"]),
                "reference_entries": len(reference) if reference is not None else 0,
                "reference_discontinuities": len(
                    details["reference_discontinuity_positions"]
                ),
                "max_gap_s": details["max_gap_s"],
                "large_gaps": len(details["large_gap_observation_ids"]),
                "nonpositive_time_steps": len(
                    details["nonpositive_time_observation_ids"]
                ),
                "isolated_nodes": details["isolated_node_count"],
                "projected_crs": trip["crs"]["projected"],
            }
            for suffix, information in file_info.items():
                row[f"{suffix}_bytes"] = information["bytes"]
            if args.align and reference is not None:
                labels = align_reference_route(trip, reference)
                destination = args.labels_dir / partition
                destination.mkdir(parents=True, exist_ok=True)
                labels.to_csv(destination / f"{trip_id}.csv", index=False)
                row["provisional_aligned"] = int(
                    (labels["alignment_status"] == "aligned").sum()
                )
                row["provisional_flagged"] = len(labels) - row["provisional_aligned"]
                details["alignment_status_counts"] = {
                    str(k): int(v)
                    for k, v in labels["alignment_status"].value_counts().items()
                }
            row["elapsed_s"] = round(time.perf_counter() - tick, 4)
            inventory.append(row)
            quality[f"{partition}/{trip_id}"] = details
            print(
                f"{partition}/{trip_id}: {row['observations']} GPS, {row['reference_entries']} reference entries, "
                f"{row['reference_discontinuities']} reference breaks, "
                f"{row.get('provisional_aligned', 'N/A')} eligible provisional labels",
                flush=True,
            )
    frame = pd.DataFrame(inventory)
    frame.to_csv(args.output_dir / "inventory.csv", index=False)
    (args.output_dir / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    # Verify byte-for-byte invariance of every inspected source after processing.
    for source in sources:
        if (
            hashlib.sha256((data_root / source["path"]).read_bytes()).hexdigest()
            != source["sha256"]
        ):
            raise RuntimeError(f"Raw source changed: {source['path']}")
    (args.output_dir / "source_checksums.json").write_text(
        json.dumps(sources, indent=2) + "\n"
    )
    summary = {
        "trips": len(frame),
        "observations": int(frame["observations"].sum()),
        "reference_entries": int(frame["reference_entries"].sum()),
        "reference_discontinuities": int(frame["reference_discontinuities"].sum()),
        "nonpositive_time_steps": int(frame["nonpositive_time_steps"].sum()),
        "max_gap_s": float(frame["max_gap_s"].max()),
        "partition_trip_counts": {
            str(k): int(v) for k, v in frame["partition"].value_counts().items()
        },
        "raw_files_verified_unchanged": len(sources),
        "elapsed_s": round(time.perf_counter() - start, 3),
        "label_source": "provisional_ordered_reference_alignment; NOT official observation-level ground truth",
        "alignment_settings": {
            "max_distance_m": 75,
            "ambiguity_margin_m": 5,
            "gap_threshold_s": 30,
            "max_progress_speed_mps": 60,
            "distance_scale_m": 25,
            "progress_penalty": 0.05,
        },
        "command": ["python", "scripts/inspect_data.py", *sys.argv[1:]],
    }
    if args.align:
        summary["provisional_aligned"] = (
            int(frame["provisional_aligned"].fillna(0).sum())
            if "provisional_aligned" in frame
            else 0
        )
        summary["provisional_flagged"] = (
            int(frame["provisional_flagged"].fillna(0).sum())
            if "provisional_flagged" in frame
            else 0
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
