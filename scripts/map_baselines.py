#!/usr/bin/env python3
"""Folium comparison of actual classifier-only predictions on a held-out trip."""
import argparse
import json
from pathlib import Path
import sys
import folium
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import load_trip, load_reference_route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Student folder")
    parser.add_argument("--partition", choices=["validation", "student_test"], default="validation")
    parser.add_argument("--trip-id")
    parser.add_argument("--results", type=Path, default=ROOT / "results/baselines")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baselines/comparison_map.html")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "results/part1/split_manifest.json").read_text())
    tid = args.trip_id or manifest[args.partition][0]
    if tid not in manifest[args.partition]:
        raise ValueError("Map trip must belong to selected development partition")
    source = manifest["source_partitions"][args.partition]
    trip = load_trip(args.data_root, source, tid)
    obs, nodes, arcs = trip["observations"], trip["nodes"], trip["arcs"]
    xy = nodes[["longitude", "latitude"]].to_numpy()
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    lines = xy[endpoints]
    map_ = folium.Map(location=[obs.latitude.median(), obs.longitude.median()], zoom_start=14)

    def layer(ids, name, color, weight=3, show=True):
        ids = np.asarray(ids, dtype=int)
        collection = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": lines[ids].tolist()}}]}
        folium.GeoJson(collection, name=name, show=show,
                       style_function=lambda _, c=color, w=weight: {"color": c, "weight": w, "opacity": .8}).add_to(map_)

    low = obs[["longitude", "latitude"]].min().to_numpy() - .002
    high = obs[["longitude", "latitude"]].max().to_numpy() + .002
    in_extent = ((lines.max(axis=1) >= low) & (lines.min(axis=1) <= high)).all(axis=1)
    layer(np.flatnonzero(in_extent), "Relevant road network", "#9ca3af", 1, False)
    reference = load_reference_route(args.data_root, source, tid, arcs=arcs)
    layer(reference.arc_id.unique(), "Supplied reference route (development only)", "#16a34a", 6)
    gps = folium.FeatureGroup(name="Raw GPS observations")
    stride = max(1, len(obs) // 1500)
    for row in obs.iloc[::stride].itertuples():
        folium.CircleMarker([row.latitude, row.longitude], radius=2, color="#2563eb", fill=True,
                            popup=f"Observation {row.observation_id}; timestamp {row.timestamp:g}").add_to(gps)
    gps.add_to(map_)
    for model, color in [("logistic_regression", "#ea580c"), ("random_forest", "#9333ea")]:
        predictions = pd.read_csv(args.results / f"{model}_{args.partition}_predictions.csv.gz", dtype={"trip_id": str})
        selected = predictions.loc[predictions.trip_id.eq(tid)]
        layer(selected.predicted_arc_id.dropna().astype(int).unique(), f"{model}: pointwise segment predictions", color, 3)
    folium.LayerControl(collapsed=False).add_to(map_)
    map_.fit_bounds([[obs.latitude.min(), obs.longitude.min()], [obs.latitude.max(), obs.longitude.max()]])
    legend = f'''<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;padding:12px;border:1px solid #aaa;font:13px sans-serif;max-width:330px">
    <b>PathSnap: {args.partition} trip {tid}</b><br>
    <span style="color:#2563eb">Blue: raw GPS</span><br><span style="color:#16a34a">Green: supplied reference</span><br>
    <span style="color:#ea580c">Orange: Logistic Regression</span><br><span style="color:#9333ea">Purple: Random Forest</span><br>
    Classifier-only segment matches; no HMM reconstruction.<br>Metrics use provisional derived reference labels.</div>'''
    map_.get_root().html.add_child(folium.Element(legend))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_.save(str(args.output))
    print(f"Saved {args.output}: {len(obs)} GPS observations, development trip {tid}")


if __name__ == "__main__":
    main()
