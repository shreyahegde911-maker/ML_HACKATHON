"""Tune decoder on permitted development trips and compare classifier vs HMM."""

import json, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import joblib, numpy as np, pandas as pd, folium
from pathsnap.data.readers import load_trip_directory
from pathsnap.api.service import reference_for_evaluation
from pathsnap.candidates import build_segments, generate_candidates
from pathsnap.features import build_feature_table
from pathsnap.models import predict_candidates
from pathsnap.evaluation import top_candidates
from pathsnap.decoding import decode, sequence_metrics


def main():
    out = ROOT / "results/final"
    model = joblib.load(ROOT / "artifacts/final/selected.joblib")["model"]
    root = ROOT / "Student folder/student_test"
    cache = {}
    rows = []
    search = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        trip = load_trip_directory(directory)
        c = generate_candidates(
            trip["observations"], build_segments(trip["nodes"], trip["arcs"])
        )
        f = build_feature_table(
            trip["trip_id"],
            trip["observations"],
            c,
            build_segments(trip["nodes"], trip["arcs"]),
        )
        p = predict_candidates(model, f)
        labels = reference_for_evaluation(trip, directory)
        cache[directory.name] = (trip, c, p, labels)
    for beta in [10.0, 30.0, 80.0]:
        scores = []
        for tid in list(cache)[:3]:
            trip, c, p, labels = cache[tid]
            d = decode(trip, c, p, beta_m=beta)
            metrics = sequence_metrics(trip, d["predictions"], labels)
            scores.append(metrics)
        score = sum(
            m["top1_accuracy"] * m["eligible_observations"] for m in scores
        ) / sum(m["eligible_observations"] for m in scores)
        search.append(
            {
                "beta_m": beta,
                "development_trips": list(cache)[:3],
                "top1_accuracy": score,
            }
        )
    beta = max(search, key=lambda r: r["top1_accuracy"])["beta_m"]
    settings = {"beta_m": beta, "gap_s": 30.0, "max_speed_mps": 60.0}
    (ROOT / "configs/decoder.json").write_text(json.dumps(settings, indent=2) + "\n")
    (out / "decoder_search.json").write_text(json.dumps(search, indent=2) + "\n")
    for tid, (trip, c, p, labels) in cache.items():
        tick = time.perf_counter()
        d = decode(trip, c, p, **settings)
        elapsed = time.perf_counter() - tick
        scored = c.copy()
        scored.insert(0, "trip_id", tid)
        point = top_candidates(scored, p)
        full = trip["observations"][["observation_id"]].copy()
        full.insert(0, "trip_id", tid)
        point = full.merge(point, on=["trip_id", "observation_id"], how="left")
        for name, pred in [("classifier", point), ("hmm", d["predictions"])]:
            metric = sequence_metrics(trip, pred, labels)
            metric.update(
                trip_id=tid,
                method=name,
                decoder_s=elapsed if name == "hmm" else 0.0,
                resets=len(d["breaks"]) if name == "hmm" else 0,
            )
            rows.append(metric)
        d["predictions"].to_csv(out / f"decoded_{tid}.csv", index=False)
        (out / f"route_{tid}.json").write_text(
            json.dumps({"route_runs": d["route_runs"], "breaks": d["breaks"]}, indent=2)
            + "\n"
        )
        if tid == list(cache)[0]:
            xy = trip["nodes"][["longitude", "latitude"]].to_numpy()
            edges = trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int)
            lines = xy[edges]
            obs = trip["observations"]
            m = folium.Map(
                [obs.latitude.median(), obs.longitude.median()], zoom_start=14
            )
            low = obs[["longitude", "latitude"]].min().to_numpy() - 0.002
            high = obs[["longitude", "latitude"]].max().to_numpy() + 0.002
            near = ((lines.max(axis=1) >= low) & (lines.min(axis=1) <= high)).all(
                axis=1
            )
            folium.GeoJson(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": lines[
                            np.flatnonzero(near)[:: max(1, int(near.sum()) // 8000)]
                        ].tolist(),
                    },
                },
                name="Supplied road network",
                show=False,
                style_function=lambda _: {"color": "#64748b", "weight": 1},
            ).add_to(m)
            for name, ids, color, width in [
                (
                    "Supplied development reference",
                    labels.reference_arc_id.unique(),
                    "#16a34a",
                    6,
                ),
                (
                    "Classifier",
                    point.predicted_arc_id.dropna().astype(int).unique(),
                    "#f97316",
                    3,
                ),
                (
                    "HMM Viterbi",
                    np.unique(np.concatenate(d["route_runs"])),
                    "#7c3aed",
                    3,
                ),
            ]:
                folium.GeoJson(
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "MultiLineString",
                            "coordinates": lines[np.asarray(ids, dtype=int)].tolist(),
                        },
                    },
                    name=name,
                    style_function=lambda _, c=color, w=width: {
                        "color": c,
                        "weight": w,
                    },
                ).add_to(m)
            gps = folium.FeatureGroup(name="Raw GPS")
            for r in obs.iloc[:: max(1, len(obs) // 1000)].itertuples():
                folium.CircleMarker(
                    [r.latitude, r.longitude], radius=2, color="#2563eb"
                ).add_to(gps)
            gps.add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            m.fit_bounds(
                [
                    [obs.latitude.min(), obs.longitude.min()],
                    [obs.latitude.max(), obs.longitude.max()],
                ]
            )
            m.get_root().html.add_child(
                folium.Element(
                    '<div style="position:fixed;bottom:30px;left:20px;z-index:9999;background:white;padding:12px;font:14px sans-serif;border:1px solid #aaa"><b>PathSnap - development trip 00000000</b><br><span style="color:#2563eb">Blue: raw GPS</span><br><span style="color:#16a34a">Green: supplied reference</span><br><span style="color:#f97316">Orange: classifier</span><br><span style="color:#7c3aed">Purple: HMM/Viterbi</span></div>'
                )
            )
            html = m.get_root().render()
            (out / "final_folium_map.html").write_text(
                "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
            )
        print(tid, rows[-2]["top1_accuracy"], rows[-1]["top1_accuracy"], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "decoder_comparison.csv", index=False)
    aggregate = []
    for method, g in table.groupby("method"):
        aggregate.append(
            {
                "method": method,
                "top1_accuracy": float(
                    np.average(g.top1_accuracy, weights=g.eligible_observations)
                ),
                "direct_step_continuity": float(
                    np.average(g.direct_step_continuity, weights=g.continuity_pairs)
                ),
                "arc_set_f1_mean": float(g.arc_set_f1.mean()),
                "eligible_observations": int(g.eligible_observations.sum()),
                "reset_count": int(g.resets.sum()),
            }
        )
    (out / "decoder_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = pd.DataFrame(aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, column in zip(axes, ["top1_accuracy", "direct_step_continuity"]):
        ax.bar(a.method, a[column], color=["#f97316", "#7c3aed"])
        ax.set_ylim(0, 1)
        ax.set_title(column.replace("_", " ").title())
    fig.suptitle("Classifier versus graph HMM on student-development trips")
    fig.tight_layout()
    fig.savefig(out / "decoder_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
