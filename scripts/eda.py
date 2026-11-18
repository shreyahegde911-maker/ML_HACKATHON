"""Recorded development-data EDA and graph-connectivity statistics."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    inventory = pd.read_csv(
        ROOT / "results/part1/inventory.csv", dtype={"trip_id": str}
    )
    rows = []
    for r in inventory.itertuples():
        arcs = pd.read_csv(
            ROOT / f"Student folder/{r.partition}/{r.trip_id}/{r.trip_id}.arcs",
            sep=r"\s+",
            header=None,
        ).to_numpy()
        graph = csr_matrix(
            (np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(r.nodes, r.nodes)
        )
        rows.append(
            {
                "partition": r.partition,
                "trip_id": r.trip_id,
                "weak_components": int(
                    connected_components(
                        graph, directed=True, connection="weak", return_labels=False
                    )
                ),
                "strong_components": int(
                    connected_components(
                        graph, directed=True, connection="strong", return_labels=False
                    )
                ),
                "isolated_nodes": r.isolated_nodes,
            }
        )
    out = ROOT / "results/final"
    pd.DataFrame(rows).to_csv(out / "graph_connectivity.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for name, g in inventory.groupby("partition"):
        axes[0, 0].hist(g.observations, bins=15, alpha=0.65, label=name)
        axes[0, 1].scatter(g.nodes, g.arcs, label=name, alpha=0.6)
    axes[0, 0].set(
        title="GPS observations per trip", xlabel="Observations", ylabel="Trips"
    )
    axes[0, 0].legend()
    axes[0, 1].set(
        title="Road graph size",
        xlabel="Nodes",
        ylabel="Arcs",
        xscale="log",
        yscale="log",
    )
    axes[1, 0].hist(np.log10(np.maximum(inventory.max_gap_s, 1)), bins=20)
    axes[1, 0].set(
        title="Maximum timestamp gap per trip", xlabel="log10(seconds)", ylabel="Trips"
    )
    graph = pd.DataFrame(rows)
    axes[1, 1].hist(graph.weak_components, bins=15)
    axes[1, 1].set(
        title="Weakly connected components", xlabel="Components", ylabel="Trips"
    )
    fig.suptitle("EDA: 80 training + 10 student-development trips")
    fig.tight_layout()
    fig.savefig(out / "eda.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
