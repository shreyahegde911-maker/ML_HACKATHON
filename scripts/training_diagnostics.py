"""Plot measured confusion matrices and available boosting loss histories."""

import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
import joblib, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = ROOT / "results/final"
    metrics = json.loads((out / "metrics.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, row in zip(axes.flat, metrics):
        matrix = np.array(row["candidate_confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted candidate class",
            ylabel="Reference candidate class",
            title=row["model"].replace("_", " "),
        )
    fig.suptitle("Actual student-development candidate confusion matrices")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    record = []
    for ax, name in zip(
        axes, ["hist_gradient_boosting", "hist_gradient_boosting_tuned"]
    ):
        model = joblib.load(ROOT / f"artifacts/final/{name}.joblib")["model"]
        if model.scoring != "loss" or not len(model.train_score_):
            raise ValueError("No actual loss history available")
        ax.plot(-model.train_score_, label="Training loss")
        if len(model.validation_score_):
            ax.plot(
                -model.validation_score_, label="Internal early-stop validation loss"
            )
        ax.set(
            xlabel="Boosting iteration",
            ylabel="Binary log loss",
            title=name.replace("_", " "),
        )
        ax.legend(fontsize=8)
        record.append(
            {
                "model": name,
                "iterations": int(model.n_iter_),
                "training_loss": (-model.train_score_).tolist(),
                "internal_validation_loss": (-model.validation_score_).tolist(),
                "validation_scope": "training-only internal early-stopping split, not student-development",
            }
        )
    fig.suptitle("Measured boosting histories; no synthetic epoch curves")
    fig.tight_layout()
    fig.savefig(out / "boosting_loss_histories.png", dpi=150)
    plt.close(fig)
    (out / "boosting_loss_histories.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("Saved actual confusion matrices and boosting-iteration loss histories.")


if __name__ == "__main__":
    main()
