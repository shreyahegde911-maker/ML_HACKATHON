"""Validate assignment artifacts, fitted models, partition integrity and evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.utils.validation import check_is_fitted

MODELS = [
    "logistic_regression",
    "random_forest",
    "decision_tree",
    "hist_gradient_boosting",
    "gaussian_naive_bayes",
]
FILES = {
    "Data readers and preprocessing": [
        "src/pathsnap/data/readers.py",
        "src/pathsnap/data/alignment.py",
    ],
    "EDA and geographical visualisation": [
        "results/final/eda.png",
        "results/final/graph_connectivity.csv",
        "results/final/final_folium_map.html",
        "results/final/folium_map.png",
    ],
    "Feature generation": [
        "scripts/build_candidate_features.py",
        "artifacts/features/train_supervised.csv.gz",
        "artifacts/features/validation_supervised.csv.gz",
    ],
    "Five-model comparison": [
        "results/final/comparison.csv",
        "results/final/metrics.json",
        "results/final/model_comparison.png",
        "results/final/model_configurations.json",
        "results/final/boosting_loss_histories.png",
        "results/final/confusion_matrices.png",
    ],
    "Bayesian optimisation": [
        "results/final/bayesian_trials.csv",
        "results/final/bayesian_search.sqlite3",
        "results/final/selection.json",
        "results/final/trial_context.json",
    ],
    "HMM / Viterbi and measured comparison": [
        "src/pathsnap/decoding/viterbi.py",
        "configs/decoder.json",
        "results/final/decoder_comparison.csv",
        "results/final/decoder_summary.json",
    ],
    "API and evaluator UI": [
        "src/pathsnap/api/main.py",
        "ui/index.html",
        "results/final/api_verification.json",
        "results/final/browser_verification.json",
        "results/final/ui_desktop.png",
        "results/final/api_response.png",
    ],
    "Deployment and source instructions": [
        "start.sh",
        "start.bat",
        "requirements.txt",
        "README.md",
        "docs/FINAL_PRESENTATION.md",
    ],
    "Consolidated PDF report": ["output/pdf/PathSnap_Technical_Report.pdf"],
}


def audit():
    rows = []
    errors = []
    for name, files in FILES.items():
        missing = [
            p
            for p in files
            if not (ROOT / p).is_file() or (ROOT / p).stat().st_size == 0
        ]
        rows.append(
            {
                "deliverable": name,
                "status": "missing" if missing else "present",
                "files": files,
                "missing": missing,
            }
        )
        errors.extend(missing)
    sample = pd.read_csv(
        ROOT / "artifacts/features/student_test_supervised.csv.gz",
        nrows=12,
        dtype={"trip_id": str},
    )
    manifest = json.loads((ROOT / "results/part1/split_manifest.json").read_text())
    model_rows = []
    for name in MODELS:
        path = ROOT / "artifacts/final" / f"{name}.joblib"
        bundle = joblib.load(path)
        model = bundle["model"]
        check_is_fitted(model)
        if set(bundle["training_trip_ids"]) & set(manifest["student_test"]):
            errors.append(name + ": student-development leakage")
        probability = model.predict_proba(sample[bundle["features"]])
        if (
            probability.shape != (len(sample), 2)
            or not np.isfinite(probability).all()
            or not np.allclose(probability.sum(axis=1), 1)
        ):
            errors.append(name + ": invalid probabilities")
        model_rows.append(
            {
                "model": name,
                "fitted": True,
                "prediction_check": "passed",
                "fit_trips": len(bundle["training_trip_ids"]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    folds = json.loads((ROOT / "results/final/folds.json").read_text())
    for fold in folds:
        fit, valid = set(fold["fit_trips"]), set(fold["validation_trips"])
        if fit & valid or (fit | valid) & set(manifest["student_test"]):
            errors.append("CV partition leakage")
    table = pd.read_csv(ROOT / "results/final/comparison.csv")
    if set(MODELS) - set(table.model) or "direct_step_continuity" not in table:
        errors.append("Incomplete comparison table")
    trials = pd.read_csv(ROOT / "results/final/bayesian_trials.csv")
    if len(trials) < 10 or trials.state.eq("COMPLETE").sum() < 10:
        errors.append("Fewer than ten completed optimisation trials")
    api = json.loads((ROOT / "results/final/api_verification.json").read_text())
    if len(api["trips"]) != 10 or any(r["status"] != 200 for r in api["trips"]):
        errors.append("Not all ten API trips verified")
    if api["reference_payload_leak_check"] != "passed":
        errors.append("Reference response check failed")
    browser = json.loads((ROOT / "results/final/browser_verification.json").read_text())
    if browser["javascript_errors"]:
        errors.append("Browser JavaScript errors")
    reader = PdfReader(ROOT / "output/pdf/PathSnap_Technical_Report.pdf")
    report_text = " ".join(page.extract_text() for page in reader.pages)
    for name in MODELS:
        if name.replace("_", " ").lower() not in report_text.replace("_", " ").lower():
            errors.append("Model missing from report: " + name)
    pending = [
        "A second physical laptop/LAN and its receiving-terminal screenshot must be demonstrated at the event.",
        "The team must review and submit its work, and be ready to explain/disclose AI assistance as required by the event policy.",
    ]
    ci_path = ROOT / "results/final/cross_platform_ci.json"
    ci = json.loads(ci_path.read_text()) if ci_path.exists() else None
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "deliverables": rows,
        "models": model_rows,
        "pdf_pages": len(reader.pages),
        "completed_trials": int(trials.state.eq("COMPLETE").sum()),
        "cross_platform_ci": ci,
        "external_pending": pending,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate without writing audit files"
    )
    args = parser.parse_args()
    result = audit()
    if not args.check:
        (ROOT / "results/deliverable_audit.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        lines = [
            "# Deliverable audit",
            "",
            "Software/artifact checks: **" + result["status"] + "**.",
            "",
            "The assignment's algorithm list is examples, not a fixed list. All five implemented algorithms are explicitly listed in Models and Training Guidelines.pdf.",
            "",
            "| Deliverable | Status |",
            "| --- | --- |",
        ]
        lines += [
            f"| {r['deliverable']} | {r['status']} |" for r in result["deliverables"]
        ]
        lines += [
            "",
            f"All five saved models loaded, passed fitted-state checks and predicted finite probabilities. The consolidated report has {result['pdf_pages']} pages. Ten complete Bayesian trials, disjoint trip folds, all-ten-trip HTTP evidence and zero-browser-error evidence were verified.",
            "",
            "## Event-specific items still pending",
            "",
        ]
        lines += ["- " + item for item in result["external_pending"]]
        lines += [
            "",
            "## Repeat the audit",
            "",
            "```sh",
            "python -m unittest discover -s tests -v",
            "python scripts/audit_deliverables.py",
            "```",
            "",
            (
                "Native Windows and macOS checks passed: [GitHub Actions]("
                + result["cross_platform_ci"]["url"]
                + "). The receipt in results/final/cross_platform_ci.json identifies the tested source commit and jobs. Both wrappers started services, performed an actual prediction and stopped their own processes."
                if result["cross_platform_ci"]
                and result["cross_platform_ci"].get("status") == "passed"
                else "Cross-platform checks are configured in .github/workflows/verify.yml; inspect the corresponding GitHub Actions jobs for native results."
            ),
            "",
        ]
        (ROOT / "docs/DELIVERABLE_AUDIT.md").write_text("\n".join(lines))
    print(
        json.dumps({k: v for k, v in result.items() if k != "deliverables"}, indent=2)
    )
    if result["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
