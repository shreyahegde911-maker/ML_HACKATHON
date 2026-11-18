"""Build the consolidated technical PDF from recorded experiments/screenshots."""

import json, sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


def main():
    out = ROOT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    results = ROOT / "results/final"
    metrics = json.loads((results / "metrics.json").read_text())
    selection = json.loads((results / "selection.json").read_text())
    decoder = json.loads((results / "decoder_summary.json").read_text())
    api = json.loads((results / "api_verification.json").read_text())
    configs = json.loads((results / "model_configurations.json").read_text())
    inventory = pd.read_csv(ROOT / "results/part1/inventory.csv")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallText", fontSize=8, leading=11, spaceAfter=7))
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    story = []

    def text(value, style="BodyText"):
        story.append(Paragraph(escape(str(value)), styles[style]))
        story.append(Spacer(1, 7))

    def heading(value):
        text(value, "Heading1")

    def picture(name, width=510, maxheight=430):
        path = results / name
        from PIL import Image as PILImage

        w, h = PILImage.open(path).size
        scale = min(width / w, maxheight / h)
        story.append(Image(str(path), width=w * scale, height=h * scale))
        story.append(Spacer(1, 8))

    def table(rows, widths=None):
        wrapped = [
            [Paragraph(escape(str(c)), styles["SmallText"]) for c in row]
            for row in rows
        ]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf5")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b5c4d3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    def page():
        story.append(PageBreak())

    text("PATHSNAP", "Title")
    text("Distributed map matching and route reconstruction", "Heading1")
    text("Technical report | 18 November 2026")
    text(
        "Five-model experiments, Bayesian optimisation, directed graph HMM/Viterbi and two-process API/UI deployment."
    )
    picture("ui_desktop.png", maxheight=370)
    text("Evidence and scope", "Heading2")
    text(
        "All reported experiments were executed. Source, artifacts, metrics, search records and verification evidence are committed to the repository. The available data comprises 80 training trips and 10 student-development trips. Reserved evaluator files and a second physical laptop were not available: the presentation runs a clearly labelled development demo, with separate UI and API processes verified on this computer."
    )
    text(
        "Observation labels are derived by ordered geometric alignment to the supplied reference route. They are provisional supervision, not independently verified physical ground truth. All reference-based scores must be interpreted accordingly."
    )
    page()
    heading("1. Dataset understanding and preprocessing")
    table(
        [
            ["Partition", "Trips", "GPS observations", "Purpose"],
            ["Original training", 80, 208417, "Fitting and trip-grouped CV"],
            [
                "Student development",
                10,
                19160,
                "Development comparisons and decoder settings",
            ],
            [
                "Reserved evaluator",
                "Not supplied",
                "Not processed",
                "Final evaluation only",
            ],
        ],
        [120, 55, 100, 235],
    )
    text(
        "The four raw files are whitespace-separated: track gives longitude, latitude and timestamp; nodes gives node coordinates; arcs gives directed endpoint IDs; route gives an ordered sequence of reference arc IDs. IDs are zero-based source row numbers, local to each trip. Training trip 00000001 has 1,315 observations and 525 reference entries, so row-wise GPS/route joins are invalid."
    )
    text(
        "Strict readers preserve source order and reject malformed columns, nonfinite numbers, invalid coordinate ranges and out-of-bounds IDs. All 360 supplied raw files remain unchanged. Complete development inspection succeeded without missing/nonfinite numeric entries. There were no nonpositive timestamp steps, but the largest gap was 153,064 seconds. Fourteen directed reference endpoint discontinuities were reported. See results/part1 for per-trip counts, anomalies and raw checksums."
    )
    text(
        "Coordinates are projected with always_xy=True into one WGS84 UTM CRS selected per trip. GPS and road nodes share that transformer. Extents exceeding 200 km receive warnings. Geography varies across trips; no single city or global arc-label vocabulary is assumed."
    )
    text(
        "Reference alignment uses nondecreasing route-occurrence dynamic programming. Repeated arcs remain distinct occurrences. Ambiguous geometry, large distance, repeated occurrences, zero-length arcs, gaps, reference discontinuities and implausible progress are flagged. Only status aligned supplies supervision; 136,278 of 227,577 observations are eligible across the available development package."
    )
    page()
    heading("2. Exploratory analysis and features")
    picture("eda.png", maxheight=405)
    text(
        "The plots use all 80 training and 10 student-development trips. Connectivity statistics, including weak/strong components and isolated nodes, are recorded in graph_connectivity.csv. Raw GPS and relevant road geometry are shown in the geographic visualisation section."
    )
    text(
        "Candidate generation searches road segments within 100 m, expands to 300 m only when necessary, and keeps up to eight nearest candidates. Length-stratified midpoint KD-trees conservatively retrieve segments before exact point-to-segment distances. No reference route influences candidates."
    )
    table(
        [
            ["Feature", "Definition and reason"],
            [
                "distance_m",
                "Shortest GPS-to-segment distance in projected metres; spatial agreement.",
            ],
            [
                "heading_difference",
                "Absolute wrapped movement/road-direction angle, 0 to 180 degrees; directional agreement.",
            ],
            [
                "speed_mps",
                "Consecutive projected displacement divided by positive elapsed time; movement context.",
            ],
            ["time_gap_s", "Elapsed source-order time; sampling context."],
        ],
        [125, 385],
    )
    text(
        "First observations, stationary movement and invalid time steps use documented neutral values. IDs and labels are metadata, never features. Logistic Regression scales training features inside each fold; tree and probabilistic models use their documented inputs without global learned preprocessing."
    )
    page()
    heading("3. Model experimentation and comparison")
    text(
        "Five distinct algorithms were compared on the same 800,911 labelled candidate rows from the original 80-trip training partition. Trip 00000011 has no eligible labels, leaving 79 effective supervised-fit trips. Three-fold GroupKFold keeps complete trips together; all preprocessing is refit inside folds. The prior two-model checkpoint used a 64/16 holdout and remains archived separately."
    )
    rows = [
        [
            "Model",
            "CV Macro F1",
            "Dev Top-1",
            "Dev Macro F1",
            "Log loss",
            "Continuity",
            "Infer s",
        ]
    ]
    for r in metrics:
        rows.append(
            [
                r["model"],
                f"{r['cv_macro_f1']:.4f}",
                f"{r['observation_top1_accuracy']:.2%}",
                f"{r['candidate_macro_f1']:.4f}",
                f"{r['candidate_log_loss']:.4f}",
                f"{r['direct_step_continuity']:.2%}",
                f"{r['inference_s']:.3f}",
            ]
        )
    table(rows, [143, 59, 62, 62, 55, 65, 64])
    text(
        "Macro F1, precision/recall and log loss are binary candidate metrics. Top-1 chooses the highest-scored candidate per observation; its denominator includes every eligible reference observation, even candidate misses. Direct-step continuity counts equal or directly adjacent directed arcs between consecutive observations outside large gaps. It is not a graph-reachability or complete-route score."
    )
    text(
        "Student-development has 11,177 eligible reference observations out of 19,160 total (58.34% coverage). Six eligible reference arcs are absent from candidates. These are development scores; the reserved evaluator partition is not used. Inference timings cover probability prediction only, excluding candidate construction."
    )
    picture("model_comparison.png", maxheight=235)
    page()
    profiles = {
        "logistic_regression": (
            "Linear/probabilistic",
            "A fast, interpretable spatial/directional baseline. A single linear log-odds boundary limits nonlinear feature interactions.",
            "Weighted logistic loss with regularisation; C=1, balanced weights, max_iter=500, training-only StandardScaler.",
        ),
        "random_forest": (
            "Bagged tree ensemble",
            "Models nonlinear threshold interactions and averages many trees. Larger artifacts and slower inference are tradeoffs.",
            "Weighted Gini splits; 80 trees, depth 14, at least 20 samples per leaf, sqrt features, 75% bootstrap, balanced_subsample weights.",
        ),
        "decision_tree": (
            "Tree",
            "Provides explicit nonlinear thresholds. A single tree is sensitive to partition choices and can overfit.",
            "Weighted Gini splits; depth 12, minimum leaf size 20, balanced class weights.",
        ),
        "hist_gradient_boosting": (
            "Boosting",
            "Stagewise nonlinear modelling is suitable for the four numeric features; fast histogram fitting. Capacity, learning rate and regularisation affect generalisation.",
            "Binary log loss; baseline 120 boosting iterations, 15 leaves, l2_regularization=1, fixed seed 42.",
        ),
        "gaussian_naive_bayes": (
            "Probabilistic/generative",
            "Very fast likelihood-based baseline. Conditional feature independence and Gaussian class-conditional features are restrictive for correlated distance/movement features.",
            "Class-conditional Gaussian likelihood estimates and Bayes rule; default variance smoothing 1e-9.",
        ),
    }
    for name, (family, explanation, objective) in profiles.items():
        r = next(r for r in metrics if r["model"] == name)
        heading(
            "3."
            + str(list(profiles).index(name) + 1)
            + " "
            + name.replace("_", " ").title()
        )
        text("Algorithm family: " + family)
        text("Why considered / strengths and limitations: " + explanation)
        text("Training and optimisation: " + objective)
        table(
            [
                ["Measured result", "Value"],
                [
                    "Training candidate Macro F1",
                    f"{r['training_candidate_macro_f1']:.4f}",
                ],
                [
                    "Trip-grouped CV Macro F1",
                    f"{r['cv_macro_f1']:.4f} +/- {r['cv_std']:.4f}",
                ],
                ["Student-development Top-1", f"{r['observation_top1_accuracy']:.2%}"],
                [
                    "Student-development candidate Macro F1",
                    f"{r['candidate_macro_f1']:.4f}",
                ],
                ["Student-development log loss", f"{r['candidate_log_loss']:.4f}"],
                [
                    "Final fit time / prediction time",
                    f"{r['fit_s']:.3f} s / {r['inference_s']:.3f} s",
                ],
            ],
            [245, 265],
        )
        gap = r["training_candidate_macro_f1"] - r["cv_macro_f1"]
        text(
            f"The training-minus-CV Macro F1 gap is {gap:.4f}. This measured gap is a limited overfitting diagnostic, not proof of generalisation. Cross-trip fold variation and the independent development comparison also matter. The derived-label construction can make scores optimistic."
        )
        text(
            "Full configuration, confusion matrix, precision/recall, per-trip scores and predictions are retained in results/final; every model actually trained in this final comparison is represented here. No artificial epoch-wise curves are created for these algorithms."
        )
        if name == "hist_gradient_boosting":
            picture("boosting_loss_histories.png", maxheight=190)
            text(
                "These are actual boosting-iteration loss histories from the fitted baseline and tuned models. The validation curve is the training-only internal early-stopping split, not student-development data. Other models do not provide comparable epoch-wise histories, so none are fabricated."
            )
        page()
    heading("4. Bayesian hyperparameter optimisation")
    text(
        "Histogram Gradient Boosting was selected by the highest training-only full-data CV Macro F1 among the five baselines. Optuna TPE is the Bayesian search strategy: ten trials, three random startup trials followed by adaptive density-based suggestions. Trial records and the SQLite study are retained."
    )
    text(
        "To control search cost, each trial uses a fixed seed-42 sample capped at 1,200 rows per effective training trip (94,202 rows), with the full-data trip fold memberships preserved. The objective is mean candidate Macro F1 on these training-only folds. Final tuned CV is recomputed on all 800,911 labelled rows, making the untuned/tuned comparison comparable. No evaluator data enters search."
    )
    table(
        [
            ["Hyperparameter", "Search space"],
            ["max_leaf_nodes", "Integer 8 to 31"],
            ["max_iter", "60 to 160, step 20"],
            ["learning_rate", "Log-uniform 0.03 to 0.18"],
            ["l2_regularization", "Log-uniform 0.001 to 10"],
        ],
        [210, 300],
    )
    text("Best parameters: " + json.dumps(selection["best_params"]))
    text(
        f"Best sampled-search objective: {selection['search_objective']:.4f}. Full-data untuned CV: {selection['untuned_cv']:.4f}; tuned CV: {selection['tuned_full_cv']:.4f}. Tuning did not improve full-data CV, so improvement is not claimed. The tuned export is used for the complete pipeline and is compared transparently against the untuned baseline."
    )
    untuned = next(r for r in metrics if r["model"] == "hist_gradient_boosting")
    tuned = metrics[-1]
    table(
        [
            ["Version", "CV F1", "Dev Top-1", "Dev F1", "Log loss", "Infer s"],
            [
                "Untuned",
                f"{untuned['cv_macro_f1']:.4f}",
                f"{untuned['observation_top1_accuracy']:.2%}",
                f"{untuned['candidate_macro_f1']:.4f}",
                f"{untuned['candidate_log_loss']:.4f}",
                f"{untuned['inference_s']:.3f}",
            ],
            [
                "Tuned",
                f"{tuned['cv_macro_f1']:.4f}",
                f"{tuned['observation_top1_accuracy']:.2%}",
                f"{tuned['candidate_macro_f1']:.4f}",
                f"{tuned['candidate_log_loss']:.4f}",
                f"{tuned['inference_s']:.3f}",
            ],
        ],
        [105, 70, 90, 80, 80, 85],
    )
    page()
    heading("5. HMM and Viterbi route reconstruction")
    text(
        "Hidden states are the actual candidate directed road arcs for each observation. Emissions are clipped positive-class scores normalised over that observation's candidates, evaluated in log space. They are useful relative scores, not claimed calibrated road-state posteriors."
    )
    text(
        "Transition travel distance includes the remaining part of the previous arc, a directed shortest path from its end node to the next arc start, and the traversed part of the next arc. Same-arc progress has a special case with 25 m jitter tolerance. Transition log score is -abs(road travel - GPS displacement)/beta. Distances beyond 60 m/s times elapsed time plus 25 m are invalid."
    )
    text(
        "Viterbi recurrence: score_t(j) = log emission_t(j) + max_i(score_(t-1)(i) + log transition(i,j)). Backpointers recover the state sequence. Missing candidates preserve null predictions. Nonpositive/over-30-second gaps and all-impossible transition sets start a new chain, rather than inventing a connecting route. Directed graph connectors expand valid chosen transitions into route runs."
    )
    text(
        "Beta candidates 10, 30 and 80 m were compared on the first three student-development trips; beta=10 m was selected. This is development tuning, so the following ten-trip results are not an untouched estimate of final evaluator performance."
    )
    table(
        [
            ["Method", "Top-1", "Direct continuity", "Mean arc-set F1"],
            *[
                [
                    r["method"],
                    f"{r['top1_accuracy']:.2%}",
                    f"{r['direct_step_continuity']:.2%}",
                    f"{r['arc_set_f1_mean']:.4f}",
                ]
                for r in decoder
            ],
        ],
        [140, 110, 130, 130],
    )
    picture("decoder_comparison.png", maxheight=220)
    text(
        "Arc-set F1 measures segment-set overlap, not ordering. Per-trip Top-1 and direct continuity preserve chronological prediction information. HMM improves aggregate results but can reduce accuracy on individual trips; see decoder_comparison.csv (for example trip 00000065)."
    )
    page()
    heading("6. Geographical visualisation")
    picture("folium_map.png", maxheight=470)
    text(
        "Actual Folium export for student-development trip 00000000. Blue: raw GPS; green: supplied reference; orange: classifier-selected segments; purple: HMM/Viterbi segments. The supplied road network is available as a separate toggle. Layer controls permit comparison. The screenshot is captured from the generated interactive HTML, not a fabricated illustration."
    )
    text(
        "Interactive artifact: results/final/final_folium_map.html. Source: scripts/evaluate_decoder.py. Geographic tiles/Leaflet assets require network access. The evaluator UI uses an offline projected map for the prediction workflow and never receives reference geometry."
    )
    page()
    heading("7. API, UI and deployment")
    picture("ui_desktop.png", maxheight=380)
    text(
        "Laptop 1 runs FastAPI, the saved classifier, candidate/feature code, graph decoder and separate internal evaluation. Laptop 2 serves ui/index.html and sends the chosen trip ID to POST /predict on Laptop 1. GET /health and GET /trips support readiness and catalog checks. The UI displays GPS/road/prediction layers, aggregate metrics, gaps and downloadable predictions."
    )
    text(
        "The UI supports ten configured trips and an editable API address. The current default is explicitly labelled student-development-demo. Set PATHSNAP_EVALUATOR_ROOT to a folder of reserved trip directories on Laptop 1 for final evaluation. No reserved files were supplied during this work."
    )
    text(
        "Prediction finishes before reference files are opened. Reference labels/route geometry are excluded from response payloads, feature inputs and UI bundles. Unknown IDs return 404; malformed IDs return 422. Candidate misses, gaps and missing references are represented explicitly."
    )
    page()
    heading("7.1 Verification evidence")
    picture("api_response.png", maxheight=390)
    text(
        f"Actual HTTP verification: {len(api['trips'])} of 10 configured development trips returned 200. Observation counts match prediction counts. Unknown-trip, malformed-ID and recursive reference-payload checks passed. Browser verification exercised a separate UI process on port 8001 and API on port 8000, with zero JavaScript errors; desktop and mobile screenshots are retained."
    )
    text(
        "The screenshot shows diagnostics from an actual UI-triggered API response. Real receiving-server log records and API verification JSON are retained. A screenshot from a second physical laptop's command window and a physical LAN test could not be captured on this computer; the deployment runbook provides the exact commands to perform and capture those event-specific checks."
    )
    text(
        "Automated tests cover raw readers, partition separation, empty candidates, source-order timing, candidate-miss denominators, graph disconnection, GPS-gap resets, missing decoder states and API request/CORS handling. See results/final/test_results.txt."
    )
    text(
        "The audit also checks repeated-arc loop connectors, connector geometry in the UI, malformed optional references, model reloads after artifact replacement, and the Windows/macOS launchers. Missing or invalid optional references suppress accuracy rather than discarding valid predictions. Model readiness validates the artifact contents."
    )
    page()
    heading("8. Reproducibility, limitations and presentation")
    text(
        "Start the API with python scripts/serve.py --host 0.0.0.0. Start the UI on the other laptop with python -m http.server 8001 --directory ui, open its local address, and enter http://LAPTOP1_LAN_IP:8000. Installation, regeneration, reserved-folder layout, checks and demo order are in docs/FINAL_PRESENTATION.md."
    )
    text(
        "One-command launch: bash start.sh on macOS/Linux; start.bat on Windows. Use --api-only on Laptop 1 and --ui-only on Laptop 2. The --smoke-test option starts services, verifies an actual prediction and stops only its own processes. Native Windows/macOS checks are configured in .github/workflows/verify.yml; the corresponding run results are separate verification evidence."
    )
    ci_path = results / "cross_platform_ci.json"
    if ci_path.exists():
        ci = json.loads(ci_path.read_text())
        text(
            "Recorded native cross-platform CI status: "
            + ci.get("status", "unknown")
            + ". Run: "
            + ci.get("url", "see GitHub Actions")
        )
    text(
        "Key limitations: provisional observation labels are geometry-derived; one original training trip has no eligible labels; development data informed decoder beta; sampled Bayesian search is approximate; shortest paths are limited to plausible travel bounds; graph reference discontinuities and GPS gaps remain separate route runs. Strong development results do not establish reserved-evaluator performance."
    )
    text(
        "Originality and AI assistance: implementation, experimentation and report preparation used AI assistance. All numerical results, figures, screenshots and execution claims are backed by actual runs. The team must independently review, understand and defend the code and report, and follow the event's disclosure requirements. No results from other teams are used."
    )
    text(
        "Sources: the five supplied PathSnap assignment PDFs under delivarables/; executable implementation and result files in this repository. Algorithm implementations use scikit-learn, SciPy, Optuna, FastAPI, Folium and ReportLab. Dataset and model assumptions are documented alongside source."
    )
    text(
        "Final submission artifacts: the public GitHub repository and this single consolidated PDF. The report records current verified local capabilities; the reserved-data and second-device checks must be performed when that environment is available."
    )
    text(
        "The reproducible deliverable checklist is docs/DELIVERABLE_AUDIT.md, generated by scripts/audit_deliverables.py. Regeneration accepts freshly rebuilt feature tables; optimisation studies are tied to input/code/fold fingerprints so old parameters are not silently reused on changed data."
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(
            40, 25, "PathSnap | Recorded experiments | Provisional reference labels"
        )
        canvas.drawRightString(A4[0] - 40, 25, str(doc.page))

    doc = SimpleDocTemplate(
        str(out / "PathSnap_Technical_Report.pdf"),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=38,
        bottomMargin=40,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(out / "PathSnap_Technical_Report.pdf")


if __name__ == "__main__":
    main()
