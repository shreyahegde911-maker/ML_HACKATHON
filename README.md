# PathSnap

End-to-end GPS map matching: strict readers, geographic EDA, road candidates,
five trained classifiers, Bayesian optimisation, directed graph HMM/Viterbi,
FastAPI, a separate responsive UI, Folium maps and a technical report.

## Start the presentation

Install Python 3.12 once. Then use the launcher from the repository root:

**macOS/Linux:**

```sh
bash start.sh
```

**Windows:** double-click `start.bat`, or run:

```bat
start.bat
```

The launcher creates `.venv` and installs pinned dependencies if needed, starts
both API and UI, waits for readiness, and opens the browser. First-time setup
needs internet; later starts reuse installed dependencies and trained models.
Keep the window open; **Ctrl+C** stops only processes started by that launcher.
Existing healthy PathSnap services are reused. A conflicting app on either port
produces a clear error without stopping it. Paths containing spaces are supported.

For two laptops, use `bash start.sh --api-only` / `start.bat --api-only` on
Laptop 1, and `bash start.sh --ui-only` / `start.bat --ui-only` on Laptop 2.
UI-only mode needs no model, data or ML dependencies; enter Laptop 1's API
address in the UI. `--no-browser`, `--check`, `--skip-install`, `--api-port`
and `--ui-port` are also available (`--help` lists them).

For manual startup instead:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/serve.py --host 0.0.0.0 --port 8000
```

In another terminal (or on Laptop 2):

```sh
python scripts/serve_ui.py --port 8001
```

Open **http://localhost:8001**. For two laptops, enter
`http://LAPTOP1_LAN_IP:8000` in the UI API field. The model and data stay on
Laptop 1. The saved final model is included; retraining is not required.
The API also serves the UI at http://localhost:8000 for a one-process demo.

The default catalog is the ten supplied **student-development** trips, visibly
labelled in the UI. Reserved evaluator files were not supplied. Set
`PATHSNAP_EVALUATOR_ROOT` on the backend to switch to reserved trips. References
are read only by server evaluation after prediction, and never returned to the
UI. The physical two-laptop LAN check must be performed at the event.

## Recorded final results

Five baselines were compared using three-fold trip-grouped CV across the
original 80 training trips. One trip has no eligible derived labels, leaving
79 effective supervised-fit trips and 800,911 labelled candidate rows.
The ten student-development trips stay outside fitting.

| Model | Training CV candidate Macro F1 | Student-development Top-1 |
| --- | ---: | ---: |
| Logistic Regression | 0.7902 | 88.78% |
| Random Forest | 0.8280 | 88.87% |
| Decision Tree | 0.8140 | 88.10% |
| Histogram Gradient Boosting | **0.8510** | **89.13%** |
| Gaussian Naive Bayes | 0.8360 | 87.77% |
| Bayesian-tuned Histogram Gradient Boosting | 0.8505 | 89.06% |

Ten Optuna TPE trials used a fixed training-only sample with whole-trip folds;
final tuned CV was recomputed on all labelled training rows. Tuning did not
improve full-data CV, and no tuning gain is claimed.

The tuned classifier + HMM achieved **96.31% Top-1** versus 89.06% classifier-only,
and **97.99% direct-step continuity** versus 89.21%, on student-development trips.
Decoder beta was selected on three of those development trips. These are
**development agreement scores against provisional derived references**, not
untouched evaluator accuracy. Alignment coverage is 58.34%; flagged references
are excluded and candidate misses remain in the Top-1 denominator.

## Presentation artifacts

- [Consolidated report](output/pdf/PathSnap_Technical_Report.pdf)
- [Deployment and viva runbook](docs/FINAL_PRESENTATION.md)
- [Comparison and timings](results/final/comparison.csv)
- [Decoder comparison](results/final/decoder_summary.json)
- [Interactive Folium map](results/final/final_folium_map.html)
- [Actual UI screenshot](results/final/ui_desktop.png)
- [All-ten-trip HTTP checks](results/final/api_verification.json)
- [Browser verification](results/final/browser_verification.json)
- [Test record](results/final/test_results.txt)

The UI opens a real OpenStreetMap street map with GPS and prediction overlays.
Street tiles need internet; the **Network** view works offline and is selected
automatically if tiles fail. Leaflet 1.9.4 is bundled locally with its license.
Connection settings and evaluation diagnostics are collapsed to keep the
workspace minimal. Folium's separate report map needs online assets/tiles;
rendered screenshots are included for offline presentation.

## Verification and reproduction

Reserved evaluation uses a separate local workflow documented in
[the evaluator runbook](docs/EVALUATOR.md). Both launchers accept
`--evaluator-root "path/to/trip-parent"`, preserve actual numeric trip IDs and
support the discovered trip count. Reserved data, local paths, scores and
private reports are ignored by Git; they must not be published without permission.

```sh
python -m unittest discover -s tests -v
python scripts/verify_api.py
```

See the runbook for full regeneration, browser checks and reserved-folder setup.
Run `python scripts/audit_deliverables.py` for an artifact/model/report audit and
the event-specific pending items. The macOS/Windows workflow also exercises the
native launcher and a real start/predict/stop cycle. For regenerated feature
tables, use `python scripts/final_experiments.py --features cache/candidate_features`;
without that option it reproduces from the committed compressed shared tables.
The earlier two-model checkpoint remains under results/baselines and
docs/two_model_baseline.md; results/final is the final comparison.

## Data and architecture

Original files in Student folder/ are unchanged: track holds GPS/time; nodes
defines road coordinates; arcs defines directed segments; route is an ordered
reference sequence. GPS/reference rows cannot be joined directly.
[Data assumptions](docs/data_assumptions.md) explain projection and label flags.
Only distance, heading difference, speed and elapsed time enter the models.
IDs, references and statuses are metadata. Candidate generation and prediction
never read routes. HMM uses classifier scores and plausible directed shortest
paths; gaps/disconnections start separate runs instead of invented bridges.

Source: src/pathsnap. Commands: scripts. Interface: ui. Artifacts: artifacts/final.
Results: results/final. Original assignment PDFs: delivarables. AI assistance is
disclosed in the report; the team must review and defend the implementation.
