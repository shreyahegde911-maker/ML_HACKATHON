# Final presentation runbook

## Current verified scope

The repo includes strict data processing, geographic EDA, candidate features,
five-model comparisons, training-only trip-grouped CV, Bayesian TPE search,
saved final artifacts, directed graph HMM/Viterbi, model/decoder comparisons,
Folium visualisation, a FastAPI backend, a separate responsive UI, a consolidated
technical PDF, and actual HTTP/browser/test evidence.

The supplied data has 80 training and 10 student-development trips. No reserved
evaluator files were supplied. The default presentation is visibly labelled
`student-development-demo`; it does not claim final evaluator performance.
Separate UI/API processes and cross-origin requests were verified on one
computer. A second physical laptop/LAN was not accessed by the implementation
session. Perform the two-device connectivity check below at the event.

## Laptop 1: API and trained model

### Easy launch (macOS and Windows)

Install Python 3.12 once. On macOS/Linux run `bash start.sh`; on Windows
double-click `start.bat`. Both services start and the browser opens automatically.
The first run creates `.venv` and installs pinned dependencies, so it needs
internet. Later starts use the existing environment and saved model.

For the two-laptop presentation:

| Machine | macOS/Linux | Windows |
| --- | --- | --- |
| Laptop 1 (API/model) | `bash start.sh --api-only` | `start.bat --api-only` |
| Laptop 2 (UI) | `bash start.sh --ui-only` | `start.bat --ui-only` |

On Laptop 2, enter `http://LAPTOP1_LAN_IP:8000` and click Connect. Keep launcher
windows open. Ctrl+C stops launcher-owned processes, while pre-existing healthy
services are left running. `--no-browser` suppresses browser opening;
`--check --skip-install` checks an existing setup without starting anything.
Custom ports are available with `--api-port` and `--ui-port`. The browser URL
passes the custom API address to the UI automatically.

### Manual startup

Use Python 3.12. From a clone of the repository:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Windows activation is `.venv\Scripts\activate` instead of `source`.
No retraining is needed: `artifacts/final/selected.joblib` is included.
The server logs actual trip IDs, observation counts, runtime and metrics.
Keep this terminal visible to the evaluator. API documentation is available
at `http://localhost:8000/docs`; health at `/health`; trip catalog at `/trips`.

## Laptop 2: evaluator UI

Only the UI file and Python are needed; model/data/reference files stay on
Laptop 1. From the repository root on Laptop 2:

```sh
python -m http.server 8001 --directory ui
```

Open `http://localhost:8001`. Enter `http://LAPTOP1_LAN_IP:8000` in the API address
field, click Connect, choose one of the ten trips, then Reconstruct route.
Both laptops must share a reachable network, and Laptop 1 must permit inbound
TCP port 8000. Confirm `/health` is reachable from Laptop 2 before presenting.
The UI's projected map works without tile/CDN downloads. Folium HTML uses online
Leaflet assets/background tiles; screenshots are included for offline viewing.

## Reserved evaluator trip configuration

On Laptop 1 only, arrange the reserved folder as:

```text
/path/to/evaluator_trips/
  00000100/
    00000100.track
    00000100.nodes
    00000100.arcs
    00000100.route   # server-only reference, if supplied
  ... nine more trip directories ...
```

Set the root before starting the API:

```sh
export PATHSNAP_EVALUATOR_ROOT=/path/to/evaluator_trips
python scripts/serve.py --host 0.0.0.0
```

Windows PowerShell: `$env:PATHSNAP_EVALUATOR_ROOT="C:\path\to\evaluator_trips"`.
The API catalog switches to `reserved-evaluator`. Eight-digit numeric IDs are
required. The prediction loader reads only track/nodes/arcs. After prediction,
the separate server evaluator may align the private route to calculate metrics.
If route is absent, predictions still run and reference-based scores are N/A.
Do not copy evaluator routes/labels to the UI laptop or use evaluator scores to
change models, thresholds or decoder settings. Confirm official scoring rules:
the available assignment specifies route sequences, not official point labels.

## Five-minute demonstration

1. Explain the noisy GPS / parallel-road problem and four input file types.
2. Show `results/final/eda.png` and the raw/reference row-count mismatch.
3. Show `model_comparison.png`: five measured models and the tuned winner.
4. Show `bayesian_trials.csv`: ten real TPE trials; disclose sampled search and
   the slight CV decrease after tuning rather than claiming improvement.
5. Open the UI, select any configured trip and click Reconstruct. Toggle raw
   GPS/classifier/HMM layers, pan/zoom, and show the API terminal request log.
6. Show `decoder_comparison.png`: aggregate accuracy and continuity improve;
   discuss per-trip exceptions and gaps.
7. Show the report and verification evidence. State the provisional-label and
   development-only evaluation limits clearly.

## Verification

```sh
python -m unittest discover -s tests -v
python scripts/verify_api.py
```

Optional browser QA: `npm install`, `npx playwright install chromium`, then
`node scripts/verify_browser.cjs` while UI/API are running. An installed Chrome
can be selected with `PATHSNAP_BROWSER_PATH` if Chromium is not installed.
Results include desktop/mobile screenshots and a zero-JavaScript-error record.

## Reproduce experiments and report

```sh
python scripts/inspect_data.py --partition all --align
python scripts/build_candidate_features.py --partition train --references-dir data/derived/reference_labels --reliable-status aligned
python scripts/build_candidate_features.py --partition validation --references-dir data/derived/reference_labels --reliable-status aligned
python scripts/build_candidate_features.py --partition student_test --references-dir data/derived/reference_labels --reliable-status aligned
python scripts/final_experiments.py --features cache/candidate_features
python scripts/eda.py
python scripts/evaluate_decoder.py
python scripts/training_diagnostics.py
python scripts/verify_api.py
node scripts/verify_browser.cjs
python scripts/build_report.py
python scripts/audit_deliverables.py
```

The feature artifacts already in `artifacts/features/` can be used directly.
The original 64/16 training split is recombined into the original 80-trip
partition for final CV/fitting; one trip lacks eligible labels, leaving 79
effective fit trips. The ten student-development trips remain outside fitting.
TPE uses a fixed per-trip capped training sample, then the selected configuration
is re-evaluated on full-data folds. Studies are identified by exact input hashes,
model configuration, fold memberships, sample and script version. An unchanged
context reuses its ten completed trials; changed data/code creates a new study
in the same SQLite archive. The final experiment script now computes continuity
and artifact checksums itself, so the report can be rebuilt without manual steps.

After verification, run `python scripts/audit_deliverables.py` for the generated
assignment checklist and explicit event-specific pending items. The same audit
runs in the native macOS/Windows CI workflow.

## Presentation evidence

- `output/pdf/PathSnap_Technical_Report.pdf`: consolidated technical report.
- `results/final/metrics.json`, `comparison.csv`, `model_configurations.json`.
- `results/final/folds.json`, `bayesian_trials.csv`, `bayesian_search.sqlite3`.
- `results/final/decoder_comparison.csv`, `decoder_summary.json`, `decoder_search.json`.
- `results/final/final_folium_map.html`, `folium_map.png`.
- `results/final/ui_desktop.png`, `ui_mobile.png`, `api_response.png`.
- `results/final/api_verification.json`, `browser_verification.json`, `test_results.txt`.

The report includes an actual browser API-response screenshot. Capture the
receiving terminal and second-laptop UI at the event to satisfy the physical
two-laptop screenshot requirement; no such screenshot is fabricated here.
AI assistance was used for implementation and writing; the team must review,
understand and disclose it according to the event's policy.
