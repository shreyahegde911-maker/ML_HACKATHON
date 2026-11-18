# PathSnap

PathSnap reconstructs vehicle routes from noisy GPS observations and a road
network. The existing module layout supports data processing, candidate
generation, classifiers, HMM/Viterbi decoding, evaluation, FastAPI and a separate
evaluator UI.

## Implemented: Parts 1 through 4 (two-model baseline)

- Strict trip readers preserving original observation/node/arc IDs and order.
- Validation of shapes, coordinates, ID bounds, timestamps and route breaks.
- Shared per-trip WGS84 UTM coordinates in metres for observations and roads.
- Prediction-data loading that never reads reference routes.
- Separate ordered **provisional** reference alignment with uncertainty flags.
- Seed-42 whole-trip split: 64 training, 16 validation, 10 student-development.
- Runnable inspection script, source-invariance checks and focused tests.

The candidate/feature pipeline and two classifiers are implemented and run:
Logistic Regression and Random Forest. Actual shared feature tables, trained
artifacts, trip-aware three-fold CV, validation/student-development comparisons,
predictions, plots and a Folium comparison map are included. Scores measure
agreement with provisional derived references, not official pointwise accuracy.
See [the baseline workflow](docs/two_model_baseline.md) and
[recorded results](results/baselines/README.md).

The full assignment still requires at least five models, Bayesian tuning,
production HMM/Viterbi experiments, the API/UI and final report.

## Setup and execution

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/inspect_data.py --data-root "Student folder" --partition train --trip 00000001 --align --output-dir results/part1_example
python scripts/inspect_data.py --data-root "Student folder" --partition all --align --output-dir results/part1
python scripts/train_baselines.py --features artifacts/features --jobs 2
python scripts/map_baselines.py
python scripts/predict_baseline.py --model random_forest --partition student_test --trip-id 00000000
```

Inspection outputs are in `results/part1/`; the shared trip manifest is
`results/part1/split_manifest.json`. Generated reference-label CSVs are in
`data/derived/reference_labels/{partition}/{trip_id}.csv` and are gitignored.
Run `scripts/inspect_data.py --partition all --align` to regenerate them.

## Part 2 handoff

```python
from pathsnap.data import load_trip

trip = load_trip("Student folder", "train", "00000001")
observations, nodes, arcs = trip["observations"], trip["nodes"], trip["arcs"]
```

Set `PYTHONPATH=src` or add `src` to Python's import path. The exact DataFrame
schemas, reference-alignment method, status filtering, partition mapping and
limitations are documented in [the data notes](docs/data_assumptions.md).

## Data handling

Use the original supplied files in `Student folder/` without modification.
`.track` rows are GPS observations; `.nodes` rows define node IDs; `.arcs` rows
define arc IDs; `.route` is an ordered reference sequence, not one label per GPS
observation. Only alignment status `aligned` is eligible provisional supervision.
Keep references out of candidate generation and prediction inputs. Keep reserved
evaluator data out of development and evaluator ground truth server-side.

## Layout

- `src/pathsnap/`: reusable processing, candidates, features, models, decoding,
  evaluation, tuning, visualisation and API modules.
- `scripts/`, `configs/`: runnable entry points and reproducible settings.
- `notebooks/`, `docs/`, `data/`: exploration and data/implementation notes.
- `artifacts/`, `results/`: actual trained artifacts and recorded run outputs.
- `tests/`, `ui/`: checks and the separate evaluator interface.
- `delivarables/`: original assignment PDFs (existing repository spelling).
