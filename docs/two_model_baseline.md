# Parts 2, 3 and 4: reproducible two-model baseline

## What is implemented

Part 2 generates network-only road candidates and four model features. Empty
candidate tables retain reference coverage and no longer crash label attachment.
Features preserve source observation order, so backwards time steps receive the
documented neutral treatment. Candidate generation uses NumPy arrays for exact
distances, avoiding repeated per-observation pandas slicing. Integer arc IDs,
length-stratified conservative spatial indexing, exact distances and reference
independence are preserved.

Part 3 trains a StandardScaler + Logistic Regression pipeline. Part 4 trains
Random Forest and creates a Folium comparison map. Both models use exactly
distance_m, heading_difference, speed_mps and time_gap_s. IDs, references,
alignment status and search diagnostics are excluded from input features.

## Reproduce from raw files

Use Python 3.12, install `requirements.txt`, and run from the repository root:

```sh
python -m unittest discover -s tests -v
python scripts/inspect_data.py --partition all --align
python scripts/build_candidate_features.py --partition train --references-dir data/derived/reference_labels --reliable-status aligned
python scripts/build_candidate_features.py --partition validation --references-dir data/derived/reference_labels --reliable-status aligned
python scripts/build_candidate_features.py --partition student_test --references-dir data/derived/reference_labels --reliable-status aligned
python scripts/train_baselines.py --jobs 2
python scripts/map_baselines.py
```

The candidate script accepts `--data-root PATH` and `--output PATH`. Alignment
CSV files remain separate from candidate generation and are joined afterwards.
The generated shared tables are cached under `cache/candidate_features/`.

Compressed copies of the actual shared supervised feature tables and
observation-coverage tables are committed under `artifacts/features/`. To train
directly from those tables, without regenerating candidates:

```sh
python scripts/train_baselines.py --features artifacts/features --jobs 2
```

Run either saved model on GPS/network data without reading references:

```sh
python scripts/predict_baseline.py --model logistic_regression --trip-id 00000000
python scripts/predict_baseline.py --model random_forest --trip-id 00000000
```

Feature-artifact compressed/uncompressed checksums are recorded in
`results/baselines/feature_artifacts.json`. These tables include derived reference
metadata for training/evaluation; they must not become prediction/UI inputs.

## Fit and evaluation design

The shared seed-42 manifest assigns 64 training trips, 16 validation trips and
10 student-development trips. Fit only candidate rows whose provisional labels
are present. One manifest training trip has no eligible candidate labels, so
the actual supervised fit and group cross-validation cover 63 training trips.
The excluded supervised-fit trip is `00000011`.
Record its identity in run.json; retain its observations in data/coverage outputs.

Use three-fold GroupKFold within the training subset, grouped by trip ID. This
is baseline cross-validation on the 64-trip training subset, not the final
assignment-wide cross-validation over all 80 training trips. StandardScaler is
refitted independently inside each fold, then on final training rows only.
Validation and student-development trips never enter model fitting. Parameters
are fixed baseline choices with seed 42; Bayesian optimisation is not claimed.

Logistic Regression uses C=1, balanced class weights, and up to 500 optimisation
iterations. Random Forest uses 80 trees, depth at most 14, at least 20 samples
per leaf, square-root feature subsampling, 75% bootstrap sampling and balanced
subsample class weights. Full primitive model settings and dependency versions
are recorded in run.json. Saved joblib bundles include the exact feature order,
actual training trip IDs and provisional-label provenance.

## Metric definitions and limitations

- Candidate classification uses threshold 0.5. Report binary Macro F1, Macro
  Precision/Recall, accuracy, binary log loss and 2x2 confusion matrices.
- Observation Top-1 selects the candidate with highest positive-class score.
  Ties follow stable candidate order, without consulting references.
- Top-1 denominator is all eligible reference observations, including missing
  reference candidates and observations with no candidates. Missing predictions
  count as incorrect. Flagged reference alignments are excluded and alignment
  coverage is reported alongside the score.
- Candidate miss rate is the fraction of eligible observations whose reference
  arc is absent from their candidate set. It is separate from classifier errors.
- Per-trip tables expose variability hidden by aggregate observation weighting.
- Training timing excludes cross-validation; per-fold timing includes its fit
  and scoring. Inference timing measures candidate probability prediction only;
  it excludes raw loading, candidate generation, plotting and evaluation joins.
- Classification scores are not calibrated per-observation road probabilities.
  No production HMM/Viterbi, route interpolation or decoder improvement is
  claimed. The Folium map draws selected road segments, not an invented
  connected reconstructed route.

All reference-based metrics use the provisional alignment from Part 1. They are
agreement with derived supervision, not independent physical ground-truth
accuracy. Alignment geometry overlaps model inputs, which can make these
scores optimistic. Candidate-miss and alignment-coverage results must accompany
the comparison. Reserved evaluator data is not processed.

## Outputs

- `artifacts/models/`: fitted Logistic Regression pipeline and Random Forest.
- `artifacts/features/`: compressed shared candidate/label/coverage tables.
- `results/baselines/comparison.csv`, `metrics.json`: actual model metrics.
- `results/baselines/cross_validation.json`: fold scores and trip boundaries.
- `results/baselines/*_predictions.csv.gz`: one row per reference observation,
  including flagged alignments and missing predictions.
- `results/baselines/*_per_trip.csv`: per-trip eligible counts and Top-1 scores.
- `results/baselines/comparison.png`: actual held-out metric comparisons.
- `results/baselines/comparison_map.html`: interactive Folium layers for GPS,
  relevant roads, supplied development reference and both classifiers.

Open the Folium HTML in a browser; Leaflet assets and background tiles require
network access. The committed comparison PNG can be viewed offline.

The full assignment still requires at least five models, Bayesian tuning,
classifier-plus-HMM/Viterbi experiments, the API/UI deployment and final report.
