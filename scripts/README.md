# Scripts

`train_baselines.py` fits both models, records grouped cross-validation and
held-out metrics, exports joblib bundles and creates the comparison plot.
`map_baselines.py` creates the interactive development-trip comparison map.
`predict_baseline.py` runs either saved model from GPS/network inputs without
reading reference labels. See docs/two_model_baseline.md for commands.

`python scripts/build_candidate_features.py --partition train` writes a shared candidate feature CSV and observation/trip coverage CSV under `cache/candidate_features/`, outside the raw data. It reads Part 1's `results/part1/split_manifest.json` by default, with logical `train`, `validation`, and `student_test` splits, and calls Part 1's `load_trip`. Supply a different JSON manifest with `--manifest PATH` if needed. A raw-file adapter is used only if the Part 1 loader is unavailable. Supply provisional Part 1 labels separately with `--references PATH`; the script writes a separate `*_supervised.csv` and observation-level `*_reference_coverage.csv`. Labels are never used to build candidates. See candidate/features README files for radii and missing-value behavior.
