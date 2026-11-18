# Actual two-model baseline results

Scores measure agreement with **provisional derived labels**, not official ground-truth accuracy.

| Model | Group CV candidate Macro F1 | Validation Top-1 | Validation candidate Macro F1 | Student-development Top-1 | Student-development candidate Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.7891 ± 0.0068 | 84.12% | 0.7887 | 88.74% | 0.8080 |
| random_forest | 0.8259 ± 0.0124 | 84.70% | 0.8333 | 88.91% | 0.8459 |

Random Forest is the selected accuracy baseline using validation results. Logistic Regression is much faster; compare the recorded inference timings before choosing a deployment tradeoff.

The manifest contains 64 training / 16 validation / 10 student-development trips. Actual supervised fitting uses 681,341 labelled candidate rows across 63 training trips; trip 00000011 has no eligible labels. Both models use the same rows and three-fold trip-grouped CV.

Validation contains 16,775 eligible reference observations out of 29,235 total. Student-development contains 11,177 eligible observations out of 19,160. Scores exclude flagged reference alignments; missing reference candidates stay in the Top-1 denominator.

Candidate misses: training 87 / 108,326; validation 834 / 16,775; student-development 6 / 11,177. All observations currently have at least one candidate, although reference arcs can be missing.

See comparison.csv / metrics.json for candidate precision, recall, binary log loss, confusion matrices and measured timings. cross_validation.json retains fold trip IDs and scores. The timings exclude candidate generation; final-fit timings exclude CV.

comparison.png and cross_validation.png use actual runs. comparison_map.html shows GPS, roads, the supplied development reference and both classifier outputs for validation trip 00000001. Predictions are pointwise segment matches; no HMM route reconstruction is claimed.

Compressed shared tables are under artifacts/features/ and saved models under artifacts/models/. Run instructions and metric definitions: docs/two_model_baseline.md. Five-model comparison, Bayesian tuning, HMM/Viterbi, API/UI and the final report remain outside this checkpoint.
