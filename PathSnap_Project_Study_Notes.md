# PathSnap — Simple Project Study Notes

> This guide is based on the uploaded `ML_HACKATHON-main` project. It explains what the code actually implements, how the models are trained, how predictions are calculated, and what the recorded results mean.

## 1. What is our project?

**PathSnap is a GPS map-matching system.**

A phone or GPS device records locations as latitude/longitude points. These points are noisy: they may fall beside a road, between two parallel roads, or slightly away from the actual road because of GPS error.

Our project takes:
- A sequence of GPS points with timestamps.
- A road network made of nodes and directed road segments (arcs).

It predicts which road segment each GPS point most likely belongs to, then reconstructs a sensible route on the road network.

**In simple words:** We convert a messy GPS trace into a route that follows real connected roads.

## 2. What are the input files?

Each trip has four files in `Student folder/`:

| File | Meaning |
|---|---|
| `.track` | GPS observations: longitude, latitude and timestamp |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Directed connections between nodes; these form road segments |
| `.route` | Ordered reference road-arc sequence used for reference alignment/evaluation |

Important: the prediction reader loads the track, nodes and arcs. It does **not** read `.route` while making predictions. The route is kept separate for deriving provisional reference labels and evaluating results.

## 3. Overall workflow

1. **Read and validate data.** Read GPS points and road network; preserve the original observation order.
2. **Convert coordinates.** Convert longitude/latitude into a local UTM coordinate system measured in metres. This makes distance and speed calculations meaningful.
3. **Build road segments.** Join each arc's start and end nodes to obtain a directed line segment.
4. **Generate candidates.** For every GPS point, find nearby road segments that could explain that point.
5. **Calculate features.** Measure distance, heading difference, speed and time gap for each point–road candidate.
6. **Train classifiers.** Train models to estimate how likely each candidate road is to be the correct road.
7. **Choose the best candidate.** The classifier scores candidates for each GPS point.
8. **Apply HMM + Viterbi.** Use road connectivity and movement between consecutive points to select a more consistent sequence of roads.
9. **Display the result.** The FastAPI backend provides predictions to the separate web UI, which displays the reconstructed route and map layers.

## 4. Candidate generation — how do we find possible roads?

We do not compare a GPS point against every road without limits. We first find nearby road segments.

The implemented default settings are:
- Normal search radius: **100 metres**
- Expanded fallback radius: **300 metres**
- Maximum candidates per GPS point: **8**

The code uses a spatial KD-tree to retrieve likely nearby segments, then calculates the exact shortest distance from the GPS point to each line segment. It keeps the nearest candidates within 100 m. If none are found, it tries the expanded 300 m radius and marks that search as expanded.

A candidate is simply a possible road for one GPS observation. One observation can therefore have several candidate roads.

## 5. The four features used by the ML models

The project uses exactly these four model input features:

| Feature | Meaning | Why it helps |
|---|---|---|
| `distance_m` | Distance from GPS point to candidate road segment, in metres | A road closer to the GPS point is usually more plausible |
| `heading_difference` | Absolute angular difference between direction of GPS movement and direction of the road, in degrees (0–180) | A road pointing in the same direction as movement is more plausible |
| `speed_mps` | Estimated GPS movement speed in metres/second | Helps judge whether the movement is realistic |
| `time_gap_s` | Time elapsed since the previous observation, in seconds | Helps interpret how much movement could happen between points |

### How are speed and heading calculated?

- Time gap = current timestamp − previous timestamp, if positive.
- GPS displacement = straight-line distance between consecutive projected GPS points.
- Speed = displacement ÷ time gap. If the point is not moving or the time gap is invalid, the implementation uses 0.
- GPS movement heading is calculated from the change in projected x/y coordinates.
- Road heading is calculated from the direction of the road segment.
- Heading difference is the absolute wrapped angular difference, between 0° and 180°.

**Important:** IDs, reference route IDs and alignment status are metadata, not model features. The models only receive the four features above.

## 6. How were training labels created?

This is an important detail to explain honestly.

The supplied `.route` file is an ordered sequence of road arcs, while the GPS track is a sequence of observations. They do not necessarily have the same number of rows, so they cannot simply be joined row by row.

The project first creates a separate **ordered reference alignment** between GPS observations and the supplied route. It checks geometry and route order and flags questionable observations, for example:
- Ambiguous nearby road geometry.
- Large timestamp gaps or invalid timestamp order.
- Discontinuities in the reference route.
- Implausible progress along the route.
- A reference road segment that is too far from the GPS point.

Only observations with reliable alignment statuses (such as `aligned`, `reliable` or `ok`) are used to create supervised candidate labels.

For each labelled GPS observation:
- Candidate road equals the aligned reference road → label **1** (positive).
- Other candidate road → label **0** (negative).

If the aligned reference road is not among the generated candidates, it is recorded as a **candidate miss**. The model cannot select a road that candidate generation never supplied.

**Limitation:** These are provisional labels derived from route alignment, not official independent point-by-point ground truth. The project explicitly records this warning. Therefore, reported accuracy is agreement with these provisional references, not proof of perfect real-world GPS accuracy.

## 7. Which ML models did we compare?

The final experiment compares six classifier configurations: five model families plus a tuned version of the winner.

| Model | Simple explanation |
|---|---|
| Logistic Regression | Learns a weighted relationship between the four features and the probability that a candidate is correct. The feature pipeline standardises inputs. |
| Random Forest | Trains many decision trees on varied samples/features and combines their predictions. |
| Decision Tree | Learns a sequence of feature-based if/else splits. Easy to understand, but can overfit. |
| Histogram Gradient Boosting | Builds decision trees sequentially. Each new tree focuses on reducing the errors left by the previous trees. |
| Gaussian Naive Bayes | Estimates class probabilities using feature distributions and a conditional-independence assumption. |
| Tuned Histogram Gradient Boosting | Same winning model family, with hyperparameters searched using Optuna's TPE sampler. |

The final script uses these model families:
- Logistic Regression and Random Forest from the shared model factory.
- Decision Tree with maximum depth 12 and minimum leaf size 20.
- Histogram Gradient Boosting with 120 iterations, at most 15 leaf nodes and L2 regularisation 1.0.
- Gaussian Naive Bayes.

The final training uses `GroupKFold` with **3 folds**, grouping by trip ID. This matters because observations from the same trip should not be randomly scattered across training and validation folds.

## 8. Which model was best, and why?

### Best individual classifier: Histogram Gradient Boosting

The recorded final comparison is:

| Model | 3-fold training CV Macro F1 | Student-development Top-1 |
|---|---:|---:|
| Logistic Regression | 0.7902 | 88.78% |
| Random Forest | 0.8280 | 88.87% |
| Decision Tree | 0.8140 | 88.10% |
| **Histogram Gradient Boosting** | **0.8510** | **89.13%** |
| Gaussian Naive Bayes | 0.8360 | 87.77% |
| Tuned Histogram Gradient Boosting | 0.8505 | 89.06% |

**Why did we select Histogram Gradient Boosting?**
- It achieved the highest 3-fold training cross-validation Macro F1 among the compared classifiers (0.8510).
- It also achieved the highest student-development Top-1 score (89.13%) among the listed classifier results.
- It can learn non-linear relationships and combinations between distance, heading, speed and time gap.

### Did hyperparameter tuning improve it?

We used **Optuna with the TPE (Tree-structured Parzen Estimator) sampler** for 10 trials. It searched parameters such as:
- `max_leaf_nodes`
- `max_iter`
- `learning_rate`
- `l2_regularization`

The search objective was training-only, trip-grouped cross-validation Macro F1, using a fixed capped sample from training trips. The selected settings were then evaluated again using all training rows and the three folds.

The full-data CV score changed from **0.851024** to **0.850467** after tuning. The student-development Top-1 changed from **89.13%** to **89.06%**.

So, be accurate: **tuning did not improve the recorded full-data CV or student-development score.** We keep the tuned artifact as an experiment, but the untuned Histogram Gradient Boosting result is slightly better in the recorded comparison. Do not claim that tuning improved performance.

## 9. How exactly is the model trained?

The final experiment follows this process:

1. Load the prepared candidate-feature tables for the training and validation partitions.
2. Combine the original training and validation trips for the final training experiment.
3. Keep only candidate rows with reliable labels.
4. Separate the inputs `X` (the four features) and target `y` (0 or 1).
5. Use trip IDs as groups in 3-fold `GroupKFold` cross-validation.
6. For each fold, fit the model on two groups of trips and calculate Macro F1 on the held-out trip group.
7. Calculate the average CV Macro F1 across the three folds.
8. Fit the model on all eligible labelled training rows.
9. Save the trained model as a `.joblib` artifact.
10. Run predictions on the ten student-development trips, which were not used for fitting.

Recorded final training facts:
- Original training pool: 80 trips.
- One trip had no eligible labels, leaving **79 effective training trips**.
- Labelled candidate rows used for fitting: **800,911**.
- Ten student-development trips were kept outside model fitting.

### What does `fit()` do?

During training, the model receives:
- `X`: rows of four feature values.
- `y`: the correct candidate label (1 or 0).

It learns patterns connecting the features to the label. For example, it can learn that a candidate with a small GPS distance and a compatible heading is more likely to be correct.

### What does `predict_proba()` do?

For each candidate row, the trained classifier returns a probability-like score for class 1 (candidate is correct). The project uses the class-1 score to rank the candidate roads for each observation.

The score is not a GPS coordinate and it is not itself a final route. It is evidence used by the next stage.

## 10. How are the results calculated?

### A. Candidate Macro F1

The classifier also treats each candidate row as a binary classification:
- Positive class: candidate matches the provisional reference.
- Negative class: candidate does not match.

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Macro F1 calculates F1 separately for class 0 and class 1, then takes their average. This gives both classes equal weight, rather than letting the more common class dominate.

### B. Observation Top-1 Accuracy

This is closer to the actual map-matching task:
1. Group candidate roads belonging to the same GPS observation.
2. Select the candidate with the highest class-1 probability.
3. Compare its road-arc ID with the provisional reference road-arc ID.
4. Count the fraction of eligible observations where they match.

Top-1 accuracy = correct highest-scored observations ÷ eligible reference observations.

The final untuned Histogram Gradient Boosting classifier achieved **89.13%** on the ten student-development trips.

### C. Candidate log loss

Log loss measures how well the predicted probabilities match the labels. It penalises confident wrong predictions more heavily. **Lower is better.** The untuned Histogram Gradient Boosting result was 0.1632 in the final comparison.

### D. Alignment coverage and candidate misses

Only a portion of the GPS observations have reliable aligned references. In the final student-development results:
- Eligible observations: **11,177**
- Alignment coverage: **58.34%**
- Candidate miss rate: approximately **0.054%**

The coverage figure means only 58.34% of observations were eligible for this reference-based evaluation. It does not mean the model only predicted 58.34% of points. Candidate misses are retained in the Top-1 denominator and count as incorrect when no candidate is available.

## 11. Why do we use HMM and Viterbi after the classifier?

A classifier can choose the best road for each GPS point independently. But GPS points are sequential: a vehicle cannot normally jump from one distant road to another between two close timestamps.

For example, two parallel roads may both be close to a GPS point. The classifier may choose Road A for one point and Road B for the next, even though the vehicle could not realistically move between them.

The project therefore adds an **HMM-style sequence decoder using the Viterbi algorithm**.

- **Hidden state:** the actual road arc occupied by the vehicle at each observation.
- **Observation/evidence:** the classifier's candidate probability scores.
- **Emission score:** the classifier score for a candidate road, converted to a log score.
- **Transition score:** whether movement from the previous candidate road to the current one is possible and plausible on the directed road network.

The decoder:
1. Gets candidate roads and their classifier probabilities for each GPS observation.
2. Calculates directed shortest-path travel distances between candidate road positions.
3. Rejects transitions that are unreachable or exceed the configured speed-based movement limit.
4. Scores plausible transitions by how closely road-network travel distance matches the straight-line GPS displacement.
5. Uses Viterbi dynamic programming to find the highest-scoring sequence of road states.
6. Splits the sequence at missing candidates, invalid/large time gaps or impossible transitions. It does not invent a bridge over a gap.

### What is Viterbi doing?

Viterbi is a dynamic-programming algorithm. Instead of choosing the best road independently at every point, it keeps track of the best cumulative score for each possible current road, remembers the best previous road, and works backwards at the end to recover the best overall sequence.

**In simple words:** The classifier asks, “Which road looks best for this point?” HMM + Viterbi asks, “Which sequence of roads makes the most sense for the whole journey?”

## 12. Final result: classifier alone vs classifier + HMM

On the ten student-development trips, the recorded decoder comparison is:

| Method | Top-1 agreement | Direct-step continuity |
|---|---:|---:|
| Classifier alone (tuned HGB artifact) | 89.06% | 89.21% |
| **Classifier + HMM/Viterbi** | **96.31%** | **97.99%** |

The HMM improves the sequence-level result because it uses road connectivity and movement consistency, not just the independent candidate score.

- **Top-1 agreement** checks whether the predicted road arc matches the provisional reference for eligible observations.
- **Direct-step continuity** checks whether consecutive predicted arcs are the same arc or connect directly (end node of one equals start node of the next), for valid consecutive timestamp pairs.

The HMM had 16 sequence resets in the recorded evaluation. Resets are deliberate: the decoder starts a new run when it encounters missing candidates, timestamp gaps or implausible/disconnected transitions.

**Best overall system:** the project pipeline uses the Histogram Gradient Boosting classifier to score candidate roads, then HMM/Viterbi to choose a connected sequence. The HMM is not a separate classifier trained on the labels; it is a sequence-decoding stage that uses classifier scores and the road graph.

**Evaluation caution:** decoder beta was selected using three student-development trips. These scores are development agreement against provisional derived references, not untouched evaluator accuracy. The project documentation explicitly warns about this limitation.

## 13. How the complete application works

- **Python ML pipeline:** reads data, creates candidates/features, trains models and performs sequence decoding.
- **Saved model artifacts:** trained model files are stored under `artifacts/`.
- **FastAPI backend:** loads the saved model, receives a trip request, runs prediction/decoding and returns the result.
- **Web UI:** lets the user select a configured trip, request reconstruction and view GPS/classifier/HMM route layers on a map.
- **Folium map/report tools:** create map and technical-report outputs.

The saved final model is included in the project, so the normal demo launcher does not need to retrain the model.

## 14. Important points to remember for a presentation or viva

- We are solving **GPS map matching**, not general route planning.
- A GPS point may be noisy, so we first generate nearby road candidates.
- We use four features: distance, heading difference, speed and time gap.
- The best individual classifier in the final comparison is **Histogram Gradient Boosting**.
- We compared Logistic Regression, Random Forest, Decision Tree, Histogram Gradient Boosting and Gaussian Naive Bayes, then tested a tuned HGB version.
- HGB had the best recorded untuned CV Macro F1 (0.8510) and student-development classifier Top-1 (89.13%).
- Optuna tuning was tested, but it did not improve the final recorded score.
- The classifier gives a score to each candidate road; it does not directly create the whole route.
- HMM + Viterbi uses those scores plus directed road connectivity to produce a more consistent route.
- The combined decoder result was 96.31% Top-1 agreement and 97.99% direct-step continuity on the student-development set.
- Training labels are derived from ordered reference alignment and are provisional, not independent official ground truth.
- Only 58.34% of student-development observations had eligible aligned references, so evaluation coverage must be mentioned.
- The ten student-development trips were not used to fit the classifier, but they are still development data, not a reserved untouched evaluator set.

## 15. One-minute explanation you can say

“Our project, PathSnap, performs GPS map matching. GPS points are noisy and may not lie exactly on roads, so we first generate nearby road-segment candidates for every point. For each candidate, we calculate four features: distance from the GPS point, heading difference, estimated speed and time gap. We trained and compared Logistic Regression, Random Forest, Decision Tree, Histogram Gradient Boosting and Gaussian Naive Bayes using trip-grouped cross-validation. Histogram Gradient Boosting performed best among the individual classifiers, with a Macro F1 of about 0.851 and student-development Top-1 agreement of about 89.13%. The classifier scores each possible road, but independent predictions can create unrealistic jumps. So we pass its scores to an HMM decoder that uses directed road-network distances and the Viterbi algorithm to find the most likely connected road sequence. This improved the recorded student-development Top-1 agreement to 96.31% and direct-step continuity to 97.99%. Our reference labels are derived through route alignment, so we clearly treat these as provisional development results rather than claiming independent ground-truth accuracy.”

---

## Files worth opening when studying the implementation

- `README.md` — project overview, launch instructions and headline results.
- `docs/data_assumptions.md` — data format, projection and alignment assumptions.
- `docs/two_model_baseline.md` — earlier Logistic Regression/Random Forest baseline.
- `scripts/build_candidate_features.py` — candidate and feature creation.
- `src/pathsnap/candidates/generator.py` — nearby road candidate generation.
- `src/pathsnap/features/engineering.py` — four feature calculations and label attachment.
- `scripts/final_experiments.py` — final model comparison and Optuna search.
- `src/pathsnap/models/` — classifier definitions.
- `src/pathsnap/decoding/viterbi.py` — HMM/Viterbi road-sequence decoder.
- `results/final/metrics.json` — recorded classifier metrics.
- `results/final/selection.json` — selected model and tuning details.
- `results/final/decoder_summary.json` — classifier vs HMM comparison.

*Study tip: First understand Sections 3–5 (pipeline, candidates and features), then Sections 6–10 (labels, training and metrics), and finally Sections 11–12 (HMM/Viterbi and final results).*
