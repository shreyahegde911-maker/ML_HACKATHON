# Part 1: data contract and provisional reference labels

## Running

Use Python 3.12 and install the pinned dependencies from `requirements.txt`.
Run commands from the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/inspect_data.py --data-root "Student folder" --partition train --trip 00000001 --align --output-dir results/part1_example
python scripts/inspect_data.py --data-root "Student folder" --partition all --align --output-dir results/part1
```

The last command generates labels for all 80 training and 10 student-development
trips. Labels are written under `data/derived/reference_labels/{partition}/{trip_id}.csv`.
These generated files are gitignored; regenerate them from the committed code.
Small inventories, summaries, quality records and source checksums are committed
under `results/part1/`. The script checks raw SHA-256 values before and after the
run and fails if an inspected source changes. No raw file is overwritten.

## Interface for Part 2

```python
import sys
sys.path.insert(0, "src")
from pathsnap.data import load_trip, load_reference_route, align_reference_route

trip = load_trip("Student folder", "train", "00000001")
observations = trip["observations"]
nodes = trip["nodes"]
arcs = trip["arcs"]
# Candidate generation uses the three tables above, without references.

reference = load_reference_route("Student folder", "train", "00000001", arcs=arcs)
labels = align_reference_route(trip, reference)
eligible_labels = labels.loc[labels.alignment_status.eq("aligned")]
```

`load_trip(data_root, partition, trip_id)` accepts numeric or zero-padded IDs and
returns a dictionary:

| Key | Columns / contents |
| --- | --- |
| observations | observation_id, longitude, latitude, timestamp, x_m, y_m |
| nodes | node_id, longitude, latitude, x_m, y_m |
| arcs | arc_id, from_node_id, to_node_id |
| crs | source/projected CRS, EPSG, units, WKT, selection method |
| validation | counts, bounds, isolated nodes, timestamp anomalies, warnings |
| trip_id, partition | identity metadata |

All IDs are the original zero-based row indices. Source rows are neither dropped
nor sorted. Arc direction follows the supplied endpoint order; reverse arcs are
not invented. Arc IDs are local to each trip and are not global model classes.
The loader never opens `.route` and works when that file is absent.

Rows must have the exact documented column count. Blank rows, invalid numeric
values, nonfinite coordinates, invalid longitude/latitude bounds, fractional IDs
and out-of-range node references fail explicitly. Nonpositive timestamps remain
in source order and are reported; alignment excludes both sides of those steps.

## Coordinates

Raw order is longitude, latitude. Each trip uses the WGS84 UTM zone selected from
the median observed longitude/latitude (north/south EPSG 326xx/327xx). Both GPS and
road nodes use exactly the same transformer with `always_xy=True`, yielding
metres. CRS selection is deterministic geometry, not fitted model preprocessing.
There is no cross-trip scaling/imputation. Projected extents exceeding 200 km
raise a recorded warning; inspect distortion before using such networks. Trip
centres outside the standard UTM latitude range are rejected. This implementation
targets the supplied local trips, not global or antimeridian trajectories.

## The reference alignment is derived supervision

The PDFs describe an ordered reference route but do not specify an official
GPS-observation-to-arc alignment or executable observation-level scoring rule.
For example, training trip 00000001 contains 1,315 GPS observations and 525 route
entries. They cannot be joined by row number. Confirm the organisers' intended
scoring/alignment procedure before treating pointwise scores as official.

The separate alignment utility is a provisional geometric method:

1. Construct the segment for each reference *occurrence*. Repeated arc IDs remain
   distinct states in their original route positions.
2. Calculate point-to-segment distances in projected metres.
3. Use dynamic programming with nondecreasing route-position transitions.
   Starting and ending partway through a route is allowed. Each observation has
   squared distance cost (distance / 25 m)^2, capped at 10,000. Forward route
   distance carries a small penalty: 0.05 * distance / (60 m/s * elapsed seconds).
   A prefix-minimum recurrence avoids a quadratic transition matrix.
4. For time gaps over 30 seconds or invalid time steps, relax the forward-distance
   penalty, retain route order, and mark both boundary observations. Do not create
   synthetic observations or require adjacent reference arcs at every GPS step.
5. Backtrace a proposed arc occurrence per original observation. Flag uncertainty
   rather than manufacturing a perfect label table.

This is reference-label preparation, not the production HMM decoder. It consumes
reference routes and must never run inside candidate generation, inference, or
the UI. Its parameters are provisional defaults, not experimentally optimised.

Output columns:

```text
trip_id, observation_id, reference_arc_id,
alignment_status, alignment_distance_m,
route_position, reference_section_id, label_source
```

Only `alignment_status == "aligned"` is eligible provisional supervision.
Other statuses can be combined using `|`:

- `ambiguous_geometry`: another distinct reference arc is within 5 m of the
  proposed arc's distance, including cases where that other arc is closer.
- `repeated_arc_occurrence`: the selected arc occurs more than once in the
  reference; conservatively exclude it despite the ordered proposed occurrence.
- `too_far`: distance exceeds 75 m.
- `zero_length_segment`: the selected arc has identical endpoint coordinates.
- `gap_boundary`: either side of a time gap over 30 seconds.
- `invalid_timestamp`: either side of a nonpositive time step.
- `route_discontinuity`: a selected arc borders a reference endpoint break, or
  consecutive observations cross reference sections. Missing connections are
  not fabricated.
- `implausible_progress`: projected route progress exceeds 60 m/s plus 25 m
  tolerance, or reverses more than 25 m, outside a large time gap.

`reference_arc_id` remains the proposed value even on flagged rows, solely for
diagnostics. Downstream code MUST filter by status before assigning labels.
Do not interpret unflagged rows as verified physical ground truth. The method
uses straight endpoint segments, a globally monotone route assumption and local
distance-based ambiguity checks; it does not estimate calibrated alignment
confidence or all alternate-sequence probabilities. Backpointer memory is
O(observations * reference entries); geometric batches are bounded at 256 rows.

## Joining candidates and labels

Join on `(trip_id, observation_id)` using string trip IDs (read CSV with
`dtype={"trip_id": str}` to retain zero padding). For eligible observations,
candidate label is the equality of candidate `arc_id` and `reference_arc_id`.
Keep all unreliable observations in a diagnostic table but leave their training
labels unset. Never use the reference route to generate, filter or insert
candidates. When an eligible reference arc is absent, report a candidate miss;
do not silently drop that observation from Top-1 accuracy. Report alignment
coverage alongside derived-label metrics and make the scoring denominator
explicit. Candidate-level binary Macro F1 and observation-level Top-1 accuracy
are different metrics. Keep the IDs themselves out of model input features.

## Shared partitions

The script writes `split_manifest.json`: 64 training trips and 16 validation trips
from the original 80, using sorted IDs and NumPy default_rng(42).permutation,
with the first 16 permuted IDs assigned to validation. The 10 student_test trips
are a separate development-evaluation set. Consumers map both train/validation
back to the raw `train` folder. No observations from one trip cross these splits.
This fixed holdout is a baseline split; it does not replace the required
group-aware cross-validation or five-model final comparison.

The readers and inspection script allow only train/student_test. No reserved
evaluator data is processed. No learned preprocessing is fitted in Part 1.
