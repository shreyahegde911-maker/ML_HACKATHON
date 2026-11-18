# Recorded Part 1 inspection

Actual run on the supplied 80 training and 10 student-development trips:

| Measurement | Value |
| --- | ---: |
| Trips inspected | 90 |
| GPS observations | 227,577 |
| Reference-route entries | 34,813 |
| Eligible provisional aligned observations | 136,278 |
| Flagged observations | 91,299 |
| Directed reference endpoint discontinuities | 14 |
| Nonpositive timestamp steps | 0 |
| Largest timestamp gap, seconds | 153,064 |
| Raw files verified unchanged | 360 |

Training data contributes 125,101 eligible provisional labels out of 208,417
observations. Student-development contributes 11,177 out of 19,160. These are
alignment coverage counts, not classifier accuracy or verified ground truth.

`summary.json` records arguments, settings and measured runtime. `inventory.csv`
contains per-trip sizes and counts. `quality.json` records timestamp anomalies,
reference-break positions, CRS metadata and combined alignment statuses.
`source_checksums.json` records raw SHA-256 hashes. `split_manifest.json` is the
shared seed-42 partition for the model teammates.

The separate one-trip demonstration is under `results/part1_example/`.
Generated label tables are gitignored; reproduce them using the command in the
root README. See `docs/data_assumptions.md` before joining labels to candidates.
