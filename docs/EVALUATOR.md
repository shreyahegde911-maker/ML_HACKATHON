# Reserved evaluator workflow

The final model and the prediction/scoring pipeline must be frozen before
reserved files are inspected or evaluated. This workflow never trains, refits,
tunes, selects models or adjusts predictions. Development and reserved evidence
have separate locations.

Before the freeze, finish development checks and commit implementation changes.
Then, from the repository root:

```sh
python scripts/evaluate_reserved.py freeze
python scripts/evaluate_reserved.py inspect --root evaluator_test
```

The inspector reports the actual trip root and IDs, validates observations,
network IDs and optional references, and checks all original file hashes.
Do not assume ten trips. Check any additional supplied scoring documents before
running the evaluation. If no official scorer is supplied, scores are explicitly
agreement with the already frozen provisional ordered-reference alignment.

Both launchers accept the actual parent folder reported by inspection:

```sh
bash start.sh --evaluator-root "path/to/trip-parent" --api-only
```

Windows CMD:

```bat
start.bat --evaluator-root "path\to\trip-parent" --api-only
```

The launcher resolves this path to an absolute PATHSNAP_EVALUATOR_ROOT for the
backend. An optional ignored `.pathsnap-local.json` can contain an
`evaluator_root` value to select the local dataset on normal starts. Never
commit that file or machine-specific paths. Environment variables override it.
Use `PATHSNAP_LOCAL_CONFIG` set to an empty string to disable the local profile
for development checks.

On Laptop 2 use `bash start.sh --ui-only` or `start.bat --ui-only`, open
http://localhost:8001 and set Connection settings to Laptop 1's reachable API
address. Reserved mode is visibly labelled and uses the offline network map;
external street tiles are disabled so reserved locations are not sent to a
map provider. `/trips` reports the actual catalog; `/system` reports pipeline
and model fingerprints without references or private filesystem paths.

Run the final checks with the reserved API active:

```sh
python scripts/evaluate_reserved.py run
node scripts/verify_evaluator_ui.cjs
python scripts/build_evaluator_report.py
```

Use the project's `.venv` Python. Browser checks require the Playwright package
listed in package.json and an installed browser; PATHSNAP_BROWSER_PATH can select
Chrome. The browser checker blocks external requests and verifies every catalog
entry, actual inference, displayed counts, deterministic repeated predictions
and exports. A real second-laptop/LAN demonstration is still an event check.

Evidence is local only under `results/evaluator/`: pre-evaluation freeze,
inspection and original hashes, per-trip/aggregate metrics, prediction-only
artifacts and browser screenshots. The private consolidated report is written
to `output/private/`. These paths and `evaluator_test/` are ignored by Git.
No reference labels, reference geometry or derived alignment tables are exposed
by API responses, UI assets or downloads. Missing/invalid references retain
valid predictions and clearly suppress accuracy. Generic runtime corrections
after freezing must be documented; model/configuration changes require a new
experiment and must not be motivated by reserved scores.
