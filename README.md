# PathSnap

PathSnap is a planned map-matching project for reconstructing vehicle routes from noisy GPS observations and a road network. The supplied trip files are in `Student folder/`: `.track` contains GPS observations, `.nodes` contains road-node coordinates, `.arcs` contains node connections, and `.route` contains reference arc sequences where provided.

## Repository status

This commit establishes the project layout only. The analysis, model training, tuning, route decoding, API, and UI workflows have not been implemented yet. No results or trained models are included. The original assignment materials and supplied dataset are preserved in their existing locations.

## Planned layout

- `data/` and `notebooks/`: data notes and exploratory analysis
- `src/pathsnap/`: reusable data, candidate, feature, model, evaluation, tuning, decoding, visualization, and API modules
- `ui/`: separate evaluator-facing interface
- `configs/` and `scripts/`: reproducible settings and planned command-line entry points
- `artifacts/` and `results/`: artifacts and results produced by actual runs
- `tests/` and `docs/`: future checks and project documentation

## Setup and execution

Python dependencies and runnable commands will be documented here once the corresponding implementation exists. `requirements.txt` is currently a placeholder; there are no project commands to install or run yet.

## Data handling

Use the supplied files in `Student folder/` in place. Do not treat GPS observation rows as direct road-network row matches. Keep the reserved evaluator-testing partition out of model selection and hyperparameter tuning, and keep ground-truth route labels server-side in the evaluator-facing system.
