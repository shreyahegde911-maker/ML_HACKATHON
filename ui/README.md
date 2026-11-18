# Evaluator UI

Serve with `python scripts/serve_ui.py --port 8001`. Open Connection settings
to use Laptop 1's API address, select a trip and reconstruct.

The minimal light workspace shows a real OpenStreetMap street map, GPS points,
classifier/HMM overlays, start/finish markers and evaluation metrics. Street
tiles need internet; Network view uses the local projected graph and is the
automatic fallback when tiles fail. Leaflet 1.9.4 and its license are bundled
in vendor/leaflet; copy the entire ui folder for a separate UI laptop.

Layer controls, zoom/fit, prediction export and collapsed evaluation details
are retained. Repeated explanations were removed. No reference labels or model
weights are present. Browser evidence includes desktop/mobile, two live trips,
map centering, controls, export, and simulated connection/tile/reference failures.
