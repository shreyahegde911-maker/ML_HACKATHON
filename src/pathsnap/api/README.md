# Model API

FastAPI implements /health, /trips and POST /predict. Prediction uses the saved classifier and directed graph HMM. References are read only afterwards for server-only aggregate evaluation. UI payloads exclude reference IDs and geometry. See docs/FINAL_PRESENTATION.md for deployment and verification.
