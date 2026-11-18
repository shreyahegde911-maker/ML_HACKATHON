# Runnable workflows

See docs/FINAL_PRESENTATION.md for exact commands. inspect_data.py prepares validated inputs and provisional labels; build_candidate_features.py uses the strict reader and joins separate references; final_experiments.py trains five models and performs TPE search; evaluate_decoder.py tunes and compares the graph HMM; serve.py runs FastAPI; verify_api.py checks all ten configured trips; verify_browser.cjs checks the separate UI process; build_report.py builds the consolidated PDF. Earlier baseline scripts remain for provenance.
