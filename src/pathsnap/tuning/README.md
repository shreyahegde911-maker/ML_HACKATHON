# Bayesian optimisation

scripts/final_experiments.py compares five models and runs ten Optuna TPE trials (three startup trials) on a fixed per-trip capped training sample with original fold memberships. The final tuned model is re-evaluated on full-data folds and refitted. Trials CSV, SQLite study, settings and measured untuned/tuned results are in results/final. Tuning did not improve full-data CV in this run.
