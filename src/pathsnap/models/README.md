# Models

The final workflow in scripts/final_experiments.py trains Logistic Regression, Random Forest, Decision Tree, Histogram Gradient Boosting and Gaussian Naive Bayes on the original training partition with trip-grouped cross-validation. It performs Bayesian tuning and exports artifacts/final/selected.joblib for the API. baselines.py retains the earlier LR/RF factory and shared feature-only scoring helper.
