# Verification

Run python -m unittest discover -s tests -v. The suite covers data readers,
model-evaluation, decoder loops/connectors, API/reference-error handling, model
reload/readiness, and launcher setup/ownership/Windows process groups. The
executed count is recorded in results/final/test_results.txt rather than fixed
here. Real all-ten-trip HTTP and browser checks have separate evidence under
results/final. Native Windows/macOS checks are configured in
.github/workflows/verify.yml; inspect the corresponding GitHub Actions results.
