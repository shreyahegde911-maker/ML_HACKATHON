# Deliverable audit

Software/artifact checks: **passed**.

The assignment's algorithm list is examples, not a fixed list. All five implemented algorithms are explicitly listed in Models and Training Guidelines.pdf.

| Deliverable | Status |
| --- | --- |
| Data readers and preprocessing | present |
| EDA and geographical visualisation | present |
| Feature generation | present |
| Five-model comparison | present |
| Bayesian optimisation | present |
| HMM / Viterbi and measured comparison | present |
| API and evaluator UI | present |
| Deployment and source instructions | present |
| Consolidated PDF report | present |

All five saved models loaded, passed fitted-state checks and predicted finite probabilities. The consolidated report has 15 pages. Ten complete Bayesian trials, disjoint trip folds, all-ten-trip HTTP evidence and zero-browser-error evidence were verified.

## Event-specific items still pending

- Reserved evaluator trip files were not supplied; final evaluator performance remains untested.
- A second physical laptop/LAN and its receiving-terminal screenshot must be demonstrated at the event.
- The team must review and submit its work, and be ready to explain/disclose AI assistance as required by the event policy.

## Repeat the audit

```sh
python -m unittest discover -s tests -v
python scripts/audit_deliverables.py
```

Cross-platform checks are configured in .github/workflows/verify.yml. A workflow configuration alone is not evidence of a passing native Windows run; inspect the corresponding GitHub Actions jobs.
