# Verification

Run python -m unittest discover -s tests -v. The 34 tests include the 24 core
reader, model-evaluation, decoder and API checks, plus 10 launcher checks for
setup, conflicting ports, ownership/cleanup, UI-only mode and Windows process
groups. Real all-ten-trip HTTP verification and browser checks are separate
scripts with retained evidence under results/final. Native Windows execution
was not available during launcher verification; the batch wrapper shares the
same standard-library runner as the tested shell launcher.
