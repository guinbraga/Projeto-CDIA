# Superseded duplicate reports

The reports in this directory predate the content index and the global split
registry. They are retained as historical evidence only.

- They are not consumed by the DVC pipeline or by training.
- The manual cross-dataset quarantine path they describe has been retired.
- Counts, file lists and thresholds here may not match current data.
- `scripts/remove_exact_duplicates.py` is kept only to reproduce these reports.

The current build is defined by `dvc.yaml` and described in the repository
`README.md`. Use the content index, the global split registry and the isolation
gate as the source of truth.
