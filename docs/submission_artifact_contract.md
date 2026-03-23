# Submission Artifact Contract

The submission path depends on a smaller set of stable artifacts than the full
codebase suggests. This document fixes the outputs consumed by readiness
checks, dashboards, and paper-prep scripts.

## Scope

This contract covers:

- `suite_summary.json`
- `suite_summary.md`
- `bench_summary.json`
- `submission_table.csv`
- `submission_table.md`

The contract is enforced by golden tests in
`tests/test_submission_artifact_contract.py`.

## Research Suite Outputs

Producer:

- `deltatau_audit.research_suite.run_research_suite(...)`

Always writes:

- `suite_summary.json`
- `suite_summary.md`

May also write:

- `dashboard.html`

Stable `suite_summary.json` fields:

- `generated_at_utc`
- `config`
- `stages`
- `recommendations`

Stable stage entry fields:

- `name`
- `status`
- `reason`
- `deployment_score`
- `stress_score`
- `output_dir`
- `duration_sec`
- `traceback_text`

The markdown summary must include:

- run metadata block
- per-stage status table
- recommendations section

## Benchmark / Submission Table Outputs

Producer path:

- `deltatau_audit.bench.run_manifest(...)`
- `deltatau_audit.bench.write_submission_tables_for_summary(...)`

Always writes:

- `bench_summary.json`
- `submission_table.csv`
- `submission_table.md`

Stable `bench_summary.json` fields:

- `status`
- `started_at_utc`
- `finished_at_utc`
- `duration_sec`
- `output_root`
- `counts`
- `jobs`
- `artifacts`

Stable `artifacts` fields:

- `submission_csv`
- `submission_md`
- `submission_rows`

Stable flattened submission columns:

- `job_id`
- `status`
- `command`
- `env`
- `algo`
- `seed`
- `variant`
- `protocol`
- `deployment_score`
- `deployment_rating`
- `stress_score`
- `stress_rating`
- `stress_worst_scenario`
- `stress_worst_return_ratio`
- `stress_worst_ci_lower`
- `stress_ci_gate_pass`
- `quadrant`
- `diagnosis_pattern`
- `summary_path`

`submission_table.csv` and `submission_table.md` must encode those same
columns in a fixed order.

## Change Policy

If any field names, file names, or column ordering change, update:

1. this document
2. the fixtures in `tests/golden/`
3. `tests/test_submission_artifact_contract.py`
