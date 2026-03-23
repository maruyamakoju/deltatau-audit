# Core Output Contract

This project has many moving parts, but the externally consumed outputs are a
small, stable surface. That surface is the release contract.

## Scope

The contract covered here is:

- `generate_report(...)` report directory outputs
- `summary.json` machine-readable audit artifact
- `generate_badges(...)` SVG badge outputs
- `write_ci_summary(...)` / `generate_ci_summary(...)` CI gate outputs

The contract is enforced by golden tests in
`tests/test_output_contract.py`.

## `generate_report(...)`

Producer:

- `deltatau_audit.report.generate_report(audit_result, output_dir, title=...)`

Always writes:

- `summary.json`
- `index.html`
- `robustness_bars.png`

Conditionally writes, when reliance plots are applicable:

- `return_vs_speed.png`
- `reliance_rmse.png`
- `reliance_bars.png`
- `quadrant.png`

Normalization rules before serialization:

- `schema_version` is backfilled to the bundled schema version
- `manifest` is backfilled with a minimal valid manifest
- `diagnosis` is synthesized when absent and merged when partial
- `_version` and `_timestamp` are attached at report time

## `summary.json`

`summary.json` is the machine-readable audit artifact consumed by report diff,
badge generation, CI summaries, bench tooling, and submission scripts.

Stable top-level keys:

- `schema_version`
- `_version`
- `_timestamp`
- `speeds`
- `n_episodes`
- `supports_intervention`
- `reliance`
- `robustness`
- `sensitivity`
- `summary`
- `diagnosis`
- `manifest`

Notes:

- `_version`, `_timestamp`, and `manifest.created_at` are dynamic metadata
- `manifest.runtime`, `manifest.git`, and `manifest.dependencies` are allowed
  to vary by environment, but their container shape is part of the contract
- the bundled JSON Schema is the source of truth for required fields

## Badge Outputs

Producer:

- `deltatau_audit.badge.generate_badges(summary_json, output_dir, prefix="badge")`

Writes exactly four SVG files:

- `{prefix}-deployment.svg`
- `{prefix}-stress.svg`
- `{prefix}-reliance.svg`
- `{prefix}-status.svg`

Each file must contain a standalone SVG document.

## CI Summary Outputs

Producers:

- `deltatau_audit.ci.write_ci_summary(summary, robustness, output_dir, ...)`
- `deltatau_audit.ci.generate_ci_summary(audit_result, out_dir=..., ...)`

Always writes:

- `ci_summary.json`
- `ci_summary.md`

Stable JSON fields:

- `status`
- `exit_code`
- `deployment_score`
- `deployment_rating`
- `stress_score`
- `stress_rating`
- `thresholds`
- `gate_mode`
- `gate_scores`
- `deploy_pass`
- `stress_pass`
- `_version`
- `_timestamp`

Optional fields may appear when robustness detail is available, such as:

- `deployment_worst`
- `stress_worst`
- `significant_drop_count`
- `significant_change_count`
- `scenario_effect_sizes`

## Change Policy

If any of the above file names, required fields, or normalization guarantees
change, update:

1. this document
2. the golden fixtures in `tests/golden/`
3. the contract tests in `tests/test_output_contract.py`
