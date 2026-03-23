# Pipeline Artifact Contract

The long-running submission supervisor produces a small set of state and event
artifacts that external tooling can depend on. This document fixes that
surface.

## Scope

This contract covers:

- `active_jobs.json`
- `monitor_snapshot.json`
- `supervisor_state.json`
- `supervisor_events.jsonl`

The contract is enforced by `tests/test_pipeline_artifact_contract.py`.

## `active_jobs.json`

Producer:

- `scripts/run_submission_pipeline.py::_save_active_jobs(...)`

Stable top-level fields:

- `generated_at_utc`
- `jobs`

Stable job entry fields:

- `name`
- `manifest`
- `output_root`
- `protocol`
- `no_resume`
- `out_log`
- `err_log`
- `pid`
- `started_at_utc`

## `monitor_snapshot.json`

Producer path:

- `scripts/run_submission_pipeline.py::_collect_diagnose_rows(...)`
- `scripts/run_submission_pipeline.py::_save_monitor_snapshot(...)`

Stable top-level fields:

- `updated_at_utc`
- `jobs`

Stable per-job snapshot fields:

- `timestamp_utc`
- `timestamp_s`
- `done`
- `total`
- `pct`
- `child_cpu_s_total`

## `supervisor_state.json`

Producer path:

- `scripts/run_submission_pipeline.py::_supervise(...)`
- `scripts/run_submission_pipeline.py::_save_supervisor_state(...)`

Stable top-level fields:

- `updated_at_utc`
- `jobs`

Stable per-job state fields:

- `restart_count`
- `consecutive_recoverable`
- `consecutive_signature_hits`
- `no_progress_cycles`
- `last_done`
- `last_progress_ts`
- `last_restart_ts`
- `last_reason`
- `last_diagnosis`
- `last_signature`
- `restarts_by_reason`
- `restarts_by_signature`
- `restart_times_s`

## `supervisor_events.jsonl`

Producer path:

- `scripts/run_submission_pipeline.py::_emit_supervisor_event(...)`

Each line must be one JSON object.

Stable event envelope fields:

- `time_utc`
- `cycle`
- `job`
- `event`

For `diagnosis` events, the current contract also expects:

- `diagnosis_code`
- `diagnosis_recoverable`
- `done`
- `total`
- `pid`
- `no_progress_cycles`
- `no_progress_seconds`
- `forced_recovery_reason`

## Change Policy

If these field names or file names change, update:

1. this document
2. the fixtures in `tests/golden/`
3. `tests/test_pipeline_artifact_contract.py`
