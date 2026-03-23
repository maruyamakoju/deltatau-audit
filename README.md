# deltatau-audit

[![PyPI version](https://img.shields.io/pypi/v/deltatau-audit)](https://pypi.org/project/deltatau-audit/)
[![Temporal Safety Gate](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/safety-gate.yml/badge.svg)](https://github.com/maruyamakoju/deltatau-audit/actions/workflows/safety-gate.yml)

`deltatau-audit` audits timing robustness in reinforcement-learning agents. It focuses on timing shifts such as jitter, delay, spikes, and speed changes, and it emits contract-tested artifacts for local analysis, CI gates, benchmark runs, and submission prep.

## What is stable today

- Audit outputs: `summary.json`, `index.html`, `ci_summary.json`, and SVG badges
- Submission outputs: `suite_summary.json`, `suite_summary.md`, `bench_summary.json`, `submission_table.csv`, and `submission_table.md`
- Supervisor outputs: `active_jobs.json`, `monitor_snapshot.json`, `supervisor_state.json`, and `supervisor_events.jsonl`
- Contract-focused gate: `python scripts/check_contracts.py`

## Install

For local development:

```bash
pip install -e ".[dev]"
```

For package usage:

```bash
pip install deltatau-audit
```

Optional extras include `demo`, `sb3`, `dm_control`, `mujoco`, and `hf`.

## Quick start

Generate a small demo audit report:

```bash
python -m deltatau_audit demo cartpole --episodes 5 --out audit_report
```

This writes:

- `audit_report/baseline/summary.json`
- `audit_report/baseline/index.html`
- `audit_report/robust_wide/summary.json`
- `audit_report/robust_wide/index.html`

Generate CI gate artifacts from the demo:

```bash
python -m deltatau_audit demo cartpole --ci --episodes 5 --out ci_report
```

This adds:

- `ci_report/robust_wide/ci_summary.json`
- `ci_report/robust_wide/ci_summary.md`

## Stable contracts

The repository now documents its stable output surface explicitly:

- [Core output contract](docs/core_output_contract.md)
- [Submission artifact contract](docs/submission_artifact_contract.md)
- [Pipeline artifact contract](docs/pipeline_artifact_contract.md)

If you change those artifact formats, update the corresponding contract docs and golden tests.

## Quality gates

Run the fast contract gate:

```bash
python scripts/check_contracts.py
```

Run the full test suite:

```bash
python -m pytest -q
```

Run the strict submission readiness gate:

```bash
python scripts/prepare_submission.py --check-only --strict-check
```

## Research and submission workflows

Run the staged research suite:

```bash
python -m deltatau_audit research-full \
  --env CartPole-v1 \
  --episodes 10 \
  --speeds 1 2 5 \
  --out research_full_report \
  --fail-fast
```

Key outputs:

- `research_full_report/suite_summary.json`
- `research_full_report/suite_summary.md`
- stage directories under `deliberative/`, `ltc/`, and `bridge/`

Run a paper-grade benchmark manifest:

```bash
python -m deltatau_audit bench run \
  --manifest bench/high_rigor_10seed_manifest.yaml \
  --protocol paper
```

This writes:

- `bench_runs/.../bench_summary.json`
- `bench_runs/.../submission_table.csv`
- `bench_runs/.../submission_table.md`

For the longer operational path, use:

```bash
python scripts/run_submission_pipeline.py --mode autopilot --preflight --auto-recover
```

See [docs/submission_checklist.md](docs/submission_checklist.md) for the current paper-prep and operations checklist.

## Project status

- Code quality: the artifact boundaries above are contract-tested and wired into CI/release gates.
- Research status: the paper-grade experimental program is still `PREPARING`.
- Current strongest claim: the repository can audit timing fragility, generate reproducible artifacts, and support repair/benchmark workflows; not all target benchmark claims are final yet.

## Citation

```bibtex
@software{deltatau_audit2026,
  author = {maruyamakoju},
  title = {deltatau-audit},
  version = {0.8.0},
  year = {2026}
}
```
