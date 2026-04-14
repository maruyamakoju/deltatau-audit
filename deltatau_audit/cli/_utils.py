"""Shared audit pipeline runner used by all CLI subcommands."""

import json as json_mod
import os
import sys
import time


def _run_audit_pipeline(
    adapter,
    env_factory,
    args,
    *,
    title=None,
    extra_audit_kwargs=None,
    compare=True,
    adapter_factory=None,
    protocol_meta=None,
):
    """Shared post-setup pipeline for all audit handlers.

    Migrated to use AuditSession and class-based Auditors directly.
    """
    from deltatau_audit.auditors import RelianceAuditor, RobustnessAuditor, ReasoningAuditor
    from deltatau_audit.core.session import AuditSession
    from deltatau_audit.cli import (
        _build_seed_sweep_result,
        _emit_json,
        _handle_ci,
        _json_redirect,
        _maybe_compare,
        _print_failure_explanation,
        _print_markdown_summary,
        _resolve_seed_list,
        _resolve_workers,
        _result_experiment_manifest,
    )
    from deltatau_audit.report import generate_report
    from deltatau_audit.seed_sweep import run_seed_sweep
    from deltatau_audit.tracker import maybe_log
    from deltatau_audit.auditor import _print_summary, _convert_report_to_legacy_dict

    _json_mode = getattr(args, "output_format", "text") == "json"
    _verbose = not getattr(args, "quiet", False) and not _json_mode
    n_workers = _resolve_workers(args)

    common_kwargs = dict(
        n_episodes=args.episodes,
        gamma=getattr(args, "gamma", 0.99),
        device=getattr(args, "device", "cpu"),
        n_workers=n_workers,
        seed=getattr(args, "seed", None),
        verbose=_verbose,
    )

    seed_list = _resolve_seed_list(args)

    with _json_redirect(args):
        if seed_list:
            # Multi-seed still uses run_seed_sweep (which currently returns Dict)
            print(f"  Multi-seed protocol: {seed_list}")
            t0 = time.time()
            if adapter_factory is None:
                adapter_factory = lambda _seed: adapter
            
            seed_payload = run_seed_sweep(
                adapter_factory,
                env_factory,
                seed_list,
                keep_full_results=True,
                **common_kwargs,
            )
            result = _build_seed_sweep_result(seed_payload, args)
            elapsed = time.time() - t0
            print(f"\n  Multi-seed audit completed in {elapsed:.1f}s")
        else:
            # Direct use of AuditSession for single-seed
            session = AuditSession(adapter, env_factory, output_dir=".tmp_audit")
            
            robustness_auditor = RobustnessAuditor(
                adaptive=getattr(args, "adaptive", False),
                target_ci_width=getattr(args, "target_ci_width", 0.10),
                max_episodes=getattr(args, "max_episodes", 500),
                bootstrap_samples=getattr(args, "bootstrap_samples", 2000),
                **common_kwargs
            )
            
            reliance_auditor = RelianceAuditor(
                speeds=args.speeds,
                interventions=getattr(args, "interventions", None),
                **common_kwargs
            )
            
            reasoning_auditor = ReasoningAuditor(
                **common_kwargs
            )

            t0 = time.time()
            report = session.run_full_audit(
                robustness_auditor=robustness_auditor,
                reliance_auditor=reliance_auditor,
                reasoning_auditor=reasoning_auditor,
                scenarios=getattr(args, "robustness_scenarios", None)
            )
            elapsed = time.time() - t0
            print(f"\n  Audit completed in {elapsed:.1f}s")

            # Convert to legacy result for downstream tools (report gen, CI, etc.)
            result = _convert_report_to_legacy_dict(
                report,
                adapter,
                args.episodes,
                args.speeds,
                getattr(args, "deploy_threshold", 0.80),
                getattr(args, "stress_threshold", 0.50),
                verbose=False
            )

        if not _verbose:
            print()
            _print_summary(result["summary"])

        print()
        from deltatau_audit.schema import SCHEMA_VERSION

        result["schema_version"] = SCHEMA_VERSION
        result["manifest"] = _result_experiment_manifest(
            args,
            title=title,
            seed_list=seed_list,
            n_workers=n_workers,
            protocol_meta=protocol_meta,
        )

        generate_report(result, args.out, title=title)

        if seed_list:
            seed_path = os.path.join(args.out, "seed_sweep.json")
            with open(seed_path, "w", encoding="utf-8") as fs:
                json_mod.dump(result["seed_sweep"], fs, indent=2, default=str)
            print("  seed_sweep.json      -- Multi-seed aggregate + per-seed stats")

        if compare:
            _maybe_compare(args, args.out)

        if getattr(args, "output_format", "text") == "markdown":
            print()
            _print_markdown_summary(result, label=title or "")

        maybe_log(result, args)

        if getattr(args, "explain_fail", False):
            _print_failure_explanation(result)

    _emit_json(result, args)

    exit_code = _handle_ci(result, args.out, args)
    if args.ci:
        sys.exit(exit_code)
