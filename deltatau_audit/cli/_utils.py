"""Shared audit pipeline runner used by all CLI subcommands."""
import json as json_mod
import os
import sys
import time

def _run_audit_pipeline(adapter, env_factory, args, *, title=None,
                        extra_audit_kwargs=None, compare=True,
                        adapter_factory=None, protocol_meta=None):
    """Shared post-setup pipeline for all audit handlers.

    Calls run_full_audit(), prints timing/summary, generates reports,
    handles --format markdown, logs to trackers, emits JSON, and runs CI.

    Parameters
    ----------
    adapter       : the loaded agent adapter
    env_factory   : callable that returns a fresh env
    args          : parsed argparse namespace
    title         : report/label title string (default: None)
    extra_audit_kwargs : dict of extra kwargs forwarded to run_full_audit()
                         (e.g. interventions, sensitivity_episodes)
    compare       : whether to call _maybe_compare() (default: True)
    """
    from deltatau_audit.cli import (
        _resolve_workers, _resolve_seed_list, _json_redirect,
        _print_markdown_summary, _emit_json, _handle_ci,
        _maybe_compare, _result_experiment_manifest,
        _print_failure_explanation, _build_seed_sweep_result,
    )
    from deltatau_audit.auditor import _print_summary, run_full_audit
    from deltatau_audit.report import generate_report
    from deltatau_audit.seed_sweep import run_seed_sweep
    from deltatau_audit.tracker import maybe_log

    _json_mode = getattr(args, "output_format", "text") == "json"
    _verbose = not getattr(args, "quiet", False) and not _json_mode
    n_workers = _resolve_workers(args)

    audit_kwargs = dict(
        speeds=args.speeds,
        n_episodes=args.episodes,
        device=getattr(args, "device", "cpu"),
        seed=getattr(args, "seed", None),
        n_workers=n_workers,
        verbose=_verbose,
        deploy_threshold=getattr(args, "deploy_threshold", 0.80),
        stress_threshold=getattr(args, "stress_threshold", 0.50),
        adaptive=getattr(args, "adaptive", False),
        target_ci_width=getattr(args, "target_ci_width", 0.10),
        max_episodes=getattr(args, "max_episodes", 500),
        bootstrap_samples=getattr(args, "bootstrap_samples", 2000),
    )
    if extra_audit_kwargs:
        audit_kwargs.update(extra_audit_kwargs)

    seed_list = _resolve_seed_list(args)

    with _json_redirect(args):
        if seed_list:
            run_kwargs = dict(audit_kwargs)
            run_kwargs.pop("seed", None)
            if adapter_factory is None:
                adapter_factory = lambda _seed: adapter

            print(f"  Multi-seed protocol: {seed_list}")
            t0 = time.time()
            seed_payload = run_seed_sweep(
                adapter_factory,
                env_factory,
                seed_list,
                keep_full_results=True,
                **run_kwargs,
            )
            result = _build_seed_sweep_result(seed_payload, args)
            elapsed = time.time() - t0
            print(f"\n  Multi-seed audit completed in {elapsed:.1f}s")
        else:
            t0 = time.time()
            result = run_full_audit(adapter, env_factory, **audit_kwargs)
            elapsed = time.time() - t0
            print(f"\n  Audit completed in {elapsed:.1f}s")

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


