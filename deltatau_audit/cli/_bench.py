"""Bench subcommand handlers: bench run, bench table."""
import sys


def _run_bench(args):
    """Run benchmark jobs from a matrix manifest."""
    from deltatau_audit.bench import run_manifest

    summary = run_manifest(
        args.manifest,
        output_root=args.out,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
        protocol_name=args.protocol,
        allow_protocol_override=args.allow_protocol_override,
    )
    counts = summary.get("counts", {})
    passed = int(counts.get("passed", 0))
    failed = int(counts.get("failed", 0))
    skipped = int(counts.get("skipped", 0))
    artifacts = summary.get("artifacts", {})
    print("Benchmark run complete")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Status:  {summary.get('status')}")
    print(f"  Summary: {summary.get('output_root')}/bench_summary.json")
    if isinstance(artifacts, dict):
        if artifacts.get("submission_csv"):
            print(f"  Submission CSV: {artifacts['submission_csv']}")
        if artifacts.get("submission_md"):
            print(f"  Submission MD:  {artifacts['submission_md']}")
    if failed > 0:
        sys.exit(1)


def _run_bench_table(args):
    """Regenerate submission tables from an existing bench summary."""
    from deltatau_audit.bench import load_run_summary, write_submission_tables_for_summary

    summary = load_run_summary(args.summary)
    artifacts = write_submission_tables_for_summary(summary, output_root=args.out)
    print("Submission tables generated")
    print(f"  CSV: {artifacts.get('submission_csv')}")
    print(f"  MD:  {artifacts.get('submission_md')}")


