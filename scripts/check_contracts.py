"""Run the stable contract test suites only."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_TESTS = [
    "tests/test_output_contract.py",
    "tests/test_submission_artifact_contract.py",
    "tests/test_pipeline_artifact_contract.py",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run contract-focused pytest suites.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use normal pytest output instead of -q.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to pytest.",
    )
    args = parser.parse_args(argv)

    cmd = [sys.executable, "-m", "pytest"]
    if not args.verbose:
        cmd.append("-q")
    cmd.extend(CONTRACT_TESTS)
    cmd.extend(args.pytest_args)

    print("contract suites:")
    for path in CONTRACT_TESTS:
        print(f"  {path}")

    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
