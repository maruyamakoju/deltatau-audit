"""Frontier proposal materializer.

When a critique call returns a ``proposed_new_frontier`` payload, the
autonomous loops hand it here. This module:

1. Validates the proposal's name and shape.
2. Writes a skeleton Python file to ``experiments/frontier_proposals/``
   (NOT to the registered ``experiments/frontiers/`` directory — new
   proposals stay candidates until a human reviews and promotes them).
3. Syntax-checks the skeleton so unusable junk doesn't accumulate.
4. Records the proposal in a per-run JSONL journal for later review.

Safety boundary: the materializer never edits ``FRONTIER_REGISTRY``.
Claude/Codex can seed ideas but cannot auto-register them into the
bandit — that promotion is a human decision.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROPOSALS_DIR = PROJECT_ROOT / "experiments" / "frontier_proposals"
REGISTERED_FRONTIERS_DIR = PROJECT_ROOT / "experiments" / "frontiers"

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,39}$")

DEFAULT_SKELETON = '''"""Auto-generated frontier proposal: {name}.

Description: {description}
Hypothesis: {hypothesis}
Proposed at: {timestamp}

This is a stub. A human must review and implement before promotion to
experiments/frontiers/ and FRONTIER_REGISTRY.
"""

from __future__ import annotations
from typing import Dict, Any


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder. Replace with actual frontier experiment.

    Expected to return a dict with keys: composite, finding, metrics.
    """
    return {{
        "composite": 0.0,
        "finding": "Frontier proposal {name} not yet implemented.",
        "metrics": {{}},
    }}
'''


@dataclass
class MaterializationResult:
    accepted: bool
    reason: str
    name: Optional[str] = None
    path: Optional[Path] = None


def _is_name_available(name: str) -> bool:
    """True if no file under frontiers/ or frontier_proposals/ already uses this name."""
    candidates = [
        REGISTERED_FRONTIERS_DIR / f"{name}.py",
        PROPOSALS_DIR / f"{name}.py",
    ]
    return not any(p.exists() for p in candidates)


def _render_skeleton(proposal: Dict[str, Any]) -> str:
    user_code = proposal.get("skeleton_python")
    if isinstance(user_code, str) and user_code.strip():
        try:
            ast.parse(user_code)
            return user_code
        except SyntaxError:
            # Skeleton didn't compile — fall through to default stub and
            # preserve the proposed text as a top-of-file comment for review.
            pass
    return DEFAULT_SKELETON.format(
        name=proposal["name"],
        description=proposal.get("description", "").replace('"""', "'''"),
        hypothesis=proposal.get("hypothesis", "").replace('"""', "'''"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def materialize_proposal(
    proposal: Optional[Dict[str, Any]],
    *,
    cycle: int,
    critic_session_id: Optional[str],
    out_root: Path,
) -> MaterializationResult:
    """Turn a critique's proposed_new_frontier dict into a disk artifact.

    Returns a ``MaterializationResult`` describing whether the proposal
    was accepted (written to disk) or rejected (and why). Never raises
    on malformed input — the loop must keep running.
    """
    if not isinstance(proposal, dict):
        return MaterializationResult(accepted=False, reason="no proposal payload")

    name = str(proposal.get("name", "")).strip()
    if not name or not NAME_PATTERN.match(name):
        return MaterializationResult(
            accepted=False,
            reason=f"invalid name (must match {NAME_PATTERN.pattern})",
            name=name or None,
        )

    for required in ("description", "rationale", "hypothesis"):
        if not str(proposal.get(required, "")).strip():
            return MaterializationResult(
                accepted=False,
                reason=f"missing required field: {required}",
                name=name,
            )

    if not _is_name_available(name):
        return MaterializationResult(
            accepted=False,
            reason="name collides with existing frontier or proposal",
            name=name,
        )

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    target = PROPOSALS_DIR / f"{name}.py"

    skeleton = _render_skeleton(proposal)
    try:
        ast.parse(skeleton)
    except SyntaxError as exc:
        return MaterializationResult(
            accepted=False,
            reason=f"rendered skeleton does not compile: {exc}",
            name=name,
        )

    target.write_text(skeleton, encoding="utf-8")

    journal_path = out_root / "frontier_proposals.jsonl"
    record = {
        "cycle": cycle,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "critic_session_id": critic_session_id,
        "name": name,
        "description": proposal.get("description"),
        "rationale": proposal.get("rationale"),
        "hypothesis": proposal.get("hypothesis"),
        "estimated_novelty": proposal.get("estimated_novelty"),
        "path": str(target),
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return MaterializationResult(
        accepted=True,
        reason="written to frontier_proposals/",
        name=name,
        path=target,
    )
