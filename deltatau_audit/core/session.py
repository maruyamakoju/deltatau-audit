from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

from deltatau_audit.protocols import AgentAdapter, Auditor, Fixer
from deltatau_audit.schema import AuditReport, AuditStageResult

logger = logging.getLogger("deltatau-audit")


class AuditSession:
    """Unified orchestration engine for high-stakes AI auditing.

    This class manages the lifecycle of an audit:
    1. Agent Registration & Capability Analysis.
    2. Multi-stage Stress & Adversarial Testing.
    3. Long-horizon Consistency Analysis.
    4. Automated Fix Proposal (Loop).
    5. Final Certification & Telemetry Export.
    """

    def __init__(self, agent: AgentAdapter, env_id: Union[str, Callable], output_dir: str = "audit_reports"):
        self.agent = agent
        self.env_id = env_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.datetime.now().isoformat()
        self.report: Optional[AuditReport] = None

    def run_full_audit(
        self,
        robustness_auditor: Optional[Auditor] = None,
        reliance_auditor: Optional[Auditor] = None,
        reasoning_auditor: Optional[Auditor] = None,
        **kwargs: Any,
    ) -> AuditReport:
        """Executes a multi-axis audit pipeline.

        Combines results from robustness, reliance, and reasoning auditors
        into a single, comprehensive AuditReport.
        """
        env_label = self.env_id if isinstance(self.env_id, str) else "custom_env"
        logger.info(f"🚀 Starting Multi-Axis Audit for {env_label}...")

        stages: List[AuditStageResult] = []
        
        # 1. Robustness Axis
        if robustness_auditor:
            rep = robustness_auditor.run(self.agent, self.env_id, **kwargs)
            stages.extend(rep.stages)

        # 2. Reliance Axis
        if reliance_auditor:
            rep = reliance_auditor.run(self.agent, self.env_id, **kwargs)
            stages.extend(rep.stages)

        # 3. Reasoning Axis
        if reasoning_auditor:
            rep = reasoning_auditor.run(self.agent, self.env_id, **kwargs)
            stages.extend(rep.stages)

        # Calculate Overall Reliability
        scores = [s.pass_rate for s in stages]
        reliability_score = float(np.mean(scores)) if scores else 1.0
        
        # Determine level (use logic from BaseAuditor)
        from deltatau_audit.auditors.base import BaseAuditor
        level = BaseAuditor.determine_level(reliability_score, stages)

        self.report = AuditReport(
            agent_id=str(id(self.agent)),
            timestamp=self.start_time,
            reliability_score=reliability_score,
            level=level,
            stages=stages,
            capabilities=self.agent.get_capabilities(),
            summary=f"Multi-axis audit completed with {len(stages)} stages. Level: {level.name}",
        )

        self.save_report()
        return self.report

    def _analyze_reasoning_depth(self) -> None:
        """Evaluates whether the agent's internal reasoning (pondering) is effective.

        This is a 'Long-horizon reasoning' check: Does more pondering lead to
        better temporal robustness, or is it just wasted computation?
        """
        if not self.report or not self.agent.get_capabilities().can_ponder:
            return

        # Core logic for analyzing if pondering correlates with stability
        # (This would be populated with data from Auditor's metrics)
        logger.info("Analyzing correlation between pondering depth and temporal stability...")
        # TODO: Implement MCTS/RNN-halting efficiency analysis

    def suggest_fix(self, fixer: Fixer) -> AgentAdapter:
        """Triggers the automated fix loop based on the current audit report."""
        if not self.report:
            raise ValueError("No audit report available. Run audit first.")

        logger.info(f"Attempting automated fix for vulnerabilities in {self.env_id}")
        fixed_agent = fixer.fix(self.agent, self.report)
        return fixed_agent

    def save_report(self) -> None:
        """Exports the audit results to a professional JSON/HTML artifact."""
        if not self.report:
            return

        env_name = self.env_id if isinstance(self.env_id, str) else getattr(self.env_id, "__name__", "custom_env")
        # Sanitize for windows filenames
        env_name = str(env_name).replace("<", "").replace(">", "").replace(":", "").replace(" ", "_")
        
        report_path = self.output_dir / f"audit_{env_name}_{datetime.date.today()}.json"
        with open(report_path, "w") as f:
            json.dump(self.report.to_json(), f, indent=2)

        logger.info(f"Audit report saved to {report_path}")
