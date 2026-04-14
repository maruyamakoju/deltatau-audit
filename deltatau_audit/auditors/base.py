"""Base class for all class-based Auditors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np

from deltatau_audit.protocols import Auditor
from deltatau_audit.schema import AuditReport, AuditStageResult, ReliabilityLevel


class BaseAuditor(Auditor, ABC):
    """Abstract base class for class-based auditors.

    Provides common utilities for result aggregation and CI calculation.
    """

    @abstractmethod
    def run(self, agent: Any, env_id: str, **kwargs: Any) -> AuditReport:
        """Executes the specific audit axis."""
        pass

    @staticmethod
    def determine_level(score: float, stages: List[AuditStageResult]) -> ReliabilityLevel:
        """Maps aggregate score + stage results to a ReliabilityLevel."""
        if score > 0.9 and all(s.success for s in stages):
            return ReliabilityLevel.CERTIFIED
        if score > 0.8:
            return ReliabilityLevel.ROBUST
        if score > 0.5:
            return ReliabilityLevel.DEGRADED
        return ReliabilityLevel.UNRELIABLE

    @staticmethod
    def bootstrap_ci(
        data: List[float],
        n_bootstrap: int = 2000,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Compute a bootstrap percentile confidence interval for the mean."""
        arr = np.array(data, dtype=np.float64)
        if len(arr) < 2:
            m = float(np.mean(arr))
            return m, m

        rng = np.random.default_rng(42)
        boot_means = np.array(
            [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_bootstrap)]
        )
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return lower, upper
