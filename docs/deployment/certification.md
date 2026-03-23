# Safety Certification

The Certification System (`python -m deltatau_audit certify`) is the industry's first automated pipeline for generating audit-ready safety documentation for RL agents.

## What is a Certificate?

A `deltatau-audit` certificate is a tamper-proof document that proves an agent has passed:
1.  **Temporal Reliance Test**: The agent understands and uses time.
2.  **Environmental Robustness**: Survives jitter, lag, and spikes.
3.  **Adversarial Defense**: Resists value-minimizing timing attacks.

## Registry & Verification

Each certificate contains a unique Registry ID (e.g., `DT-3E1982...`). This ID is a hash of the full audit logs, ensuring that the model deployed on the robot is the exact same model that passed the audit.
