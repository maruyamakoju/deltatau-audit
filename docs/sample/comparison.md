# Audit Comparison

| | Before | After | Change |
|---|---|---|---|
| Reliance | N/A | N/A | - |
| **Deployment** | 0.02 (FAIL) | 1.00 (PASS) | +0.98 FAIL -> PASS |
| **Stress** | -0.12 (FAIL) | 0.38 (FAIL) | +0.50 FAIL |
| Quadrant | deployment_fragile | deployment_ready | deployment_fragile -> deployment_ready |

## Per-Scenario Detail

| Scenario | Category | Before | After | Change |
|---|---|---|---|---|
| delay | Deployment | 2% (RMSE 3.92x) | 148% (RMSE 0.80x) | +146pp |
| jitter | Deployment | 28% (RMSE 2.37x) | 121% (RMSE 0.52x) | +93pp |
| spike | Deployment | 100% (RMSE 1.08x) | 113% (RMSE 0.92x) | +13pp |
| speed_5x | Stress | -12% (RMSE 4.90x) | 38% (RMSE 1.01x) | +50pp |

## Worst Scenarios

| Category | Before | After |
|---|---|---|
| Deployment | delay (drop 98.1%) | none (no drop) |
| Stress | speed_5x (drop 111.9%) | speed_5x (drop 61.6%) |
