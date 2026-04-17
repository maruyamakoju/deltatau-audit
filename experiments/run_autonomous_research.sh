#!/bin/bash
# Autonomous Research Runner — 24/7 frontier exploration
#
# Usage:
#   bash experiments/run_autonomous_research.sh          # infinite mode
#   bash experiments/run_autonomous_research.sh 50       # 50 cycles
#   bash experiments/run_autonomous_research.sh 0 certified_mcts  # specific frontier
#
# The orchestrator will:
#   1. Select the most promising frontier (UCB1 bandit)
#   2. Mutate hyperparameters toward the frontier
#   3. Run the experiment
#   4. Analyze results and update the journal
#   5. Generate a dashboard
#   6. Loop forever

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CYCLES="${1:-0}"
FRONTIER="${2:-}"
OUT_DIR="${PROJECT_ROOT}/research_runs"
JOURNAL="${OUT_DIR}/journal.json"
DASHBOARD="${OUT_DIR}/dashboard.html"
STATUS="${OUT_DIR}/status.json"
STOP_FILE="${OUT_DIR}/STOP"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "  AUTONOMOUS RESEARCH — 24/7 MODE"
echo "  Output: ${OUT_DIR}"
echo "  Cycles: ${CYCLES:-infinite}"
echo "  Frontier: ${FRONTIER:-all (UCB1 selection)}"
echo "  Stop file: ${STOP_FILE}"
echo "=========================================="

# Create output directory
mkdir -p "$OUT_DIR"

# Build Python command
CMD="python experiments/autonomous_research.py --out ${OUT_DIR} --journal ${JOURNAL} --dashboard ${DASHBOARD} --status ${STATUS} --stop-file ${STOP_FILE} --cycles ${CYCLES}"
if [ -n "$FRONTIER" ]; then
    CMD="${CMD} --frontier ${FRONTIER}"
fi

# Run with auto-restart on crash
while true; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting research cycle..."

    if $CMD; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Research completed normally."
        break
    else
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Research crashed. Generating dashboard and restarting in 10s..."
        python experiments/frontiers/research_dashboard.py --journal "$JOURNAL" --output "${DASHBOARD}" 2>/dev/null || true
        sleep 10
    fi
done

# Final dashboard
python experiments/frontiers/research_dashboard.py --journal "$JOURNAL" --output "${DASHBOARD}" 2>/dev/null || true
echo "Dashboard: ${DASHBOARD}"
echo "Status: ${STATUS}"
