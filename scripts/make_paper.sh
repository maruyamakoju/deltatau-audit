#!/usr/bin/env bash
# ============================================================================
# make_paper.sh — Regenerate all paper materials from existing runs
#
# Usage:
#   bash scripts/make_paper.sh            # Figures + statistics only (fast)
#   bash scripts/make_paper.sh --ablation # + value ablation (multi-seed, ~5min)
#   bash scripts/make_paper.sh --full     # + full training from scratch (~hours)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="runs/speed_gen_hidden_5seed"

echo "============================================"
echo "  Learning Internal Time — Paper Generation"
echo "============================================"

# ── Tier A: Figures + Statistics (from existing runs) ──
echo ""
echo "[Tier A] Generating figures and statistics..."
python generate_paper.py
echo "[Tier A] Done."

# ── Tier B: Value ablation (multi-seed, no retraining) ──
if [[ "${1:-}" == "--ablation" || "${1:-}" == "--full" ]]; then
    echo ""
    echo "[Tier B] Running multi-seed value ablation..."
    python run_ablation.py \
        --results-dir "$RESULTS_DIR" \
        --speed-hidden \
        --multi-seed \
        --n-seeds 5 \
        --n-episodes 50
    # Regenerate figures with updated ablation data
    python generate_paper.py
    echo "[Tier B] Done."
fi

# ── Tier C: Full training from scratch ──
if [[ "${1:-}" == "--full" ]]; then
    echo ""
    echo "[Tier C] Full training (this takes a long time)..."
    echo "  Training speed_gen_hidden_5seed..."
    python run_speed_generalization.py \
        --output-dir runs/speed_gen_hidden_5seed_repro \
        --speed-hidden \
        --n-seeds 5

    echo "  Running ablation on new results..."
    python run_ablation.py \
        --results-dir runs/speed_gen_hidden_5seed_repro \
        --speed-hidden \
        --multi-seed \
        --n-seeds 5 \
        --n-episodes 50

    echo "[Tier C] Done. Results in runs/speed_gen_hidden_5seed_repro/"
fi

echo ""
echo "============================================"
echo "  Paper materials in: runs/paper_figures/"
echo "============================================"
echo ""
echo "Key outputs:"
echo "  fig_hero.png          — Fig 1: Main result overview"
echo "  fig_ablation.png      — Fig 2: Ablation (value causal test)"
echo "  fig_dt_tracking_detail.png  — Fig 3: Δτ tracking detail"
echo "  results_table.tex     — Table 1: Full results"
echo "  statistical_metrics.json    — All statistics"
