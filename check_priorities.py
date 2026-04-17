import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(r"C:\Users\07013\Desktop\0215agi")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.autonomous_research import ResearchJournal, FRONTIER_REGISTRY

journal_path = PROJECT_ROOT / "research_runs" / "journal.json"
journal = ResearchJournal.load(journal_path)

priorities = journal.get_frontier_priority()
print("Frontier Priorities (UCB1):")
for name, priority in sorted(priorities.items(), key=lambda x: -x[1]):
    n_runs = len(journal.frontier_scores.get(name, []))
    best = journal.best_per_frontier.get(name, {})
    best_score = best.get("score", 0.0)
    print(f"    {name:35s} | UCB={priority:.3f} | runs={n_runs:3d} | best={best_score:.4f}")
