# BoN Experimental Branch

Parked Best-of-N related work — Sep 2026.

## Why parked (Sir's feedback)
Best-of-N improves informativeness but drops specificity. Objective 
`min(entail) + min(info)` pushes the model toward general summaries.

## Fixes attempted
1. Anchor extraction (methods, datasets, metrics) — rejected, doesn't 
   generalize to theory papers
2. Outsider penalty — kept as future direction, harder to implement

## Contents
- run_v3.sh, run_all_v2.sh — BoN experiment scripts
- code/bon_experimental/summarize_with_v3_patch.py
- viewers/bon_experimental/*.html — BoN standalone viewers
- results/bon_experimental/*.json — BoN results + evals
- logs/bon_experimental/*.log — Slurm + vLLM logs

## Status
- main branch: Phase 1 only (scale-up)
- this branch: experimental, do not merge
