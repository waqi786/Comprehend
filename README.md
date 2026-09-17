# COMPREHEND — NeurIPS 2025 Similar vs Dissimilar Hierarchy

Qwen2.5 local vLLM pipeline. Zero API cost.

## Experiment
- **Dataset**: 5286 NeurIPS 2025 papers (OpenReview, after-rebuttal scores)
- **Similar set**: 64 papers (densest KMeans cluster)
- **Dissimilar set**: 64 papers (farthest-point sampling)

## Models
- Summary: Qwen2.5-7B-Instruct (vLLM)
- Judge: Qwen2.5-3B-Instruct (vLLM, different model — no self-bias)

## Pipeline
Phase 1 only — single summary per pair (no Best-of-N).

Note: Best-of-N experiments are parked in the `bon-experimental` branch.

## Folder Structure
- `code/` — pipeline source (main.py, tree.py, summarize.py, evaluate.py)
- `trees/` — 2 hierarchy JSONs (similar + dissimilar)
- `evals/` — 2 evaluation JSONs
- `viewers/` — 2 standalone viewer HTMLs + viewer.html + tree data
- `visualizations/` — avg/variance plots

## Results (L0 = leaves, L5 = near root)

| Level | Similar P1 | Dissim P1 |
|---|---|---|
| L0 entail | 0.755 | 0.405 |
| L5 entail | 0.850 | 0.850 |

Similar papers merge naturally (high entailment); dissimilar papers 
resist (low entailment); both converge at root.

## How to view trees
Open `viewers/viewer_similar_standalone.html` or 
`viewers/viewer_dissimilar_standalone.html` directly in a browser.

Generated with `python main.py view` per README.
