# COMPREHEND — NeurIPS 2025 Similar vs Dissimilar Hierarchy

Qwen2.5 local vLLM pipeline. Zero API cost.

## Experiment
- **Dataset**: 5286 NeurIPS 2025 papers (OpenReview, after-rebuttal scores)
- **Similar set**: 64 papers (densest KMeans cluster)
- **Dissimilar set**: 64 papers (farthest-point sampling)

## Models
- Summary: Qwen2.5-7B-Instruct (vLLM)
- Judge: Qwen2.5-3B-Instruct (vLLM, different model — no self-bias)

## Phases
- **Phase 1**: Baseline (no Best-of-N)
- **Phase 2**: Best-of-N (3 candidates per pair, judge picks highest combined)

## Folder Structure
code/ — pipeline source (main.py, tree.py, summarize.py, evaluate.py)
trees/ — 4 built hierarchy JSONs
evals/ — 4 evaluation JSONs
viewers/ — 4 viewer-format DAG JSONs + viewer.html
visualizations/ — avg/variance plots

text

## Results (L0 = leaves, L5 = near root)

| Level | Similar P1 | Similar P2 | Dissim P1 | Dissim P2 |
|---|---|---|---|---|
| L0 entail | 0.755 | 0.763 | 0.405 | 0.455 |
| L5 entail | 0.850 | 0.800 | 0.850 | 0.800 |

Best-of-N improves L0 quality. Similar papers merge naturally (high entailment); dissimilar papers resist (low entailment); both converge at root.

## How to view trees
Open `viewers/viewer.html` in browser and load any `tree_data_*.json`.

Generated with `python main.py view` per README.
