# Claude Sonnet vs Haiku vs Qwen 7B — 64 NeurIPS Comparison

Comparison of 3 LLM backends + 2 similarity models, on the 64-paper
NeurIPS dataset. Runs parallel to the main Phase 1 scale-up, so the
main pipeline stays untouched.

## Purpose
Sir's request (Sep 18):
1) Compare Claude Sonnet vs Claude Haiku vs Qwen 7B (summarizer).
2) Compare similarity models: scincl vs mpnet (all-mpnet-base-v2).

## Folder Structure
- `code/`      — Phase 1 pipeline copy (main.py, tree.py, summarize.py,
                 evaluate.py, simple_eval.py, blossom_matching.py)
- `input/`     — 64 paper JSONs (similar + dissimilar) + CSVs
- `results/`   — Phase 1 baseline results + evals (reference)
- `viewers/`   — Phase 1 standalone viewers + tree_data (reference)
- `scripts/`   — New comparison scripts (Claude/OpenAI/Qwen, scincl vs mpnet)
- `env_logs/`  — Environment setup logs

## Important
- This folder is INDEPENDENT of the main Phase 1 pipeline.
- Nothing in this folder affects the A100 scale-up jobs.
- Uses separate environment + separate scripts.

## Status
- [x] Phase 1 files copied as reference
- [ ] Env setup for Claude/OpenAI API
- [ ] Summarizer comparison: Claude Sonnet vs Haiku vs Qwen 7B
- [ ] Similarity comparison: scincl vs mpnet
