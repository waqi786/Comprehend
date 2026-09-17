# Claude / OpenAI / Gemma Experiments

Legacy multi-LLM comparison trees from earlier experiments (Sep 2025).

## What's here
Comparison of 3 LLM backends across generation + evaluation roles:
- **Claude** (Anthropic API)
- **OpenAI** (GPT API)
- **Gemma** (local Qwen/Gemma vLLM)

## Contents
- `legacy_trees/` — 10 tree JSONs (all claude/openai/gemma combinations)
- `legacy_training_data/` — 4 distillation training JSONL files

## Naming
Each file `tree_data_<gen>_<eval>.json` shows:
- First tag = generation model
- Second tag = evaluation model
- E.g., `tree_data_claude_gemma.json` = Claude generates, Gemma evaluates

## Status
- `main` branch: NeurIPS 2025 Phase 1 (Qwen2.5-7B + 3B local)
- `bon-experimental` branch: Best-of-N experiments
- this branch: legacy multi-LLM comparisons
