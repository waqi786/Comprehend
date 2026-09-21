# COMPREHEND — 512-Paper NeurIPS Scaling

**Branch:** `feature/512-neurips-scaling`

**Purpose:** Scalable entailment-hierarchy construction and evaluation on a random sample of 512 NeurIPS 2025 papers, using the Blossom hierarchy method.

---

## Overview

This branch extends the COMPREHEND pipeline to the 512-paper regime:

- **Three tree variants**: Base (random 512), Similar (densest cluster), Dissimilar (farthest-point sampling)
- **LLM-based evaluation** of every internal node
- **Human review integration** — rating, quality, clarity, significance, originality and their variances
- **Interactive standalone viewers** — self-contained HTML files

---

## Repository Layout

| File | Purpose |
|------|---------|
| `main_512_similar.py` | Pipeline entry for similar 512 tree |
| `main_512_dissimilar.py` | Pipeline entry for dissimilar 512 tree |
| `run_512_similar.sh` | SLURM: build + evaluate similar tree |
| `run_512_dissimilar.sh` | SLURM: build + evaluate dissimilar tree |
| `run_512_phase1.sh` | SLURM: build + evaluate base tree |
| `neurips_random_512.csv` | Random 512 sample input |
| `neurips_similar_512.csv` | Similar 512 sample input |
| `neurips_dissimilar_512.csv` | Dissimilar 512 sample input |
| `Results/` | Tree outputs and eval outputs |
| `visualisation/` | Merged tree data and standalone viewers |

---

## Tree Structure

10-level Blossom hierarchy over 1023 nodes:

| Level | Nodes |
|-------|-------|
| 0 | 512 |
| 1 | 256 |
| 2 | 128 |
| 3 | 64 |
| 4 | 32 |
| 5 | 16 |
| 6 | 8 |
| 7 | 4 |
| 8 | 2 |
| 9 | 1 |

---

## Metrics Available

Node-level (LLM): evalScore, specificity_faithfulness, specificity_informativeness, faithfulness1/2, informativeness1/2, entail1/2

Paper-level (human, aggregated upward): avg_rating, var_rating, avg_quality, var_quality, avg_clarity, var_clarity, avg_significance, var_significance, avg_originality, var_originality

External judges: wb_scores (Mistral, Llama, GPT-4, Claude), text_similarity, idea_similarity

---

## Requirements

- Python 3.10+
- CUDA GPU with ~40 GB memory (A100 MIG 3g.40gb or larger)
- vLLM for LLM serving

---

## Setup

git clone https://github.com/waqi786/Comprehend.git
cd Comprehend
git checkout feature/512-neurips-scaling

python3 -m venv comprehen_env
source comprehen_env/bin/activate
pip install -r requirements.txt

export HF_HOME=$PWD/.cache_hf
mkdir -p $HF_HOME

---

## Running

SLURM:
sbatch run_512_phase1.sh
sbatch run_512_similar.sh
sbatch run_512_dissimilar.sh

Manual:
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8001 --max-model-len 8192
python3 main_512_similar.py run --method blossom

---

## Viewing Results

Three self-contained viewers:
- visualisation/viewer_512_base_standalone.html
- visualisation/viewer_512_similar_standalone.html
- visualisation/viewer_512_dissimilar_standalone.html

Features: metric dropdown, node coloring, hover tooltip, click-to-zoom, full-text search.

---

## Data Source

Human review data from NeurIPS 2025 public review dataset. Papers with fewer than 3 reviews excluded.

---

## Notes

- This branch is self-contained; viewers embed data inline
- Original 64-paper pipeline lives on main branch
- All-NeurIPS scaling lives on feature/all-neurips-scaling branch

---

## Contact

Maintainer: prajwals
Lab: hawshiuan
Repository: https://github.com/waqi786/Comprehend
