# Claude Sonnet vs Haiku vs Qwen 7B — 64 NeurIPS Comparison

**Date:** 2026-09-18
**Dataset:** 64 similar + 64 dissimilar NeurIPS 2025 papers
**Judge:** GPT-4o-mini (same for all 3 summarizers = fair)
**Prompts:** Same as Phase 1 (prompt14)

---

## 1. Summarizer Comparison

### Similar Set

| Level | Metric | Sonnet | Haiku | Qwen 7B | Winner |
|---|---|---|---|---|---|
| L1 | entail_min | 0.531 | 0.531 | 0.469 | Sonnet |
| L1 | info_min | 0.500 | 0.500 | 0.406 | Sonnet |
| L1 | spec_faith | 0.688 | 0.781 | 0.594 | Haiku |
| L2 | entail_min | 0.562 | 0.625 | 0.500 | Haiku |
| L2 | info_min | 0.500 | 0.500 | 0.500 | Sonnet |
| L2 | spec_faith | 0.562 | 0.500 | 0.500 | Sonnet |
| L3 | entail_min | 0.500 | 0.625 | 0.500 | Haiku |
| L3 | info_min | 0.500 | 0.500 | 0.500 | Sonnet |
| L3 | spec_faith | 0.500 | 0.625 | 0.375 | Haiku |
| L4 | entail_min | 0.750 | 0.750 | 0.750 | Sonnet |
| L4 | info_min | 0.500 | 0.500 | 0.500 | Sonnet |
| L4 | spec_faith | 0.500 | 0.500 | 0.500 | Sonnet |
| L5 | entail_min | 0.000 | 0.000 | 0.000 | Sonnet |
| L5 | info_min | 0.000 | 0.000 | 0.000 | Sonnet |
| L5 | spec_faith | 0.000 | 0.000 | 0.000 | Sonnet |

### Dissimilar Set

| Level | Metric | Sonnet | Haiku | Qwen 7B | Winner |
|---|---|---|---|---|---|
| L1 | entail_min | 0.469 | 0.406 | 0.094 | Sonnet |
| L1 | info_min | 0.125 | 0.125 | 0.000 | Sonnet |
| L1 | spec_faith | 0.969 | 1.000 | 0.906 | Haiku |
| L2 | entail_min | 0.500 | 0.438 | 0.375 | Sonnet |
| L2 | info_min | 0.500 | 0.375 | 0.312 | Sonnet |
| L2 | spec_faith | 0.688 | 0.875 | 0.875 | Haiku |
| L3 | entail_min | 0.625 | 0.500 | 0.375 | Sonnet |
| L3 | info_min | 0.500 | 0.500 | 0.375 | Sonnet |
| L3 | spec_faith | 0.625 | 0.875 | 0.875 | Haiku |
| L4 | entail_min | 0.500 | 0.500 | 0.500 | Sonnet |
| L4 | info_min | 0.500 | 0.500 | 0.500 | Sonnet |
| L4 | spec_faith | 0.500 | 0.500 | 0.500 | Sonnet |
| L5 | entail_min | 0.000 | 0.000 | 0.000 | Sonnet |
| L5 | info_min | 0.000 | 0.000 | 0.000 | Sonnet |
| L5 | spec_faith | 0.000 | 0.000 | 0.000 | Sonnet |

## 2. Similarity Model Comparison (scincl vs mpnet)

| Model | Within-Sim | Within-Dis | Cross | Separation |
|---|---|---|---|---|
| mpnet | 0.7038 | 0.1450 | 0.2184 | +0.4854 |
| scincl | 0.9024 | 0.6868 | 0.7110 | +0.1914 |

**Observation:** mpnet separates similar vs dissimilar papers much better (+0.485 vs +0.191). scincl gives high similarity for all papers (0.687 for dissimilar), reducing its usefulness for blossom matching. **mpnet retained.**

## 3. Files

### Trees
- `results/UF_random_prompts_64_base_similar_{sonnet,haiku,qwen7b}_blossom.json`
- `results/UF_random_prompts_64_base_dissimilar_{sonnet,haiku,qwen7b}_blossom.json`

### Evaluations
- `results/{tag}_64_eval_by_gpt4o_mini.json` (similar)
- `results/{tag}_64_dissimilar_eval_by_gpt4o_mini.json` (dissimilar)

### Viewers (standalone HTML)
- `viewers/viewer_similar_{sonnet,haiku,qwen7b}_standalone.html`
- `viewers/viewer_dissimilar_{sonnet,haiku,qwen7b}_standalone.html`

### Similarity stats
- `results/similarity_mpnet_stats.json`
- `results/similarity_scincl_stats.json`

## 4. Conclusion

- **Sonnet and Haiku perform comparably**, both slightly ahead of Qwen 7B on entailment.
- **Haiku shows stronger specificity** (higher spec_faith) in both similar and dissimilar sets.
- **mpnet is the better similarity model** for this pipeline — clearer separation of similar vs dissimilar papers.
- **Judge caveat:** Results are from GPT-4o-mini (not Qwen 3B used in Phase 1), so absolute numbers differ from Phase 1. Relative comparison across 3 summarizers remains fair.
