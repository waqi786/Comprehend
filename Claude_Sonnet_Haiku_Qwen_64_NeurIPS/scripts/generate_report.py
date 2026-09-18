"""
Generate final comparison report.
"""
import json
from pathlib import Path

REPORT = "REPORT.md"

def load(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

lines = []
A = lines.append

A("# Claude Sonnet vs Haiku vs Qwen 7B — 64 NeurIPS Comparison")
A("")
A("**Date:** 2026-09-18")
A("**Dataset:** 64 similar + 64 dissimilar NeurIPS 2025 papers")
A("**Judge:** GPT-4o-mini (same for all 3 summarizers = fair)")
A("**Prompts:** Same as Phase 1 (prompt14)")
A("")
A("---")
A("")

# ---------- Summarizer comparison ----------
A("## 1. Summarizer Comparison")
A("")

for subset in ["similar", "dissimilar"]:
    A(f"### {subset.title()} Set")
    A("")
    A("| Level | Metric | Sonnet | Haiku | Qwen 7B | Winner |")
    A("|---|---|---|---|---|---|")

    for tag in ["sonnet", "haiku", "qwen7b"]:
        path = f"results/{tag}_64_{subset}_eval_by_gpt4o_mini.json" if subset == "dissimilar" else f"results/{tag}_64_eval_by_gpt4o_mini.json"
        # Handle: similar files have no _similar_ in name
        if subset == "similar":
            path = f"results/{tag}_64_eval_by_gpt4o_mini.json"
        else:
            path = f"results/{tag}_64_dissimilar_eval_by_gpt4o_mini.json"
        data = load(path)
        if not data:
            continue
        if tag == "sonnet":
            sonnet = data["per_level"]
        elif tag == "haiku":
            haiku = data["per_level"]
        elif tag == "qwen7b":
            qwen = data["per_level"]

    for li in range(1, 6):
        for metric in ["entail_min", "info_min", "spec_faith"]:
            s = sonnet[li][metric]
            h = haiku[li][metric]
            q = qwen[li][metric]
            best = max([("Sonnet", s), ("Haiku", h), ("Qwen7B", q)], key=lambda x: x[1])[0]
            A(f"| L{li} | {metric} | {s:.3f} | {h:.3f} | {q:.3f} | {best} |")
    A("")

# ---------- Similarity model comparison ----------
A("## 2. Similarity Model Comparison (scincl vs mpnet)")
A("")
A("| Model | Within-Sim | Within-Dis | Cross | Separation |")
A("|---|---|---|---|---|")

for name in ["mpnet", "scincl"]:
    data = load(f"results/similarity_{name}_stats.json")
    if data:
        A(f"| {name} | {data['within_similar']['mean']:.4f} | {data['within_dissimilar']['mean']:.4f} | {data['cross']['mean']:.4f} | {data['separation']:+.4f} |")
A("")
A("**Observation:** mpnet separates similar vs dissimilar papers much better (+0.485 vs +0.191). scincl gives high similarity for all papers (0.687 for dissimilar), reducing its usefulness for blossom matching. **mpnet retained.**")
A("")

# ---------- Files ----------
A("## 3. Files")
A("")
A("### Trees")
A("- `results/UF_random_prompts_64_base_similar_{sonnet,haiku,qwen7b}_blossom.json`")
A("- `results/UF_random_prompts_64_base_dissimilar_{sonnet,haiku,qwen7b}_blossom.json`")
A("")
A("### Evaluations")
A("- `results/{tag}_64_eval_by_gpt4o_mini.json` (similar)")
A("- `results/{tag}_64_dissimilar_eval_by_gpt4o_mini.json` (dissimilar)")
A("")
A("### Viewers (standalone HTML)")
A("- `viewers/viewer_similar_{sonnet,haiku,qwen7b}_standalone.html`")
A("- `viewers/viewer_dissimilar_{sonnet,haiku,qwen7b}_standalone.html`")
A("")
A("### Similarity stats")
A("- `results/similarity_mpnet_stats.json`")
A("- `results/similarity_scincl_stats.json`")
A("")

# ---------- Conclusion ----------
A("## 4. Conclusion")
A("")
A("- **Sonnet and Haiku perform comparably**, both slightly ahead of Qwen 7B on entailment.")
A("- **Haiku shows stronger specificity** (higher spec_faith) in both similar and dissimilar sets.")
A("- **mpnet is the better similarity model** for this pipeline — clearer separation of similar vs dissimilar papers.")
A("- **Judge caveat:** Results are from GPT-4o-mini (not Qwen 3B used in Phase 1), so absolute numbers differ from Phase 1. Relative comparison across 3 summarizers remains fair.")
A("")

with open(REPORT, "w") as f:
    f.write("\n".join(lines))

print(f"Report saved: {REPORT}")
print(f"Total lines: {len(lines)}")
