"""
Generate professional report (screenshot-ready).
"""
import json
from pathlib import Path

def load(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

lines = []
A = lines.append

sep  = "=" * 78
dash = "-" * 78

A(sep)
A("   CLAUDE SONNET  vs  CLAUDE HAIKU  vs  QWEN 2.5-7B")
A("   64-Paper NeurIPS 2025 Comparison Report")
A(sep)
A("")
A(f"  Dataset  : 64 similar + 64 dissimilar NeurIPS 2025 papers")
A(f"  Judge    : GPT-4o-mini (fixed for all 3 summarizers -> fair)")
A(f"  Prompts  : Same prompt14 as Phase 1 pipeline")
A(f"  Trees    : 7 levels, 127 nodes each")
A("")
A(sep)
A("   SECTION 1  --  SUMMARIZER COMPARISON  (Similar Set)")
A(sep)
A("")
A(f"  {'Level':<7}{'Metric':<16}{'Sonnet':<10}{'Haiku':<10}{'Qwen 7B':<10}{'Winner'}")
A("  " + dash)
for li in range(1, 5):
    for metric in ["entail_min", "info_min", "spec_faith"]:
        s_row = load("results/sonnet_64_eval_by_gpt4o_mini.json")["per_level"][li]
        h_row = load("results/haiku_64_eval_by_gpt4o_mini.json")["per_level"][li]
        q_row = load("results/qwen7b_64_eval_by_gpt4o_mini.json")["per_level"][li]
        s = s_row[metric]; h = h_row[metric]; q = q_row[metric]
        best = max([("Sonnet", s), ("Haiku", h), ("Qwen7B", q)], key=lambda x: x[1])[0]
        A(f"  L{li:<6}{metric:<16}{s:<10.3f}{h:<10.3f}{q:<10.3f}{best}")
A("")
A(sep)
A("   SECTION 2  --  SUMMARIZER COMPARISON  (Dissimilar Set)")
A(sep)
A("")
A(f"  {'Level':<7}{'Metric':<16}{'Sonnet':<10}{'Haiku':<10}{'Qwen 7B':<10}{'Winner'}")
A("  " + dash)
for li in range(1, 5):
    for metric in ["entail_min", "info_min", "spec_faith"]:
        s_row = load("results/sonnet_64_dissimilar_eval_by_gpt4o_mini.json")["per_level"][li]
        h_row = load("results/haiku_64_dissimilar_eval_by_gpt4o_mini.json")["per_level"][li]
        q_row = load("results/qwen7b_64_dissimilar_eval_by_gpt4o_mini.json")["per_level"][li]
        s = s_row[metric]; h = h_row[metric]; q = q_row[metric]
        best = max([("Sonnet", s), ("Haiku", h), ("Qwen7B", q)], key=lambda x: x[1])[0]
        A(f"  L{li:<6}{metric:<16}{s:<10.3f}{h:<10.3f}{q:<10.3f}{best}")
A("")
A(sep)
A("   SECTION 3  --  SIMILARITY MODEL  (scincl vs mpnet)")
A(sep)
A("")
A(f"  {'Model':<10}{'Within-Sim':<13}{'Within-Dis':<13}{'Cross':<10}{'Separation'}")
A("  " + dash)
for name in ["mpnet", "scincl"]:
    d = load(f"results/similarity_{name}_stats.json")
    if d:
        A(f"  {name:<10}{d['within_similar']['mean']:<13.4f}{d['within_dissimilar']['mean']:<13.4f}{d['cross']['mean']:<10.4f}{d['separation']:+.4f}")
A("")
A(sep)
A("   KEY FINDINGS")
A(sep)
A("")
A("  1. Sonnet and Haiku perform comparably on both sets, both")
A("     ahead of Qwen 7B on entailment and informativeness.")
A("")
A("  2. Haiku shows stronger SPECIFICITY (spec_faith) across")
A("     levels in both sets (peaks at 1.000 on dissimilar L1).")
A("")
A("  3. mpnet separates similar vs dissimilar papers far better")
A("     than scincl (+0.485 vs +0.191). scincl gives ~0.687 avg")
A("     similarity for dissimilar papers, i.e. too high to be")
A("     discriminative. ** mpnet retained for pipeline. **")
A("")
A("  4. Judge caveat: scores from GPT-4o-mini (not Qwen 3B used")
A("     in Phase 1). Absolute values differ; relative ranking")
A("     across 3 summarizers remains valid.")
A("")
A(sep)
A("   END OF REPORT")
A(sep)

with open("REPORT_PROFESSIONAL.txt", "w") as f:
    f.write("\n".join(lines))
print("Saved: REPORT_PROFESSIONAL.txt")
print(f"Lines: {len(lines)}")
