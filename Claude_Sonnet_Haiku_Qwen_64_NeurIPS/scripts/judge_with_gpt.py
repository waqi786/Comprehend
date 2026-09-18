"""
Judge Sonnet/Haiku/Qwen summaries using GPT-4o-mini.
Computes: entail, faithfulness, informativeness, specificity
for both source abstracts A and B (8 metrics per summary).

Same judge for all 3 summarizers -> fair comparison.
"""
import json, os, sys, re
from pathlib import Path
from openai import OpenAI

# ------------------------------------------------------------------
# Load Phase 1 blossom (for source prompts + reference metrics)
# ------------------------------------------------------------------
with open("results/UF_random_prompts_64_base_similar_blossom.json") as f:
    tree = json.load(f)

level_0 = tree["levels"][0]["nodes"]
id_to_prompt = {n["id"]: n.get("prompt", "") for n in level_0}

with open("results/pairs_64.json") as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} pairs")

# ------------------------------------------------------------------
# Judge prompts (from Phase 1 evaluate.py)
# ------------------------------------------------------------------
FAITHFULNESS_PROMPT = """You will be given two sentences: Sentence 1 and Sentence 2.

Your task is to rate the degree to which Sentence 1 entails Sentence 2.

Use a 0-1 scale:
1.0 = Sentence 1 clearly entails Sentence 2
0.5 = partial entailment
0.0 = Sentence 1 does not entail Sentence 2

Sentence 1: {source}
Sentence 2: {summary}

Respond with ONLY a single number between 0 and 1.
"""

INFORMATIVENESS_PROMPT = """You will be given a source sentence (Sentence 1) and a summary sentence (Sentence 2).

Rate how informative Sentence 2 is with respect to Sentence 1 — i.e., how much specific content from Sentence 1 is preserved in Sentence 2.

Use a 0-1 scale:
1.0 = Sentence 2 captures all key specifics of Sentence 1
0.5 = partial preservation
0.0 = no specific content preserved

Sentence 1: {source}
Sentence 2: {summary}

Respond with ONLY a single number between 0 and 1.
"""

SPECIFICITY_PROMPT = """You will be given a summary sentence and an UNRELATED source abstract.

Rate how strongly the summary is implied by the unrelated abstract, on a 0-1 scale:
1.0 = unrelated abstract clearly entails the summary (summary is TOO GENERIC)
0.5 = partial entailment
0.0 = unrelated abstract does NOT entail the summary (summary is SPECIFIC — good)

Unrelated abstract: {outsider}
Summary: {summary}

Respond with ONLY a single number between 0 and 1.
"""

def score_one(client, prompt):
    """Call GPT-4o-mini and extract a number between 0 and 1."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = r.choices[0].message.content.strip()
        m = re.search(r"(\d+(\.\d+)?)", txt)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        return 0.0
    except Exception as e:
        print(f"  Judge call failed: {e}")
        return None

# ------------------------------------------------------------------
# Load all 3 summarizers
# ------------------------------------------------------------------
with open("results/sonnet_64_summaries.json") as f:
    sonnet = json.load(f)
with open("results/haiku_64_summaries.json") as f:
    haiku = json.load(f)
with open("results/qwen7b_64_summaries.json") as f:
    qwen = json.load(f)

# ------------------------------------------------------------------
# Outsider (from dissimilar set for specificity check)
# ------------------------------------------------------------------
DISSIM_PATH = Path("input/UF_random_prompts_64_base_dissimilar.json")
outsider = None
if DISSIM_PATH.exists():
    with open(DISSIM_PATH) as f:
        dissim = json.load(f)
    d0 = dissim["levels"][0]["nodes"]
    if d0:
        outsider = d0[0].get("prompt", "")
print(f"Outsider loaded: {bool(outsider)}")

# ------------------------------------------------------------------
# Run judging
# ------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def judge_summarizer(tag, summaries):
    print(f"\n=== Judging {tag} ({len(summaries)} summaries) ===")
    out = []
    for i, item in enumerate(summaries):
        a_text = id_to_prompt.get(item["a_id"], "")[:800]
        b_text = id_to_prompt.get(item["b_id"], "")[:800]
        s = item["summary"] or ""

        if not s:
            out.append({**item, "e_a": None, "e_b": None, "i_a": None, "i_b": None,
                        "spec_faith": None})
            continue

        e_a = score_one(client, FAITHFULNESS_PROMPT.replace("{source}", a_text).replace("{summary}", s))
        e_b = score_one(client, FAITHFULNESS_PROMPT.replace("{source}", b_text).replace("{summary}", s))
        i_a = score_one(client, INFORMATIVENESS_PROMPT.replace("{source}", a_text).replace("{summary}", s))
        i_b = score_one(client, INFORMATIVENESS_PROMPT.replace("{source}", b_text).replace("{summary}", s))
        spec = score_one(client, SPECIFICITY_PROMPT.replace("{outsider}", outsider[:800]).replace("{summary}", s)) if outsider else None

        # spec_faith = 1 - outsider entail (higher = more specific)
        spec_faith = (1.0 - spec) if spec is not None else None

        out.append({
            **item,
            "e_a": e_a, "e_b": e_b,
            "i_a": i_a, "i_b": i_b,
            "spec_outsider": spec,
            "spec_faith": spec_faith,
            "entail_min": min(e_a, e_b) if e_a is not None and e_b is not None else None,
            "info_min":   min(i_a, i_b) if i_a is not None and i_b is not None else None,
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(summaries)}")

    out_file = f"results/{tag}_64_judged_by_gpt4o_mini.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_file}")
    return out

sonnet_j = judge_summarizer("sonnet", sonnet)
haiku_j  = judge_summarizer("haiku", haiku)
qwen_j   = judge_summarizer("qwen7b", qwen)

# ------------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------------
def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

print("\n" + "=" * 60)
print("COMPARISON SUMMARY (GPT-4o-mini judge)")
print("=" * 60)
print(f"{'Metric':<22}{'Sonnet':<10}{'Haiku':<10}{'Qwen7B':<10}")
print("-" * 60)
for m in ["e_a", "e_b", "i_a", "i_b", "entail_min", "info_min", "spec_faith"]:
    s = avg([x.get(m) for x in sonnet_j])
    h = avg([x.get(m) for x in haiku_j])
    q = avg([x.get(m) for x in qwen_j])
    fmt = lambda v: f"{v:.3f}" if v is not None else "-"
    print(f"{m:<22}{fmt(s):<10}{fmt(h):<10}{fmt(q):<10}")
print()
