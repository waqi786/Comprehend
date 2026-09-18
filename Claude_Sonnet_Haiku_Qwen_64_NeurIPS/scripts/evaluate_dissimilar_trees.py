"""
Evaluate all 3 trees (sonnet, haiku, qwen7b) with GPT-4o-mini judge.
For each level (except root), score: entail_min, info_min, spec_faith.
Uses Phase 1 pairs (each level node's match partner).
"""
import json, os, re
from pathlib import Path
from openai import OpenAI

FAITH_PROMPT = """You will be given two sentences: Sentence 1 and Sentence 2.
Rate the degree to which Sentence 1 entails Sentence 2 (0-1 scale):
1.0 = clear entailment, 0.5 = partial, 0.0 = no entailment.
Sentence 1: {source}
Sentence 2: {summary}
Respond with ONLY a single number between 0 and 1."""

INFO_PROMPT = """You will be given a source sentence (Sentence 1) and a summary (Sentence 2).
Rate how much specific content from Sentence 1 is preserved in Sentence 2 (0-1):
1.0 = all key specifics preserved, 0.5 = partial, 0.0 = none.
Sentence 1: {source}
Sentence 2: {summary}
Respond with ONLY a single number between 0 and 1."""

SPEC_PROMPT = """You will be given a summary sentence and an UNRELATED abstract.
Rate how strongly the summary is implied by the unrelated abstract (0-1):
1.0 = clearly entailed (too generic), 0.5 = partial, 0.0 = not entailed (specific).
Unrelated abstract: {outsider}
Summary: {summary}
Respond with ONLY a single number between 0 and 1."""

def score(client, prompt):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=10, temperature=0.0,
            messages=[{"role": "user", "content": prompt}])
        m = re.search(r"(\d+(?:\.\d+)?)", r.choices[0].message.content)
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception as e:
        print(f"    judge failed: {e}")
        return None

# Outsider comes from node.third (Phase 1 logic). No global outsider needed.

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

for tag in ["sonnet", "haiku", "qwen7b"]:
    print(f"\n=== Evaluating {tag} ===")
    f = f"results/UF_random_prompts_64_base_dissimilar_{tag}_blossom.json"
    with open(f) as fp:
        tree = json.load(fp)

    levels = tree["levels"]
    per_level = []

    for li in range(6):  # L0..L5
        nodes = levels[li]["nodes"]
        level_scores = {"entail_min": [], "info_min": [], "spec_faith": []}
        seen = set()

        for i, node in enumerate(nodes):
            if i in seen:
                continue
            m = node.get("match")
            if m is None or m >= len(nodes) or m == i:
                continue
            seen.add(i)
            seen.add(m)

            # For L0: node.prompt = original paper
            # For L1+: node.prompt = the summary itself (from lower-level merge)
            # We need source = the summary at this node (already generated)
            # and children = the two nodes below that produced it
            # Actually simpler: at each level, the node's 'prompt' IS the summary
            # The 'sources' are the two children in the level below.

            # But easier: use node.summary as summary, and the two children
            # (from level li-1) as sources. For L0, node.prompt = paper.

            if li == 0:
                # Skip L0 pairs (they are original papers, no summary)
                continue

            summary = node.get("summary") or node.get("prompt", "")
            if not summary:
                continue

            # Find the two children that produced this node's summary
            # In Phase 1, at level li, the summary for node came from
            # merging two nodes at level li-1
            kids = node.get("children") or []
            if len(kids) < 2:
                continue
            lower = levels[li-1]["nodes"]
            id_to_idx = {n["id"]: idx for idx, n in enumerate(lower)}
            i1 = id_to_idx.get(kids[0])
            i2 = id_to_idx.get(kids[1])
            if i1 is None or i2 is None:
                continue
            src1 = lower[i1].get("prompt", "")[:800]
            src2 = lower[i2].get("prompt", "")[:800]

            e_a = score(client, FAITH_PROMPT.replace("{source}", src1).replace("{summary}", summary))
            e_b = score(client, FAITH_PROMPT.replace("{source}", src2).replace("{summary}", summary))
            i_a = score(client, INFO_PROMPT.replace("{source}", src1).replace("{summary}", summary))
            i_b = score(client, INFO_PROMPT.replace("{source}", src2).replace("{summary}", summary))
            # Outsider = node.third (same level) as Phase 1
            third_idx = node.get("third")
            if third_idx is None or third_idx >= len(nodes) or third_idx == i:
                continue
            outsider_text = nodes[third_idx].get("prompt", "")[:800]
            spec = score(client, SPEC_PROMPT.replace("{outsider}", outsider_text).replace("{summary}", summary))

            if e_a is not None and e_b is not None:
                level_scores["entail_min"].append(min(e_a, e_b))
            if i_a is not None and i_b is not None:
                level_scores["info_min"].append(min(i_a, i_b))
            if spec is not None:
                level_scores["spec_faith"].append(1.0 - spec)

        def avg(x): return sum(x)/len(x) if x else 0.0
        per_level.append({
            "level": li,
            "n": len(level_scores["entail_min"]),
            "entail_min": avg(level_scores["entail_min"]),
            "info_min": avg(level_scores["info_min"]),
            "spec_faith": avg(level_scores["spec_faith"]),
        })
        print(f"  L{li}: n={per_level[-1]['n']:2d} | entail={per_level[-1]['entail_min']:.3f} | info={per_level[-1]['info_min']:.3f} | spec_faith={per_level[-1]['spec_faith']:.3f}")

    out = f"results/{tag}_64_dissimilar_eval_by_gpt4o_mini.json"
    with open(out, "w") as fp:
        json.dump({"per_level": per_level}, fp, indent=2)
    print(f"  Saved: {out}")

# ---- Cross-model comparison ----
print("\n" + "="*70)
print("CROSS-MODEL COMPARISON (GPT-4o-mini judge)")
print("="*70)
all_results = {}
for tag in ["sonnet", "haiku", "qwen7b"]:
    with open(f"results/{tag}_64_dissimilar_eval_by_gpt4o_mini.json") as fp:
        all_results[tag] = json.load(fp)["per_level"]

print(f"{'Level':<6}{'Metric':<15}{'Sonnet':<10}{'Haiku':<10}{'Qwen7B':<10}{'Winner'}")
print("-"*70)
for li in range(6):
    for m in ["entail_min", "info_min", "spec_faith"]:
        s = all_results["sonnet"][li][m]
        h = all_results["haiku"][li][m]
        q = all_results["qwen7b"][li][m]
        best = max([("Sonnet", s), ("Haiku", h), ("Qwen7B", q)], key=lambda x: x[1])[0]
        print(f"L{li:<5}{m:<15}{s:<10.3f}{h:<10.3f}{q:<10.3f}{best}")
    print()
