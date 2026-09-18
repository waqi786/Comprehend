"""
Claude Sonnet vs Haiku vs Qwen 7B — 64-paper summarizer comparison.

Reads Phase 1 blossom result tree, extracts level-0 -> level-1 pairs,
generates summaries with each model using the SAME prompt14, saves
for later evaluation with Qwen 3B.

Saves:
  results/sonnet_64_summaries.json
  results/haiku_64_summaries.json
  results/pairs_64.json  (shared pairs list for reference)
"""
import json, os, sys, re
from pathlib import Path

# ------------------------------------------------------------------
# 1) Load Phase 1 blossom result (which has the ACTUAL tree structure)
# ------------------------------------------------------------------
PHASE1_TREE = Path("results/UF_random_prompts_64_base_similar_blossom.json")
if not PHASE1_TREE.exists():
    print(f"ERROR: {PHASE1_TREE} not found. Run from scripts/ dir.")
    sys.exit(1)

with open(PHASE1_TREE) as f:
    tree = json.load(f)

# Find level 0 (64 papers) and level 1 (32 parent summaries)
levels = tree["levels"]
level_0 = levels[0]["nodes"]  # 64 papers
level_1 = levels[1]["nodes"]  # 32 parents

print(f"Level 0: {len(level_0)} nodes")
print(f"Level 1: {len(level_1)} nodes")

# ------------------------------------------------------------------
# 2) Extract pairs (SAME AS PHASE 1: node.match partner in same level)
# ------------------------------------------------------------------
# Phase 1 summarize.py: story1 = node.prompt, story2 = level_obj.nodes[node.match].prompt
# So pairs = (i, node.match) for unique i, skip if partner already paired.
seen = set()
pairs = []
for i, node in enumerate(level_0):
    if i in seen:
        continue
    m = node.get("match")
    if m is None or m == i or m >= len(level_0):
        continue
    seen.add(i)
    seen.add(m)
    pairs.append({
        "parent_id": None,
        "a_id": node["id"],
        "b_id": level_0[m]["id"],
        "a_idx": i,
        "b_idx": m,
    })

print(f"Extracted {len(pairs)} pairs (Phase 1 match-based)")
if not pairs:
    print("ERROR: no pairs extracted.")
    sys.exit(1)

# Save pairs reference
os.makedirs("results", exist_ok=True)
with open("results/pairs_64.json", "w") as f:
    json.dump(pairs, f, indent=2)
print("Saved results/pairs_64.json")

# ------------------------------------------------------------------
# 3) The PROMPT (same as Phase 1 prompt14)
# ------------------------------------------------------------------
PROMPT_TEMPLATE = """You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to generate a common Abstract C where Abstract C is implied by BOTH Abstract A and Abstract B. Every detail in Abstract C must be directly supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary sentence or paragraph, NOT a writing task, instruction, or request.
- The more similar Abstract A and B are, the longer Abstract C can be. The more different they are, the more general Abstract C should be.
- If A and B share no specific finding, method, or topic, write ONE concise sentence capturing their shared broader research area.
- NEVER invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- Begin your Abstract C with "(start)" and end with "(end)."

Example 1 (Dissimilar subfields):
Abstract A: "This paper presents the first machine translation system for English-Pnar, achieving BLEU 14.97 Pnar-to-English."
Abstract B: "We propose a sentiment analysis model for Hausa using AfriBertA, outperforming baselines on NaijaSenti."
Abstract C: (start) This work develops natural language processing systems for under-resourced languages. (end)

Example 2 (Moderately similar):
Abstract A: "We fine-tune mBERT for POS tagging on Kyrgyz, achieving 91% accuracy."
Abstract B: "We pretrain maiBERT for Maithili, achieving 87.02% on news classification."
Abstract C: (start) This work develops language-specific BERT models for under-resourced South and Central Asian languages and evaluates them on downstream NLP tasks. (end)

Example 3 (Very similar):
Abstract A: "We compare rule-based, MT, and LLM approaches for Zarma GEC, with M2M100 achieving 95.82% detection rate."
Abstract B: "We study GEC for Zarma using rule-based, MT, and LLM approaches, validating on Bambara."
Abstract C: (start) This work compares rule-based, machine translation, and large language model approaches for grammatical error correction in West African low-resource languages. (end)

Now carry out this task and output Abstract C for the following:

Abstract A: {story1}
Abstract B: {story2}
"""

def extract_summary(raw):
    """Extract text between (start) and (end)."""
    m = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
    return m.group(1).strip() if m else raw.strip()

# ------------------------------------------------------------------
# 4) Load IDs -> prompts map from level 0
# ------------------------------------------------------------------
id_to_prompt = {n["id"]: n.get("prompt", "") for n in level_0}

# ------------------------------------------------------------------
# 5) Claude (Sonnet + Haiku) generation
# ------------------------------------------------------------------
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku":  "claude-haiku-4-5-20251001",
}

for tag, model_name in MODELS.items():
    print(f"\n=== Generating with {tag} ({model_name}) ===")
    out = []
    for i, pair in enumerate(pairs):
        a_text = id_to_prompt.get(pair["a_id"], "")
        b_text = id_to_prompt.get(pair["b_id"], "")
        prompt = PROMPT_TEMPLATE.replace("{story1}", a_text[:3000]).replace("{story2}", b_text[:3000])

        try:
            r = client.messages.create(
                model=model_name,
                max_tokens=512,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = r.content[0].text
            summary = extract_summary(raw)
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")
            summary = ""

        out.append({
            "parent_id": pair["parent_id"],
            "a_id": pair["a_id"],
            "b_id": pair["b_id"],
            "summary": summary,
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(pairs)} done")

    out_file = f"results/{tag}_64_summaries.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_file} ({len(out)} summaries)")

print("\n=== DONE ===")
