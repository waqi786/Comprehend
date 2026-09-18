"""
Generate full 7-level tree using a given summarizer model.
Usage: python3 scripts/generate_full_tree.py <tag> <model_name>
       tag: sonnet / haiku / qwen7b
"""
import json, os, sys, re, copy
from pathlib import Path

PHASE1_TREE = Path("results/UF_random_prompts_64_base_similar_blossom.json")

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
    m = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
    return m.group(1).strip() if m else raw.strip()


# ---- Args ----
if len(sys.argv) < 3:
    print("Usage: python3 scripts/generate_full_tree.py <tag> <model_name>")
    sys.exit(1)

tag = sys.argv[1]
model_name = sys.argv[2]
print(f"Tag: {tag}")
print(f"Model: {model_name}")

# ---- Load Phase 1 template ----
with open(PHASE1_TREE) as f:
    template = json.load(f)
tree = copy.deepcopy(template)

# ---- Load existing level-1 summaries ----
l1_file = f"results/{tag}_64_summaries.json"
l1_map = {}
if os.path.exists(l1_file):
    with open(l1_file) as f:
        l1_data = json.load(f)
    for item in l1_data:
        key = (item["a_id"], item["b_id"])
        l1_map[key] = item["summary"]
        l1_map[(item["b_id"], item["a_id"])] = item["summary"]
    print(f"Loaded {len(l1_map)} existing L1 summaries")
else:
    print("No existing L1 summaries (will use Phase 1 for qwen7b)")

# ---- For qwen7b: just use Phase 1 tree as-is ----
if tag == "qwen7b":
    out_file = f"results/UF_random_prompts_64_base_similar_qwen7b_blossom.json"
    with open(out_file, "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_file} (Phase 1 blossom as qwen7b tree)")
    sys.exit(0)

# ---- Apply L1 summaries to tree ----
for node in tree["levels"][1]["nodes"]:
    kids = node.get("children") or []
    if len(kids) < 2:
        continue
    for combo in [(kids[0], kids[1]), (kids[1], kids[0])]:
        if combo in l1_map:
            node["prompt"] = l1_map[combo]
            node["summary"] = l1_map[combo]
            break

# ---- Generate levels 2-6 ----
from anthropic import Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

total_calls = 0
for L in range(1, 6):
    nodes_above = tree["levels"][L+1]["nodes"]
    print(f"\n=== Level {L+1} ({len(nodes_above)} nodes) ===")
    level_nodes = tree["levels"][L]["nodes"]
    id_to_idx = {n["id"]: idx for idx, n in enumerate(level_nodes)}

    for i, node in enumerate(nodes_above):
        kids = node.get("children") or []
        if len(kids) < 2:
            continue
        c1, c2 = kids[0], kids[1]
        i1 = id_to_idx.get(c1)
        i2 = id_to_idx.get(c2)
        if i1 is None or i2 is None:
            continue
        story1 = level_nodes[i1].get("prompt", "")
        story2 = level_nodes[i2].get("prompt", "")
        if not story1 or not story2:
            continue
        prompt = PROMPT_TEMPLATE.replace("{story1}", story1[:3000]).replace("{story2}", story2[:3000])
        try:
            r = client.messages.create(
                model=model_name, max_tokens=512, temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = r.content[0].text
            summary = extract_summary(raw)
            node["prompt"] = summary
            node["summary"] = summary
            total_calls += 1
            print(f"  [{L+1}:{i}] OK ({len(summary)} chars)")
        except Exception as e:
            print(f"  [{L+1}:{i}] FAILED: {e}")

out_file = f"results/UF_random_prompts_64_base_similar_{tag}_blossom.json"
with open(out_file, "w") as f:
    json.dump(tree, f, indent=2, ensure_ascii=False)
print(f"\nSaved {out_file}")
print(f"Total new API calls: {total_calls}")
