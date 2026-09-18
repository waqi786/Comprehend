"""
Dissimilar 64-paper comparison: Sonnet vs Haiku vs Qwen 7B.
Generates full trees (levels 1-6) using each model, same Phase 1 prompt.
"""
import json, os, sys, re, copy
from pathlib import Path
from anthropic import Anthropic

PHASE1_DISSIM = Path("results/UF_random_prompts_64_base_dissimilar_blossom.json")

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

if len(sys.argv) < 3:
    print("Usage: python3 scripts/compare_dissimilar.py <tag> <model_name>")
    sys.exit(1)

tag = sys.argv[1]
model_name = sys.argv[2]
print(f"Tag: {tag}, Model: {model_name}")

with open(PHASE1_DISSIM) as f:
    tree = copy.deepcopy(json.load(f))

# ---- Generate Level 1 (32 pairs) ----
from anthropic import Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
L0 = tree["levels"][0]["nodes"]
L1 = tree["levels"][1]["nodes"]

seen = set()
pairs = []
for i, node in enumerate(L0):
    if i in seen: continue
    m = node.get("match")
    if m is None or m == i or m >= len(L0): continue
    seen.add(i); seen.add(m)
    pairs.append((i, m))

print(f"Level 0: {len(L0)} nodes, {len(L1)} L1 nodes")
# Generate L1 summary for each L1 node using its children
id_to_prompt_l0 = {n["id"]: n.get("prompt", "") for n in L0}
print(f"=== Generating Level 1 ({len(L1)} summaries) ===")
for k, node in enumerate(L1):
    kids = node.get("children") or []
    if len(kids) < 2:
        continue
    s1 = id_to_prompt_l0.get(kids[0], "")[:3000]
    s2 = id_to_prompt_l0.get(kids[1], "")[:3000]
    if not s1 or not s2: continue
    prompt = PROMPT_TEMPLATE.replace("{story1}", s1).replace("{story2}", s2)
    try:
        r = client.messages.create(model=model_name, max_tokens=512, temperature=0.7,
            messages=[{"role": "user", "content": prompt}])
        smry = extract_summary(r.content[0].text)
        node["prompt"] = smry
        node["summary"] = smry
    except Exception as e:
        print(f"  [{k}] FAILED: {e}")
    if (k+1) % 5 == 0:
        print(f"  {k+1}/{len(L1)}")

# ---- Levels 2-6 ----
for L in range(1, 6):
    upper = tree["levels"][L+1]["nodes"]
    lower = tree["levels"][L]["nodes"]
    id_to_idx = {n["id"]: idx for idx, n in enumerate(lower)}
    print(f"\n=== Level {L+1} ({len(upper)} nodes) ===")
    for j, node in enumerate(upper):
        kids = node.get("children") or []
        if len(kids) < 2: continue
        i1 = id_to_idx.get(kids[0]); i2 = id_to_idx.get(kids[1])
        if i1 is None or i2 is None: continue
        s1 = lower[i1].get("prompt", "")[:3000]
        s2 = lower[i2].get("prompt", "")[:3000]
        if not s1 or not s2: continue
        prompt = PROMPT_TEMPLATE.replace("{story1}", s1).replace("{story2}", s2)
        try:
            r = client.messages.create(model=model_name, max_tokens=512, temperature=0.7,
                messages=[{"role": "user", "content": prompt}])
            smry = extract_summary(r.content[0].text)
            node["prompt"] = smry
            node["summary"] = smry
        except Exception as e:
            print(f"  [{L+1}:{j}] FAILED: {e}")

# Fix root if needed
root = tree["levels"][6]["nodes"][0]
if root.get("summary") is None and root.get("prompt"):
    root["summary"] = root["prompt"]

out = f"results/UF_random_prompts_64_base_dissimilar_{tag}_blossom.json"
with open(out, "w") as f:
    json.dump(tree, f, indent=2, ensure_ascii=False)
print(f"\nSaved {out}")
