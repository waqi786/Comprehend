"""
Build SciNCL + Haiku tree (similar + dissimilar).
Uses SciNCL embeddings for blossom matching, Claude Haiku for summaries.
No main-idea extraction (text-only similarity) to keep it simple + GPU-free.
"""
import json, os, re, pickle, sys, copy
from pathlib import Path
import numpy as np
import networkx as nx
from anthropic import Anthropic

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

def haiku_summary(client, a, b):
    prompt = PROMPT_TEMPLATE.replace("{story1}", a[:3000]).replace("{story2}", b[:3000])
    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_summary(r.content[0].text)
    except Exception as e:
        print(f"  haiku failed: {e}")
        return ""

def blossom_match(embeddings):
    """Return list of (i, j) pairs from blossom matching on cosine sim graph."""
    n = embeddings.shape[0]
    sim = np.matmul(embeddings, embeddings.T)
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=float(sim[i][j]))
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    return list(matching), sim

def build_tree(emb_path, out_path, api_key):
    with open(emb_path, "rb") as f:
        cache = pickle.load(f)
    sentences = cache["sentences"]
    embeddings = cache["embeddings"]
    print(f"Loaded {len(sentences)} embeddings from {emb_path}")

    client = Anthropic(api_key=api_key)

    # ----- Level 0: nodes from sentences -----
    # Structure: {"levels": [{"level":0, "nodes":[{id, prompt, summary, match, children, ...}]}, ...]}
    levels = []
    L0_nodes = []
    for i, s in enumerate(sentences):
        L0_nodes.append({
            "id": i, "prompt": s, "summary": None,
            "match": None, "children": [], "alive": True,
            "text_similarity": None, "score": None,
        })

    # Blossom matching on L0
    pairs, sim = blossom_match(embeddings)
    print(f"Blossom produced {len(pairs)} pairs at L0")

    # Winner/loser for each pair (keep lower id alive)
    used = set()
    for a, b in pairs:
        # store match (both directions, but only winner alive)
        winner, loser = (a, b) if a < b else (b, a)
        L0_nodes[winner]["match"] = loser
        L0_nodes[winner]["text_similarity"] = float(sim[winner][loser])
        L0_nodes[winner]["alive"] = True
        L0_nodes[loser]["match"] = winner
        L0_nodes[loser]["text_similarity"] = float(sim[winner][loser])
        L0_nodes[loser]["alive"] = False
        used.add(a); used.add(b)

    # Unmatched (odd count)
    for i in range(len(sentences)):
        if i not in used:
            L0_nodes[i]["alive"] = True
            L0_nodes[i]["unmatched"] = True

    levels.append({"level": 0, "nodes": L0_nodes})

    # ----- Level 1: one summary per pair -----
    L1_nodes = []
    L1_id = 64
    pair_map = {}
    for a, b in pairs:
        winner = a if a < b else b
        loser = b if a < b else a
        pair_map[winner] = (winner, loser)

    for winner, (i, j) in pair_map.items():
        s1 = sentences[i][:3000]
        s2 = sentences[j][:3000]
        summary = haiku_summary(client, s1, s2)
        if not summary:
            continue
        node = {
            "id": L1_id, "prompt": summary, "summary": summary,
            "match": None, "children": [i, j], "alive": True,
            "text_similarity": None, "score": None,
        }
        L1_nodes.append(node)
        L1_id += 1

    levels.append({"level": 1, "nodes": L1_nodes})
    print(f"L1: {len(L1_nodes)} summaries generated (Haiku)")

    # ----- Levels 2-6: recursively build -----
    from sentence_transformers import SentenceTransformer
    print("Loading SciNCL for higher levels (re-embed summaries)...")
    m = SentenceTransformer("all-mpnet-base-v2")

    for li in range(2, 7):
        prev = levels[li-1]["nodes"]
        if len(prev) <= 1:
            break
        texts = [n["prompt"] for n in prev]
        emb = m.encode(texts, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
        denom = np.linalg.norm(emb, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        emb = emb / denom

        pairs, sim = blossom_match(emb)
        used = set()
        for a, b in pairs:
            used.add(a); used.add(b)
            prev[a]["match"] = b
            prev[b]["match"] = a

        new_level_nodes = []
        new_id = max(n["id"] for n in levels[li-1]["nodes"]) + 100  # offset
        next_id = max(n["id"] for lvl in levels for n in lvl["nodes"]) + 1
        for a, b in pairs:
            i, j = a, b
            s1 = prev[i]["prompt"][:3000]
            s2 = prev[j]["prompt"][:3000]
            summary = haiku_summary(client, s1, s2)
            if not summary:
                continue
            node = {
                "id": next_id, "prompt": summary, "summary": summary,
                "match": None, "children": [prev[i]["id"], prev[j]["id"]],
                "alive": True, "text_similarity": None, "score": None,
            }
            new_level_nodes.append(node)
            next_id += 1

        levels.append({"level": li, "nodes": new_level_nodes})
        print(f"L{li}: {len(new_level_nodes)} nodes")

    tree = {"levels": levels, "config": {"method": "blossom", "similarity": "mpnet_textonly", "summarizer": "haiku"}}
    with open(out_path, "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    api_key = os.environ["ANTHROPIC_API_KEY"]
    for subset in ["similar", "dissimilar"]:
        emb = f"outputs/all-mpnet-base-v2_embeddings_{subset}.pkl"
        out = f"results/UF_random_prompts_64_base_{subset}_haiku_mpnet_textonly_blossom.json"
        print(f"\n=== Building {subset.upper()} ===")
        build_tree(emb, out, api_key)
