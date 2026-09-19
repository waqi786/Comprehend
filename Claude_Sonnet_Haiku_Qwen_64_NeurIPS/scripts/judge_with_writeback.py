"""
Re-judge with per-node write-back. 
Updates blossom tree with new per-node metrics, THEN saves per-level stats.
"""
import json, os, re
from pathlib import Path
from openai import OpenAI

PROMPT_DIR = Path("phase1_prompts")
P_ENTAIL = (PROMPT_DIR / "evalPrompt2.txt").read_text()
P_FAITH  = (PROMPT_DIR / "evalPrompt_faithfulness.txt").read_text()
P_INFO   = (PROMPT_DIR / "evalPrompt_informativeness.txt").read_text()
P_SPEC_FAITH = (PROMPT_DIR / "evalPrompt_specificity_faithfulness.txt").read_text()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def judge(prompt_text):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=200, temperature=0.0,
            messages=[{"role": "user", "content": prompt_text}],
        )
        txt = r.choices[0].message.content
        m = re.search(r"[Ss]core[:\s]+([0-9]*\.?[0-9]+)", txt)
        if m: return max(0.0, min(1.0, float(m.group(1))))
        m = re.search(r"([0-9]*\.?[0-9]+)", txt)
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception as e:
        print(f"    judge failed: {e}")
        return None

def eval_and_writeback(tree_path):
    with open(tree_path) as f:
        tree = json.load(f)
    per_level = []
    for li, lvl in enumerate(tree["levels"]):
        nodes = lvl["nodes"]
        e1s, e2s, f1s, f2s, i1s, i2s, sfs = [], [], [], [], [], [], []
        for node in nodes:
            if li == 0: continue
            summary = node.get("summary")
            if not summary: continue
            p_a = node.get("prompt", "")
            m_idx = node.get("match")
            t_idx = node.get("third")

            if m_idx is not None and m_idx < len(nodes):
                p_b = nodes[m_idx].get("prompt", "")
                inp_a = f"\nAbstract A: {p_a}\nAbstract B: {summary}\n"
                inp_b = f"\nAbstract A: {p_b}\nAbstract B: {summary}\n"
                ea = judge(P_ENTAIL + inp_a); eb = judge(P_ENTAIL + inp_b)
                if ea is not None and eb is not None:
                    v1 = (ea + eb) / 2; v2 = min(ea, eb)
                    node["entail_1"] = v1; node["entail_2"] = v2
                    e1s.append(v1); e2s.append(v2)

                fin_a = f"\n## Abstract A (child)\n{p_a}\n## Abstract B (parent)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                fin_b = f"\n## Abstract A (child)\n{p_b}\n## Abstract B (parent)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                fa = judge(P_FAITH + fin_a); fb = judge(P_FAITH + fin_b)
                ia = judge(P_INFO + fin_a);  ib = judge(P_INFO + fin_b)
                if fa is not None and fb is not None:
                    v1 = (fa + fb) / 2; v2 = min(fa, fb)
                    node["faithfulness1"] = v1; node["faithfulness2"] = v2
                    f1s.append(v1); f2s.append(v2)
                if ia is not None and ib is not None:
                    v1 = (ia + ib) / 2; v2 = min(ia, ib)
                    node["informativeness1"] = v1; node["informativeness2"] = v2
                    i1s.append(v1); i2s.append(v2)

            if t_idx is not None and t_idx < len(nodes):
                p_t = nodes[t_idx].get("prompt", "")
                sfin_a = f"\n## Abstract A (candidate)\n{p_a}\n## Abstract B (category)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                sfin_b = f"\n## Abstract A (candidate)\n{p_t}\n## Abstract B (category)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                sa = judge(P_SPEC_FAITH + sfin_a); sb = judge(P_SPEC_FAITH + sfin_b)
                if sa is not None and sb is not None:
                    v = (sa + sb) / 2
                    node["specificity_faithfulness"] = v
                    sfs.append(v)

        def avg(x): return sum(x)/len(x) if x else 0.0
        per_level.append({
            "level": li, "n": len(e1s),
            "entail_1": avg(e1s), "entail_2": avg(e2s),
            "faithfulness": avg(f1s), "informativeness": avg(i1s),
            "spec_faith": avg(sfs),
        })

    # Write back updated tree
    with open(tree_path, "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    return per_level

# Run for all 6 combos
for subset in ["similar", "dissimilar"]:
    for tag in ["sonnet", "haiku", "qwen7b"]:
        src = f"results/UF_random_prompts_64_base_{subset}_{tag}_blossom.json"
        if not Path(src).exists():
            print(f"SKIP {src}"); continue
        print(f"\n=== Judging {tag}/{subset} ===")
        stats = eval_and_writeback(src)
        out = f"results/{tag}_{subset}_gpt4o_fixed_eval.json"
        with open(out, "w") as f:
            json.dump({"per_level": stats}, f, indent=2)
        print(f"  Updated {src} + saved {out}")
        for row in stats:
            if row["n"] > 0:
                print(f"  L{row['level']}: n={row['n']:2d} | entail_1={row['entail_1']:.3f} | spec_faith={row['spec_faith']:.3f}")
