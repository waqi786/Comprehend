"""
Judge comparison trees with GPT-4o-mini using Phase 1 EXACT prompts + formulas.
Output: per-level stats (entail_1, entail_2, faith, info, spec_faith).
"""
import json, os, re, sys
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
            model="gpt-4o-mini",
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt_text}],
        )
        txt = r.choices[0].message.content
        m = re.search(r"[Ss]core[:\s]+([0-9]*\.?[0-9]+)", txt)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        m = re.search(r"([0-9]*\.?[0-9]+)", txt)
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception as e:
        print(f"    judge failed: {e}")
        return None

def eval_tree(tree_path):
    with open(tree_path) as f:
        tree = json.load(f)

    per_level = []
    for li, lvl in enumerate(tree["levels"]):
        nodes = lvl["nodes"]
        e1s, e2s, f1s, f2s, i1s, i2s, sfs = [], [], [], [], [], [], []

        for node in nodes:
            if li == 0:
                continue  # L0 has no summary to judge
            summary = node.get("summary")
            if not summary:
                continue
            p_a = node.get("prompt", "")
            m_idx = node.get("match")
            t_idx = node.get("third")

            # --- Entail (evalPrompt2, match-based) ---
            if m_idx is not None and m_idx < len(nodes):
                p_b = nodes[m_idx].get("prompt", "")
                inp_a = f"\nAbstract A: {p_a}\nAbstract B: {summary}\n"
                inp_b = f"\nAbstract A: {p_b}\nAbstract B: {summary}\n"
                ea = judge(P_ENTAIL + inp_a)
                eb = judge(P_ENTAIL + inp_b)
                if ea is not None and eb is not None:
                    e1s.append((ea + eb) / 2)
                    e2s.append(min(ea, eb))

                # --- Faith (match-based) ---
                fin_a = f"\n## Abstract A (child)\n{p_a}\n## Abstract B (parent)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                fin_b = f"\n## Abstract A (child)\n{p_b}\n## Abstract B (parent)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                fa = judge(P_FAITH + fin_a)
                fb = judge(P_FAITH + fin_b)
                ia = judge(P_INFO + fin_a)
                ib = judge(P_INFO + fin_b)
                if fa is not None and fb is not None:
                    f1s.append((fa + fb) / 2)
                    f2s.append(min(fa, fb))
                if ia is not None and ib is not None:
                    i1s.append((ia + ib) / 2)
                    i2s.append(min(ia, ib))

            # --- Specificity (spec_faith, third-based) ---
            if t_idx is not None and t_idx < len(nodes):
                p_t = nodes[t_idx].get("prompt", "")
                sfin_a = f"\n## Abstract A (candidate)\n{p_a}\n## Abstract B (category)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                sfin_b = f"\n## Abstract A (candidate)\n{p_t}\n## Abstract B (category)\n{summary}\n\nOutput your score and a brief justification in the following format:\nScore: [SCORE]\nReason: [one or two sentences]\n"
                sa = judge(P_SPEC_FAITH + sfin_a)
                sb = judge(P_SPEC_FAITH + sfin_b)
                if sa is not None and sb is not None:
                    sfs.append((sa + sb) / 2)

        def avg(x): return sum(x)/len(x) if x else 0.0
        per_level.append({
            "level": li,
            "n": len(e1s),
            "entail_1": avg(e1s),
            "entail_2": avg(e2s),
            "faithfulness": avg(f1s),
            "informativeness": avg(i1s),
            "spec_faith": avg(sfs),
        })
    return per_level

if __name__ == "__main__":
    subset = sys.argv[1] if len(sys.argv) > 1 else "similar"
    for tag in ["sonnet", "haiku", "qwen7b"]:
        src = f"results/UF_random_prompts_64_base_{subset}_{tag}_blossom.json"
        if not Path(src).exists():
            print(f"SKIP {src}")
            continue
        print(f"\n=== Judging {tag} ({subset}) ===")
        stats = eval_tree(src)
        out = f"results/{tag}_{subset}_gpt4o_fixed_eval.json"
        with open(out, "w") as f:
            json.dump({"per_level": stats}, f, indent=2)
        print(f"  Saved {out}")
        for row in stats:
            if row["n"] > 0:
                print(f"  L{row['level']}: n={row['n']:2d} | entail_1={row['entail_1']:.3f} | entail_2={row['entail_2']:.3f} | faith={row['faithfulness']:.3f} | info={row['informativeness']:.3f} | spec_faith={row['spec_faith']:.3f}")
