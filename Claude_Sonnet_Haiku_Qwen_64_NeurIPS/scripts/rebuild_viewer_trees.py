"""
Rebuild viewer tree_data files with correct field usage:
- L0 -> prompt (original paper abstract)
- L1+ -> summary
"""
import json
import pandas as pd
import numpy as np

METRICS = ['rating', 'quality', 'clarity', 'significance', 'originality']

def build(blossom_path, csv_path, out_path, is_similar):
    with open(blossom_path) as f:
        tree = json.load(f)
    df = pd.read_csv(csv_path)
    levels = tree['levels']
    viewer_nodes = []

    for li, lvl in enumerate(levels):
        for node in lvl['nodes']:
            nid = str(node['id'])
            parent_ids = []
            if li + 1 < len(levels):
                for up in levels[li+1]['nodes']:
                    if node['id'] in (up.get('children') or []):
                        parent_ids.append(str(up['id']))

            # BUG FIX: L0 = prompt, L1+ = summary
            if li == 0:
                text = node.get('prompt', '')
            else:
                text = node.get('summary') or node.get('prompt', '')

            v = {
                "id": nid, "parentIds": parent_ids,
                "name": text[:200],
                "full_text": text,
                "similarity": node.get('text_similarity'),
                "summary_blocked": node.get('summary_blocked', False),
                "evalScore": node.get('evalScore'),
                "entail1": node.get('entail_1'), "entail2": node.get('entail_2'),
                "faithfulness1": node.get('faithfulness1'),
                "faithfulness2": node.get('faithfulness2'),
                "informativeness1": node.get('informativeness1'),
                "informativeness2": node.get('informativeness2'),
                "specificity_faithfulness": node.get('specificity_faithfulness'),
                "specificity_informativeness": node.get('specificity_informativeness'),
                "is_leaf": li == 0,
                "avg_score": None, "variance": None,
                "avg_rating": None, "var_rating": None,
                "avg_quality": None, "var_quality": None,
                "avg_clarity": None, "var_clarity": None,
                "avg_significance": None, "var_significance": None,
                "avg_originality": None, "var_originality": None,
            }
            if li == 0 and node['id'] < len(df):
                for m in METRICS:
                    vals = [df.iloc[node['id']][f'{m}_{i}'] for i in range(1, 6) if f'{m}_{i}' in df.columns]
                    vals = [float(x) for x in vals if pd.notna(x)]
                    if vals:
                        v[f'avg_{m}'] = float(np.mean(vals))
                        v[f'var_{m}'] = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0
                v['avg_score'] = v['avg_rating']
                v['variance'] = v['var_rating']
            viewer_nodes.append(v)

    id_to_v = {n['id']: n for n in viewer_nodes}
    children_map = {n['id']: [] for n in viewer_nodes}
    for n in viewer_nodes:
        for pid in n['parentIds']:
            if pid in children_map:
                children_map[pid].append(n['id'])

    cache = {}
    def get_leaves(nid):
        if nid in cache: return cache[nid]
        ch = children_map.get(nid, [])
        if not ch:
            cache[nid] = [nid]
            return [nid]
        out = []
        for c in ch:
            out.extend(get_leaves(c))
        cache[nid] = out
        return out

    for n in viewer_nodes:
        if n['is_leaf']: continue
        leaves = get_leaves(n['id'])
        for m in METRICS:
            avgs = [id_to_v[l].get(f'avg_{m}') for l in leaves if id_to_v[l].get(f'avg_{m}') is not None]
            vars_ = [id_to_v[l].get(f'var_{m}') for l in leaves if id_to_v[l].get(f'var_{m}') is not None]
            if avgs: n[f'avg_{m}'] = float(np.mean(avgs))
            if vars_: n[f'var_{m}'] = float(np.mean(vars_))
        n['avg_score'] = n.get('avg_rating')
        n['variance'] = n.get('var_rating')

    with open(out_path, 'w') as f:
        json.dump(viewer_nodes, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path} ({len(viewer_nodes)} nodes)")

# Similar
for tag in ['sonnet', 'haiku', 'qwen7b']:
    build(f'results/UF_random_prompts_64_base_similar_{tag}_blossom.json',
          'input/neurips_similar_64.csv',
          f'viewers/tree_data_similar_{tag}.json',
          True)

# Dissimilar
for tag in ['sonnet', 'haiku', 'qwen7b']:
    build(f'results/UF_random_prompts_64_base_dissimilar_{tag}_blossom.json',
          'input/neurips_dissimilar_64.csv',
          f'viewers/tree_data_dissimilar_{tag}.json',
          False)
