"""
Compare scincl (malteos/scincl) vs mpnet (all-mpnet-base-v2) on 64 similar + 64 dissimilar papers.
"""
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_texts(csv_path):
    df = pd.read_csv(csv_path)
    return (df['title'].fillna('') + '. ' + df['abstract'].fillna('')).tolist()

def embed(model, texts):
    return model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

def stats(sim_matrix):
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = sim_matrix[mask]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p90": float(np.percentile(vals, 90)),
    }

print("Loading texts...")
sim_texts = load_texts("input/neurips_similar_64.csv")
dis_texts = load_texts("input/neurips_dissimilar_64.csv")
all_texts = sim_texts + dis_texts
print(f"Similar: {len(sim_texts)}, Dissimilar: {len(dis_texts)}, Total: {len(all_texts)}")

for name, model_name in [("mpnet", "sentence-transformers/all-mpnet-base-v2"),
                          ("scincl", "malteos/scincl")]:
    print(f"\n=== {name} ({model_name}) ===")
    m = SentenceTransformer(model_name)
    emb = embed(m, all_texts)

    sim_emb = emb[:64]
    dis_emb = emb[64:]

    sim_sim = cosine_similarity(sim_emb, sim_emb)
    dis_dis = cosine_similarity(dis_emb, dis_emb)
    sim_dis = cosine_similarity(sim_emb, dis_emb)

    s_sim = stats(sim_sim)
    s_dis = stats(dis_dis)
    s_cross = stats(sim_dis)

    print(f"  Within-similar:  mean={s_sim['mean']:.4f}  median={s_sim['median']:.4f}  std={s_sim['std']:.4f}")
    print(f"  Within-dissim:   mean={s_dis['mean']:.4f}  median={s_dis['median']:.4f}  std={s_dis['std']:.4f}")
    print(f"  Cross (sim x dis): mean={s_cross['mean']:.4f}")
    print(f"  Separation:      {s_sim['mean'] - s_cross['mean']:+.4f}")

    out = f"results/similarity_{name}_stats.json"
    with open(out, "w") as f:
        json.dump({
            "model": model_name,
            "within_similar": s_sim,
            "within_dissimilar": s_dis,
            "cross": s_cross,
            "separation": s_sim['mean'] - s_cross['mean'],
        }, f, indent=2)
    print(f"  Saved: {out}")

print("\n=== DONE ===")
