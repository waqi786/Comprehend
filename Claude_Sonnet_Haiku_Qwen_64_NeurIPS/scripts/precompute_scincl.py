"""
Precompute SciNCL embeddings for 64 similar + 64 dissimilar papers.
Saves to outputs/malteos_scincl_embeddings.pkl (Phase 1 cache format).
"""
import json, pickle, os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("outputs")
CACHE_DIR.mkdir(exist_ok=True)

print("Loading SciNCL model...")
model = SentenceTransformer("malteos/scincl")

for subset in ["similar", "dissimilar"]:
    with open(f"input/{subset}_input.json") as f:
        tree = json.load(f)
    prompts = [n["prompt"] for n in tree["levels"][0]["nodes"]]
    print(f"\n{subset}: {len(prompts)} papers")
    emb = model.encode(prompts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    # Normalize
    denom = np.linalg.norm(emb, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    emb = emb / denom
    cache_path = CACHE_DIR / "malteos_scincl_embeddings.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"sentences": prompts, "embeddings": emb}, f)
    print(f"Saved {cache_path} ({emb.shape})")
    # Rename cache for this subset
    os.rename(cache_path, CACHE_DIR / f"malteos_scincl_embeddings_{subset}.pkl")
    print(f"Renamed to malteos_scincl_embeddings_{subset}.pkl")

print("\nDONE")
