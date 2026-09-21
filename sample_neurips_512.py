import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

random_seed = 42
np.random.seed(random_seed)

print("=== Load NeurIPS papers ===")
df = pd.read_csv('neurips_2025.csv')
df = df[df['num_reviews'] >= 3].reset_index(drop=True)
print(f'Loaded {len(df)} papers (>=3 reviews)')

texts = (df['title'].fillna('') + '. ' + df['abstract'].fillna('')).tolist()

print("=== Encoding embeddings ===")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
emb = model.encode(texts, show_progress_bar=True, batch_size=64)
print(f'Embeddings shape: {emb.shape}')

# === SIMILAR 512 ===
print()
print('=== Finding 512 SIMILAR papers ===')
N = 512
k = 200
km = KMeans(n_clusters=k, n_init=5, random_state=random_seed)
labels = km.fit_predict(emb)
sizes = pd.Series(labels).value_counts()
print(f'Top 5 largest clusters: {sizes.head().to_dict()}')

pool = []
for c in sizes.index:
    pool.extend(np.where(labels == c)[0])
    if len(pool) >= N:
        break
similar_idx = np.random.choice(pool, N, replace=False)
similar_df = df.iloc[similar_idx].reset_index(drop=True)
print(f'Similar set: {len(similar_df)} papers')

# === DISSIMILAR 512 ===
print()
print('=== Finding 512 DISSIMILAR papers ===')
n = len(emb)
selected = [np.random.randint(n)]
for i in range(N - 1):
    dists = euclidean_distances(emb, emb[selected])
    min_dists = dists.min(axis=1)
    min_dists[selected] = -1
    selected.append(int(np.argmax(min_dists)))
    if (i + 1) % 100 == 0:
        print(f'  Selected {i+1}/{N}')

dissimilar_idx = np.array(selected)
dissimilar_df = df.iloc[dissimilar_idx].reset_index(drop=True)
print(f'Dissimilar set: {len(dissimilar_df)} papers')

# === SAVE ===
similar_df.to_csv('neurips_similar_512.csv', index=False)
dissimilar_df.to_csv('neurips_dissimilar_512.csv', index=False)
print()
print('✅ Saved: neurips_similar_512.csv, neurips_dissimilar_512.csv')
