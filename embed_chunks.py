import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the chunks from Day 2
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

# 2. Load a small, free embedding model (downloads on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Turn every chunk into a vector
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
print("Embeddings shape:", embeddings.shape)

# 4. Save so you don't have to recompute
np.save("embeddings.npy", embeddings)

# 5. Quick sanity check: ask a question, find the closest chunks
question = "What is the standard deviation?"
q_vec = model.encode([question], normalize_embeddings=True)[0]
scores = embeddings @ q_vec
top = np.argsort(scores)[::-1][:3]

for rank, idx in enumerate(top, start=1):
    c = chunks[idx]
    print(f"\n#{rank} (score {scores[idx]:.3f}) {c['source']} | page {c['page']}")
    print(c["text"][:300])
