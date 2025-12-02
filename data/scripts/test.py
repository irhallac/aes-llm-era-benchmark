import torch, numpy as np
from sentence_transformers import SentenceTransformer

# pick ONE:
MODEL = "BAAI/bge-m3"                     # light, multilingual
# MODEL = "Qwen/Qwen3-Embedding-8B"       # larger GPU
# MODEL = "Linq-AI-Research/Linq-Embed-Mistral"
# MODEL = "nomic-ai/nomic-embed-text-v1.5"
# MODEL = "nvidia/NV-Embed-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)
texts = ["This is a test.", "How are you doing today?"]

X = model.encode(texts, normalize_embeddings=False, batch_size=32, convert_to_numpy=True)
# L2-normalize to match your regressors’ expectations
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

print(MODEL, X.shape, X[:1, :5])
np.save("embeddings.npy", X)