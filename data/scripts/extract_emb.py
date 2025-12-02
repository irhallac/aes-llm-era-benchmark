import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load dataset
file_path = "data/train_set_kaggle.csv"
data = pd.read_csv(file_path)

# ✅ Ensure expected columns exist
expected_columns = ["text_id", "full_text", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
if not all(col in data.columns for col in expected_columns):
    raise ValueError(f"Dataset is missing required columns. Expected: {expected_columns}")

# ✅ Drop rows with missing text
data = data.dropna(subset=["full_text"]).reset_index(drop=True)

# ✅ Extract text data + labels
texts = data["full_text"].tolist()
labels = data[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].values

# 🔄 Choose one model at a time
#MODEL_NAME = "BAAI/bge-m3"
#MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
#MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
#MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
# MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_NAME = "nvidia/NV-Embed-v2"

print(f"\n🔍 Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# ✅ Extract embeddings
print("\n🚀 Extracting embeddings...")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    batch_size=16,               # adjust if OOM
    show_progress_bar=True,
    normalize_embeddings=False   # keep raw, normalize later if needed
)

# ✅ Optional: normalize for regression models
embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

# ✅ Save embeddings + labels
model_short = MODEL_NAME.split("/")[-1].replace("-", "_")
output_path = f"data/embeddings/{model_short}_embeddings.npz"
np.savez_compressed(output_path, embeddings=embeddings, labels=labels)

print(f"\n✅ Saved embeddings to {output_path}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}")