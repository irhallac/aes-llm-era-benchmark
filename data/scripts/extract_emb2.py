# extract_emb_gte_qwen2.py — Qwen-family fallback compatible with older HF

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# === unchanged dataset path & handling ===
file_path = "data/train_set_kaggle.csv"
data = pd.read_csv(file_path)
expected_columns = ["text_id", "full_text", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
if not all(col in data.columns for col in expected_columns):
    raise ValueError(f"Dataset is missing required columns. Expected: {expected_columns}")
data = data.dropna(subset=["full_text"]).reset_index(drop=True)
texts = data["full_text"].tolist()
labels = data[["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]].values

# === only necessary change: use a Qwen2 model that old HF can load ===
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"

print(f"\n🔍 Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # slow tokenizer = safer on older stacks
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

print("\n🚀 Extracting embeddings...")
all_embeddings = []
bs = 8
with torch.no_grad():
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = mean_pool(last, enc["attention_mask"])
        all_embeddings.append(pooled.detach().cpu().numpy())

embeddings = np.vstack(all_embeddings)

# === normalize & save (unchanged) ===
embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
out_path = "data/embeddings/gte_Qwen2_7B_instruct_embeddings.npz"
np.savez(out_path, embeddings=embeddings, labels=labels)

print(f"\n✅ Saved embeddings to {out_path}")
print(f"Total: {embeddings.shape[0]}  Dim: {embeddings.shape[1]}")
