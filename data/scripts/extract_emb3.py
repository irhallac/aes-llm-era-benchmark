# extract_nvembed_fixed.py
import os, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- Config ---
FILE_PATH   = "data/train_set_kaggle.csv"
MODEL_NAME  = "nvidia/NV-Embed-v2"
OUT_DIR     = "data/embeddings"
OUT_NAME    = "NV_Embed_v2_embeddings.npz"
BATCH_SIZE  = 16
MAX_LEN     = 4096
NORMALIZE   = True
SEED        = 36

torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# --- Data ---
cols = ["text_id","full_text","cohesion","syntax","vocabulary","phraseology","grammar","conventions"]
df = pd.read_csv(FILE_PATH)
miss = [c for c in cols if c not in df.columns]
if miss: raise ValueError(f"Dataset missing columns: {miss}")
df = df.dropna(subset=["full_text"]).reset_index(drop=True)

texts  = df["full_text"].tolist()
labels = df[["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]].values

# --- Model ---
print(f"\n🔍 Loading HF model: {MODEL_NAME} on {device}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    attn_implementation="eager",   # NV-Embed doesn't support SDPA
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts  # [batch, hidden_size] (constant across batches)

# --- Encode (force a single consistent pathway) ---
emb_chunks = []
emb_dim = None

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🚀 Encoding", ncols=80):
    batch = texts[i:i+BATCH_SIZE]
    enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)   # NV-Embed returns a dict

    # ---- use sentence-level output, normalize shape ----
    if not isinstance(out, dict) or "sentence_embeddings" not in out:
        raise ValueError(f"Unexpected output: {type(out)} / keys={list(out.keys()) if isinstance(out, dict) else 'n/a'}")

    se = out["sentence_embeddings"]                    # torch.Tensor
    # If model returns multiple segment/sentence embeddings per text: [B, S, D] -> mean over S
    if se.dim() == 3:
        pooled = se.mean(dim=1)                        # [B, D]
    elif se.dim() == 2:
        pooled = se                                    # [B, D]
    else:
        raise RuntimeError(f"sentence_embeddings has unexpected shape {tuple(se.shape)}")

    pooled = pooled.detach().float().cpu().numpy()

    # sanity: enforce constant D
    if emb_dim is None:
        emb_dim = pooled.shape[1]
        print(f"• embedding_dim = {emb_dim}")
    elif pooled.shape[1] != emb_dim:
        raise RuntimeError(f"Inconsistent dims: got {pooled.shape[1]} vs {emb_dim}")

    emb_chunks.append(pooled)

emb = np.concatenate(emb_chunks, axis=0)
if NORMALIZE:
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


# --- Save ---
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, OUT_NAME)
np.savez_compressed(out_path, embeddings=emb, labels=labels)

print(f"\n✅ Saved: {out_path}")
print(f"Embedding shape: {emb.shape} | Label shape: {labels.shape}")
