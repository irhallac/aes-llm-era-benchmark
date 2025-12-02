import pandas as pd
from transformers import AutoTokenizer

# 🔄 Choose one model at a time
#MODEL_NAME = "BAAI/bge-m3"
# MODEL_NAME = "nvidia/NV-Embed-v2"
#MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
# MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
# MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

print(f"\n🔍 Loading tokenizer for: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Load dataset (only need text column)
file_path = "projects/paper_versions/aes_ipm_r1/data/train_set_kaggle.csv"
data = pd.read_csv(file_path).dropna(subset=["full_text"]).reset_index(drop=True)

# ✅ Compute token counts
max_len = 0
max_idx = -1
for i, text in enumerate(data["full_text"]):
    tokens = tokenizer(text, truncation=False)["input_ids"]
    if len(tokens) > max_len:
        max_len = len(tokens)
        max_idx = i

print(f"\n✅ Longest essay has {max_len} tokens (index {max_idx})")
print(f"Model supports up to {tokenizer.model_max_length} tokens")
