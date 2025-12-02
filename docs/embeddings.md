# Frozen Embedding Regression

The ML pipeline evaluates two score types for each embedding family:

| Script | Score type |
| --- | --- |
| `src/ml/per_trait/llama_per_trait.py` | Per-trait regressors for frozen LLaMA embeddings (`data/embeddings/embeddings_and_labels_llama32*_maxlen_*.npz`). |
| `src/ml/per_trait/transformer_per_trait.py` | Per-trait regressors for RoBERTa/MiniLM/MPNet. Set `EMBEDDING_TYPE` inside the script. |
| `src/ml/per_trait/doc2vec_per_trait.py` | Per-trait regressors with Doc2Vec embeddings learned per fold. |
| `src/ml/holistic/holistic_svr.py` | Holistic SVR trained on the mean of the six traits using any embedding file. |

Edit the `config` block in each script with your local paths, run it, and collect
the MAE/QWK outputs under `src/ml/results/`. Summaries can be generated with
`tools/eval_embeddings_excel.py`.
