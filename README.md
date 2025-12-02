# AES LLM Era Benchmark

This repository stores the code, configs, and supporting artifacts for the study
**"From Prompting to Fine-Tuning: A Comprehensive Evaluation of LLM Strategies for Automatic Essay Assessment".**

The goal of this repo is twofold:
1. Preserve the exact configurations and metrics that produced the preprint
   (see `docs/reference_results/`).
2. Provide a clean starting point for practitioners who want to reproduce or
   extend the prompting, fine-tuning, and embedding-based pipelines described in
   the paper.

---

## Project Overview

We evaluate three families of solutions for multi-trait automated essay scoring
using the Feedback Prize – English Language Learning dataset (3,911 essays, six
analytic traits scored from 1.0 to 5.0 in 0.5 increments):

1. **Prompting** – zero/few-shot prompting of open-weight LLMs via structured
   instructions (`src/prompting/run_prompt_eval.py` + templates).
2. **Fine-tuning** – parameter-efficient tuning of LLaMA-3.2-1B models for
   multi-output regression (`src/fine_tuning/finetune_llama_kfold.py`) and a
   holistic variant (`src/fine_tuning/overall_score_v1_fold_N.py`).
3. **Frozen embeddings + regressors** – extracting contextual embeddings (LLaMA,
   RoBERTa, MiniLM, MPNet, Doc2Vec) and training classical regressors or a small
   ANN (scripts under `src/ml/`).

Each pipeline shares a rubric-aware evaluation protocol: predictions are rounded
into the valid set {1.0, 1.5, …, 5.0} before computing per-trait and holistic
Quadratic Weighted Kappa (QWK) and Mean Absolute Error (MAE).

---

## Repository Layout (by experiment branch)

**Prompting (zero/few-shot) – `src/prompting/`**
```
src/prompting/
  ├── run_prompt_eval.py        # queries Ollama/OpenAI + saves CSV predictions
  ├── config_prompt_template.yaml
  └── runs/                     # generated CSVs
prompts/                        # Prompt templates for fine-tuning + prompting scripts
tools/eval_prompt_csv.py        # compute MAE/QWK from prompt CSVs
```

**Fine-tuning LLaMA – `src/fine_tuning/`**
```
src/fine_tuning/
  ├── finetune_llama_kfold.py
  └── overall_score_v1_fold_N.py
configs/templates/              # copy & edit for new runs
docs/fine_tuning.md             # quick-start notes
tools/eval_finetune_cv.py       # summarize per-fold metrics
```

**Frozen embeddings + classical ML – `src/ml/`**
```
src/ml/per_trait/llama_per_trait.py         # LLaMA embedding regressors
src/ml/per_trait/transformer_per_trait.py   # RoBERTa/MiniLM/MPNet (toggle `EMBEDDING_TYPE`)
src/ml/per_trait/doc2vec_per_trait.py       # doc2vec baseline (learns embeddings per fold)
src/ml/holistic/holistic_svr.py             # holistic SVR (average of six traits)
data/scripts/                               # helpers for generating embeddings
docs/embeddings.md                          # quick-start notes
tools/eval_embeddings_excel.py              # read the Excel summaries
```
The `.npz/.npy` embedding files referenced by these scripts can be recreated
using the notebooks and utilities stored under `data/scripts/` (see
`docs/embeddings.md` for guidance).

General assets:
- `docs/reference_results/` – appendix tables (read-only).
- `docs/` – short guides for each branch.
- `data/README.md` – describes where to place Kaggle CSVs + generated embeddings.
- `config.yaml` / `config_debug.yaml` – working configs (copy before editing).
- `requirements.txt` – Python dependencies.

---

## Dataset & External Assets

- **Feedback Prize – English Language Learning dataset**: download from Kaggle
  (https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
  and place the CSVs under `data/` (see `data/README.md` for the expected layout).
- **Model weights**: Meta LLaMA weights are *not* distributed here. Request
  access from Meta and place the checkpoints under `./models/` (or update the
  config paths).
- **Prompt examples**: stored in `prompts/`.

Because these assets are subject to third-party licenses, the repo only contains
configuration placeholders—users must obtain the models/data separately.

---

## Environment Setup

1. **Create/activate** a Python 3.10+ environment (conda or venv).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) configure Weights & Biases if you plan to enable `wandb_on: true` in
   the fine-tuning configs.

The repo does not install LLaMA weights or third-party embeddings—place them in
the paths referenced by the configs before running.

---

## Reproducing the Preprint

1. **Fine-tuning**
   - Copy `configs/templates/finetune_llama_template.yaml` to `configs/private/` and
     adjust the dataset/model/prompt paths.
   - Run `python src/fine_tuning/finetune_llama_kfold.py --config configs/private/<file>.yaml`
     for the per-trait CV and `python src/fine_tuning/overall_score_v1_fold_N.py --config ...`
     for the holistic folds.
   - See `docs/fine_tuning.md` for the checklist and compare outputs to
     `docs/reference_results/Appendix_B2_Finetuning_Full_Results.xlsx`.

2. **Prompting**
   - Use `src/prompting/run_prompt_eval.py --config src/prompting/config_prompt_template.yaml`
     after editing the template with your data paths/LLM choice.
   - Store the generated CSVs and feed them into the evaluation notebooks /
     metrics scripts to recreate the supplementary prompt table.
   - See `docs/prompting.md` for a short checklist.

3. **Frozen embeddings**
   - Use the scripts in `src/ml/`:
     - `per_trait/llama_per_trait.py` for LLaMA embeddings (`data/embeddings/embeddings_and_labels_llama32...npz`)
     - `per_trait/transformer_per_trait.py` for RoBERTa/MiniLM/MPNet (set `EMBEDDING_TYPE`)
     - `per_trait/doc2vec_per_trait.py` for doc2vec (learns representations on the fly)
     - `holistic/holistic_svr.py` for the holistic SVR baseline
   - Outputs (CSV/XLSX) are written to `ml_results/` (ignored by git). Compare
     against `docs/reference_results/Appendix_B3_*.xlsx`.

> 📝 A lightweight regression test (on the stratified subsets) will be added
> during the release hardening to catch deviations introduced by future refactors.

---

## Evaluation Helpers

Use the scripts under `tools/` to convert raw outputs to MAE/QWK summaries:

- `python tools/eval_prompt_csv.py --predictions <csv> --reference <csv>`
- `python tools/eval_finetune_cv.py --run <checkpoint_dir>`
- `python tools/eval_embeddings_excel.py --excel <results.xlsx>`

Each prints a JSON report matching the tables in the paper.

---

## Supplementary Material

All trait-level tables cited in the paper live under `docs/reference_results/`.
They are unchanged copies of Appendix B (prompting, fine-tuning, frozen
embeddings). Use them as the canonical targets when validating new runs.

## LLaMA Licensing Notice

This repository includes fine-tuning and evaluation scripts for Meta’s LLaMA
models. These models remain governed by Meta’s license. **We do not redistribute
the weights.** You must request access directly from Meta:
https://ai.meta.com/resources/models-and-libraries/llama-downloads/

## License & Contact

Code in this repository is released under the MIT License; external models and
datasets retain their original terms. Feel free to open issues or reach out at
ibrahimrizahallac@live.com if you have questions or would like to collaborate.
