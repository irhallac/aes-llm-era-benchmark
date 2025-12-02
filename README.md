# AES LLM Era Benchmark

This repository stores the code, configs, and supporting artifacts for the study
**"From Prompting to Fine-Tuning: A Comprehensive Evaluation of LLM Strategies for Automatic Essay Assessment".**

The goal of this repo is twofold:
1. Preserve the exact configurations and metrics that produced the preprint
   (see `reference_runs/`).
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
reference_runs/configs/         # read-only reference configs
docs/fine_tuning.md             # quick-start notes
tools/eval_finetune_cv.py       # summarize per-fold metrics
```

**Frozen embeddings + classical ML – `src/ml/`**
```
src/ml/per_trait/llama_per_trait.py         # LLaMA embedding regressors
src/ml/per_trait/transformer_per_trait.py   # RoBERTa/MiniLM/MPNet (toggle `EMBEDDING_TYPE`)
src/ml/per_trait/doc2vec_per_trait.py       # doc2vec baseline (learns embeddings per fold)
src/ml/holistic/holistic_svr.py             # holistic SVR (average of six traits)
data/embeddings/                    # .npz files (Git LFS)
data/scripts/                       # notebooks for regenerating embeddings
docs/embeddings.md                  # quick-start notes
tools/eval_embeddings_excel.py      # read the Excel summaries
```

General assets:
```
reference_runs/results/         # appendix tables (read-only)
config.yaml                     # current working config
requirements.txt                # Python dependencies
docs/                           # all short guides
```

---

## Dataset & External Assets

- **Feedback Prize – English Language Learning dataset**: download from Kaggle
  (https://www.kaggle.com/competitions/feedback-prize-english-language-learning).
  The scripts expect the CSV `data/train_set_kaggle.csv`. See `config.yaml` for
  alternative stratified subsets if you want quick smoke tests.
- **Model weights**: Meta LLaMA weights are *not* distributed here. Request
  access from Meta and place the checkpoints under `downloaded_models/` (or
  update the config paths accordingly).
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
   - Copy `reference_runs/configs/finetune_llama_top6.yaml` *or*
     `configs/templates/finetune_llama_template.yaml` to `configs/private/`
     and adjust file paths.
   - Run `python src/fine_tuning/finetune_llama_kfold.py` for the per-trait CV and, for the
     holistic model, `python src/fine_tuning/overall_score_v1_fold_N.py --target_fold N`.
   - See `docs/fine_tuning.md` for the quick checklist.

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
     - `holistic/holistic_svr.py` for the holistic SVR baseline (extend it to other regressors if desired)
   - Outputs (CSV/XLSX) are placed under `results/` as described in `docs/embeddings.md`.

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

## Roadmap

- [x] Publish reference configs + appendix tables.
- [ ] Finalize `requirements.txt` / conda `environment.yml`.
- [ ] Add public-friendly config templates and README instructions for dataset
      preparation.
- [ ] Introduce regression tests / helper scripts for automated validation.

Feel free to open issues or reach out (ibrahimrizahallac@live.com) if you’re
interested in contributing or have trouble reproducing the results.
- Git LFS is used for the `.npz` embedding files under `data/embeddings/`. Install
  LFS (`git lfs install`) before cloning or pulling to ensure those artifacts are
  downloaded correctly.
