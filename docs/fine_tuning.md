# LLaMA Fine-Tuning (Per-Trait + Holistic)

1. Copy `configs/templates/finetune_llama_template.yaml` to a working location
   (e.g., `configs/finetune_pertrait.yaml`) and adjust:
   - `model.path`: location of your LLaMA-3.2-1B weights (base or instruct).
   - `training_strategy`: `top_n_layers`, `last_layer_only`, etc.
   - `use_prompt`: `true` to prepend the rubric prompt (`prompts/fine_tune_per_trait.txt` for multi-trait runs or `prompts/fine_tune_holistic.txt` for the overall-score runs).
   - `training_mode`: `per_trait` (default) or `holistic` (single-score head).

2. Run the per-trait cross-validation:
   ```bash
   python src/fine_tuning/finetune_llama_kfold.py --config configs/finetune_pertrait.yaml
   ```
   Outputs (per fold): metrics saved under `checkpoints/<run>/fold_<k>/`, plus a
   summary CSV and `cv_results.txt` with per-trait MAE/QWK.

3. For holistic runs, switch the config to `training_mode: holistic` (or point to
   `config_overall.yaml` if you have a separate file) and execute:
   ```bash
   python src/fine_tuning/overall_score_v1_fold_N.py --config configs/finetune_holistic.yaml --target_fold 3
   ```
   Repeat for folds 1–5 to populate the holistic table.

Both scripts share the same logging pattern (timestamped folders, `used_config.yaml`),
so you can compare the per-trait vs. overall jobs directly.
