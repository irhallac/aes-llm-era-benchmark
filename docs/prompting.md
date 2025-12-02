# Prompt-Based Evaluation

This note lists the prompt-specific steps; general environment details live in
the root README.

## Quick Steps

1. Edit `src/prompting/config_prompt_template.yaml` (or copy it) with your data
   paths and model choice (`model.type`, `model.*.model_name`, API endpoint).
2. Generate predictions:
   ```bash
   python src/prompting/run_prompt_eval.py --config src/prompting/config_prompt_template.yaml
   ```
3. The script writes a timestamped CSV under `src/prompting/runs/` containing the
   six predictions, inference time, and raw model response per essay. Feed this
   CSV to the evaluation notebook to compute MAE/QWK.

## Notes

- **Ollama**: the script directly calls the Ollama REST endpoint; ensure the
  server is running and the `api_url` points to it.
- **OpenAI GPTs**: set `model.type: gpt` and export `OPENAI_API_KEY` (or the
  env variable named in `api_key_env`) before running the script.
- Holistic scores are obtained by averaging the six trait predictions per essay.
