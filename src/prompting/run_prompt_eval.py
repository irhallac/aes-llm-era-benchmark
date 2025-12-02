import csv
import os
import re
import time
from datetime import datetime

import requests
import yaml

try:
    import openai
except ImportError:  # optional dependency when only using Ollama
    openai = None


def load_few_shot_examples(path):
    if not path:
        return ""
    import pandas as pd

    examples = pd.read_csv(path)
    blocks = []
    for _, row in examples.iterrows():
        scores = f"{row['cohesion']} {row['syntax']} {row['vocabulary']} {row['phraseology']} {row['grammar']} {row['conventions']}"
        blocks.append(
            f'Eassy:\n"""{row["full_text"]}"""\nScores: {scores}\n'
        )
    return "\n".join(blocks)


def build_prompt(essay_text, few_shot_block=None):
    examples = f"Here are example essays:\n{few_shot_block}\n---" if few_shot_block else ""
    return (
        "You are an expert English essay evaluator. Score the essay on Cohesion, "
        "Syntax, Vocabulary, Phraseology, Grammar, and Conventions. Valid scores "
        "are 1.0-5.0 in 0.5 increments.\n"
        f"{examples}\nEssay:\n\"\"\"{essay_text}\"\"\"\n---\n"
        "Return six numbers separated by spaces in the same order as the traits."
    )


def call_model(prompt, cfg):
    model_type = cfg['model']['type']
    start = time.time()
    if model_type == 'ollama':
        data = {"model": cfg['model']['ollama']['model_name'], "prompt": prompt, "stream": False}
        resp = requests.post(cfg['model']['ollama']['api_url'], json=data, timeout=120)
        resp.raise_for_status()
        output = resp.json().get('response', '').strip()
    elif model_type == 'gpt':
        if openai is None:
            raise RuntimeError("openai package required for GPT mode")
        api_key_env = cfg['model']['gpt'].get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} not set")
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=cfg['model']['gpt']['model_name'],
            messages=[
                {"role": "system", "content": "You are a strict essay grader."},
                {"role": "user", "content": prompt},
            ],
        )
        output = response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return output, time.time() - start


def parse_scores(raw_output):
    scores = re.findall(r"\d+\.\d", raw_output)
    if len(scores) != 6:
        return None
    return [float(s) for s in scores]


def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output']['dir'], exist_ok=True)
    few_shot_block = None
    if cfg['data'].get('use_few_shot'):
        few_shot_block = load_few_shot_examples(cfg['data']['few_shot_file'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = cfg['model'][cfg['model']['type']]['model_name']
    safe_model = model_name.replace(':', '_').replace('/', '_')
    out_name = f"prompt_{safe_model}_{timestamp}.csv"
    out_path = os.path.join(cfg['output']['dir'], out_name)

    with open(cfg['data']['test_file'], 'r', encoding='utf-8') as infile, \
            open(out_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [
            'cohesion_pred', 'syntax_pred', 'vocabulary_pred',
            'phraseology_pred', 'grammar_pred', 'conventions_pred',
            'inference_time_s', 'raw_response'
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            essay = row['full_text']
            prompt = build_prompt(essay, few_shot_block)
            try:
                raw_output, latency = call_model(prompt, cfg)
                scores = parse_scores(raw_output)
                if scores:
                    (row['cohesion_pred'], row['syntax_pred'], row['vocabulary_pred'],
                     row['phraseology_pred'], row['grammar_pred'], row['conventions_pred']) = scores
                    row['inference_time_s'] = round(latency, 4)
                else:
                    row['raw_response'] = raw_output
                    writer.writerow(row)
                    continue
                row['raw_response'] = raw_output
                if cfg['output'].get('redact_full_text', True):
                    row['full_text'] = '<redacted>'
                writer.writerow(row)
            except Exception as exc:
                row['raw_response'] = f"ERROR: {exc}"
                writer.writerow(row)

    print(f"Saved prompt outputs to {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prompt-based AES evaluation')
    parser.add_argument('--config', default='src/prompting/config_prompt_template.yaml', help='Path to YAML config')
    args = parser.parse_args()

    main(args.config)
