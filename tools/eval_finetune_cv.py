"""Summarize per-trait MAE/QWK from finetune checkpoint folders.

Usage:
  python tools/eval_finetune_cv.py --run checkpoints/Meta-Llama-3.2-1B_top_n_layers_6_prompt_yes_2025-...

The script expects `fold_<k>/cv_summary.csv` or `fold_results.csv` files generated
by `src/finetune_llama_kfold.py`. It aggregates them into a single JSON/console
summary for quick comparison with the paper tables.
"""

import argparse
import json
import os
import glob

import pandas as pd


def load_fold_metrics(fold_dir):
    summary_path = os.path.join(fold_dir, 'cv_summary.csv')
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        return df
    results_path = os.path.join(fold_dir, 'fold_results.csv')
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        return df
    raise FileNotFoundError(f"No summary CSV found in {fold_dir}")


def aggregate(run_dir):
    folds = sorted(glob.glob(os.path.join(run_dir, 'fold_*')))
    per_trait_mae = {}
    per_trait_qwk = {}
    overall_mae = []
    overall_qwk = []

    for fold in folds:
        df = load_fold_metrics(fold)
        if 'Test MAE per Feature.cohesion' in df.columns:
            # new layout: single row with multi-columns
            row = df.iloc[0]
            for col in df.columns:
                if col.startswith('Test MAE per Feature.'):
                    trait = col.split('.', 1)[1]
                    per_trait_mae.setdefault(trait, []).append(row[col])
                if col.startswith('Test QWK per Feature.'):
                    trait = col.split('.', 1)[1]
                    per_trait_qwk.setdefault(trait, []).append(row[col])
            overall_mae.append(row.get('Test MAE Overall', float('nan')))
            overall_qwk.append(row.get('Test QWK', row.get('Test QWK Overall', float('nan'))))
        else:
            # legacy layout: one row per metric per fold
            if 'Metric' in df.columns:
                for _, row in df.iterrows():
                    if row['Metric'] == 'Avg MAE':
                        overall_mae.append(row['Overall'])
            if 'Val MAE per Feature' in df.columns:
                continue

    summary = {
        'run_dir': run_dir,
        'overall_mae': float(pd.Series(overall_mae).mean()),
        'overall_qwk': float(pd.Series(overall_qwk).mean()),
        'per_trait': {trait: {
            'MAE': float(pd.Series(values).mean()),
            'QWK': float(pd.Series(per_trait_qwk.get(trait, [])).mean())
        } for trait, values in per_trait_mae.items()}
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Checkpoint directory for a finetune job')
    parser.add_argument('--output', help='Optional JSON output path')
    args = parser.parse_args()

    summary = aggregate(args.run)
    print(json.dumps(summary, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
