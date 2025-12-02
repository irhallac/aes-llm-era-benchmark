"""Compute MAE/QWK from prompt output CSVs.

Usage:
    python tools/eval_prompt_csv.py --predictions src/prompting/runs/prompt_phi4_2025....csv \
        --reference data/test_sample_kaggle.csv --output results/prompt_eval_phi4.json

The predictions CSV must contain the original essay rows plus the columns:
    cohesion_pred, syntax_pred, vocabulary_pred, phraseology_pred,
    grammar_pred, conventions_pred.
The reference CSV needs the ground-truth columns with the same names.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

VALID_GRADES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
TRAITS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]


def round_to_valid(values):
    values = np.asarray(values)
    idx = np.abs(values[:, None] - VALID_GRADES[None, :]).argmin(axis=1)
    return VALID_GRADES[idx]


def quadratric_weighted_kappa(y_true, y_pred):
    # convert to bins before calling sklearn's cohen_kappa_score
    grade_to_bin = {grade: idx for idx, grade in enumerate(VALID_GRADES)}
    true_bins = [grade_to_bin[val] for val in y_true]
    pred_bins = [grade_to_bin[val] for val in y_pred]
    if len(set(true_bins)) == 1 or len(set(pred_bins)) == 1:
        return 0.0
    return cohen_kappa_score(true_bins, pred_bins, weights="quadratic")


def evaluate(pred_path, ref_path):
    preds = pd.read_csv(pred_path)
    refs = pd.read_csv(ref_path)

    merged = preds.merge(refs[["text_id"] + TRAITS], on="text_id", suffixes=("_pred", ""))

    summary = {"file": pred_path, "per_trait": {}}
    all_true = []
    all_pred = []

    for trait in TRAITS:
        y_true = merged[trait].values
        y_pred = merged[f"{trait}_pred"].values
        y_pred = round_to_valid(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        qwk = quadratric_weighted_kappa(y_true, y_pred)
        summary["per_trait"][trait] = {"MAE": float(mae), "QWK": float(qwk)}
        all_true.append(y_true)
        all_pred.append(y_pred)

    all_true = np.stack(all_true, axis=1).reshape(-1)
    all_pred = np.stack(all_pred, axis=1).reshape(-1)
    summary["overall"] = {
        "MAE": float(mean_absolute_error(all_true, all_pred)),
        "QWK": float(quadratric_weighted_kappa(all_true, all_pred)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with prompt outputs")
    parser.add_argument("--reference", required=True, help="CSV with reference scores")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    summary = evaluate(args.predictions, args.reference)
    print(json.dumps(summary, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
