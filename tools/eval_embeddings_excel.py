"""Extract MAE/QWK from the Excel summaries produced by per-aspect ML runs."""

import argparse
import json
import os

import pandas as pd


def evaluate(excel_path):
    xl = pd.ExcelFile(excel_path)
    per_trait = pd.read_excel(xl, sheet_name=0)  # first sheet assumed per-trait
    overall = pd.read_excel(xl, sheet_name=1)    # second sheet overall summary

    summary = {
        'file': excel_path,
        'per_trait': {},
        'overall': {}
    }
    for _, row in per_trait.iterrows():
        summary['per_trait'].setdefault(row['Model'], {})[row['Dimension']] = {
            'MAE': float(row['Avg MAE']),
            'QWK': float(row['Avg QWK'])
        }
    for _, row in overall.iterrows():
        summary['overall'][row['Model']] = {
            'MAE': float(row['Avg MAE']),
            'QWK': float(row['Avg QWK'])
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', required=True, help='Path to the per-aspect results Excel file')
    parser.add_argument('--output', help='Optional JSON output path')
    args = parser.parse_args()

    summary = evaluate(args.excel)
    print(json.dumps(summary, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
