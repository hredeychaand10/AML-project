#!/bin/sh
# Runs the full pipeline: synthetic data -> features -> train/evaluate
set -e
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
python3 generate_data.py      # writes customers.csv, raw_transactions.csv
python3 build_features.py     # writes features.csv
python3 train_model.py        # writes model_evaluation.png, scored_transactions.csv
