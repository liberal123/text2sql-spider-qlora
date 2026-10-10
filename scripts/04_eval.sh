#!/usr/bin/env bash
set -euo pipefail
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
export PYTHONPATH=src
python -m text2sql_finetune.evaluate \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter-path artifacts/adapters/qwen25-coder-1.5b-spider-qlora \
  --test-file data/processed/test.jsonl \
  --spider-dir data/raw/spider \
  --repair-on-error --max-repair-attempts 1 \
  --out-file artifacts/predictions/test_predictions.jsonl \
  --metrics-file artifacts/predictions/test_metrics.json
python src/text2sql_finetune/analyze_errors.py \
  --predictions artifacts/predictions/test_predictions.jsonl \
  --out-csv artifacts/predictions/error_analysis_by_difficulty.csv
