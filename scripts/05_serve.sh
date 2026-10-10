#!/usr/bin/env bash
set -euo pipefail
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
export PYTHONPATH=src
python -m text2sql_finetune.serve \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter-path artifacts/adapters/qwen25-coder-1.5b-spider-qlora \
  --db-root data/raw/spider/database \
  --host 127.0.0.1 \
  --port 8000
