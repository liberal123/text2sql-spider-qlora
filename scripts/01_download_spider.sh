#!/usr/bin/env bash
set -euo pipefail
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
export PYTHONPATH=src
python src/text2sql_finetune/download_spider.py --out-dir data/raw
