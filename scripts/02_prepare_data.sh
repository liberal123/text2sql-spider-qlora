#!/usr/bin/env bash
set -euo pipefail
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH=src
python -m text2sql_finetune.prepare_data --spider-dir data/raw/spider --out-dir data/processed
