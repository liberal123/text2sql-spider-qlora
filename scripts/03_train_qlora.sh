#!/usr/bin/env bash
set -euo pipefail
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH=src
python -m accelerate.commands.launch --num_processes 4 --module text2sql_finetune.train_qlora --config configs/train_qwen25_coder_1_5b_qlora.yaml
