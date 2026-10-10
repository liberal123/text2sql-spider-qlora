#!/usr/bin/env bash
set -euo pipefail
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH=src
python examples/demo.py --output-dir examples/demo_databases
python -m text2sql_finetune.serve --demo-mode --db-root examples/demo_databases --host 127.0.0.1 --port 8000
