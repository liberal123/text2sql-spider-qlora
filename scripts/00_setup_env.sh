#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
echo "Environment is ready. Activate with: source .venv/bin/activate"
