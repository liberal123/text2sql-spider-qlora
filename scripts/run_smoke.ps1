$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$sqlPython = if (Test-Path -LiteralPath '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
& $sqlPython -m text2sql_finetune.prepare_data --spider-dir data/raw/spider --out-dir data/processed_smoke --max-train 64 --max-valid 16 --max-test 16
if ($LASTEXITCODE -ne 0) { throw 'Smoke data preparation failed.' }
& $sqlPython -m text2sql_finetune.train_qlora --config configs/smoke_train.yaml
if ($LASTEXITCODE -ne 0) { throw 'Smoke training failed.' }
