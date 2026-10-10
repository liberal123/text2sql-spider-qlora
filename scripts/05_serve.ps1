$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$sqlPython = if (Test-Path -LiteralPath '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
& $sqlPython -m text2sql_finetune.serve --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter-path artifacts/adapters/qwen25-coder-1.5b-spider-qlora --db-root data/raw/spider/database --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'Service failed.' }
