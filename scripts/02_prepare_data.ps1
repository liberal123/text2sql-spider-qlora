$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$sqlPython = if (Test-Path -LiteralPath '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
& $sqlPython -m text2sql_finetune.prepare_data --spider-dir data/raw/spider --out-dir data/processed
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
