$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$sqlPython = if (Test-Path -LiteralPath '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
& $sqlPython examples/demo.py --output-dir examples/demo_databases
if ($LASTEXITCODE -ne 0) { throw 'Demo database preparation failed.' }
& $sqlPython -m text2sql_finetune.serve --demo-mode --db-root examples/demo_databases --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'Demo service failed.' }
