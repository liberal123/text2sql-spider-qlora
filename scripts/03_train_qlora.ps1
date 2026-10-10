$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$sqlPython = if (Test-Path -LiteralPath '.venv/Scripts/python.exe') { '.venv/Scripts/python.exe' } else { 'python' }
& $sqlPython -m accelerate.commands.launch --num_processes 4 --module text2sql_finetune.train_qlora --config configs/train_qwen25_coder_1_5b_qlora.yaml
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
