$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Environment is ready. Activate with: .\.venv\Scripts\Activate.ps1"
