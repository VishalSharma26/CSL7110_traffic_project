$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment with Python 3.8..."
python -m venv .venv

Write-Host "Upgrading pip..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "Installing project requirements..."
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Running environment check..."
& ".\.venv\Scripts\python.exe" .\scripts\check_environment.py
