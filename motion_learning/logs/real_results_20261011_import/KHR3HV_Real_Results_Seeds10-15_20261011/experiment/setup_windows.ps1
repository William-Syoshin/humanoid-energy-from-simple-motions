$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.11 with the Windows Python launcher first." }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& .\.venv\Scripts\python.exe -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
& .\.venv\Scripts\python.exe verify_bundle.py
if ($LASTEXITCODE -ne 0) { throw "Bundle integrity failed." }
Write-Host "Setup complete. Read README_JA.md before hardware execution."
