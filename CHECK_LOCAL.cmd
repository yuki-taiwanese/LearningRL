@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" tools\check_local.py
if errorlevel 1 (
  echo Local checks failed. Please copy the error output for review.
) else (
  echo All local checks passed. GPU training is performed by the TA.
)
pause
