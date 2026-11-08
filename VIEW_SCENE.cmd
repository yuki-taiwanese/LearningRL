@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" tools\preview_box_transfer.py --live
if errorlevel 1 pause
