@echo off
rem Starts the full demo (3D twin + photo detector) and opens it in the browser when ready.
rem Needs the frontend-full folder next to this backend folder. Keep this window open.
cd /d "%~dp0"
".venv\Scripts\python.exe" server.py --port 8765 --open
pause
