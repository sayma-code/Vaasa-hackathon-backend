@echo off
rem Runs the website on this computer without the detector: the 3D twin and walk-inside view only,
rem exactly as it appears on GitHub. Needs the frontend-full folder next to this backend folder.
rem Keep this window open while using it.
cd /d "%~dp0"
start "" http://localhost:8766/
".venv\Scripts\python.exe" -m http.server 8766 --directory ..rontend-full --bind 127.0.0.1
pause
