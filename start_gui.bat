@echo off
rem Double-click to launch the GUI
cd /d %~dp0
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw main.py gui
) else (
    python main.py gui
)
