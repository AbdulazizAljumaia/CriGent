@echo off
REM Launches CriGent. The app starts Ollama itself if it isn't already running.
cd /d "%~dp0"
start "" pythonw.exe "%~dp0crigent.py"
