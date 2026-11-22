@echo off
cd /d "%~dp0"
if not exist .env copy .env.example .env >nul
py -3.13 --version >nul 2>nul
if %errorlevel%==0 (py -3.13 start.py %*) else (py -3.12 --version >nul 2>nul && (py -3.12 start.py %*) || (python start.py %*))
pause
