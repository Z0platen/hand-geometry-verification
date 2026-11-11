@echo off
setlocal
set "APP_ROOT=%~dp0"
set "APP_PYTHON=%APP_ROOT%.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
    echo The project Python environment is missing.
    echo Run the setup instructions in README.md to create .venv first.
    pause
    exit /b 1
)
cd /d "%APP_ROOT%"
"%APP_PYTHON%" "%APP_ROOT%app.py"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo The launcher stopped with error %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%
