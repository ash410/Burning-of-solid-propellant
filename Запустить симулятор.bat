@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0app.py"
    goto finished
)

where python >nul 2>&1
if not errorlevel 1 (
    python "%~dp0app.py"
    goto finished
)

echo Python 3 was not found.
echo Install Python from https://www.python.org/downloads/ and try again.
pause
exit /b 1

:finished
set "STATUS=%ERRORLEVEL%"

if not "%STATUS%"=="0" (
    echo.
    echo Failed to start the simulator. Error code: %STATUS%
    echo Install dependencies with: py -3 -m pip install -r requirements.txt
    echo.
    pause
)

exit /b %STATUS%
