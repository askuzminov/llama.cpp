@echo off
setlocal
call "%~dp0_config.bat"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0_compare.py" prepare
    exit /b
)
where python >nul 2>&1
if errorlevel 1 (
    echo Python 3 is required for the paired tests.
    exit /b 1
)
python "%~dp0_compare.py" prepare
exit /b %ERRORLEVEL%
