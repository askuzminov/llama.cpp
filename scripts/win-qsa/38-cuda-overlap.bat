@echo off
setlocal
rem Use the same server runner for greedy checks and the performance comparison.
call "%~dp023-spec-auto.bat" overlap-quality
if errorlevel 1 (
    echo The overlap checks failed. See the 23-spec-auto logs.
    echo The speed comparison was skipped.
    exit /b 1
)
call "%~dp023-spec-auto.bat" overlap-speed
exit /b %ERRORLEVEL%
