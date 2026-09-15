@echo off
setlocal
call "%~dp0_config.bat"

echo building into %BUILD%
cmake -S "%~dp0..\.." -B "%BUILD%" %CMAKE_BACKEND% -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build "%BUILD%" --config Release -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1

rem the binaries did not exist when _config.bat ran, resolve again
set "BIN=%BUILD%\bin\Release"
if not exist "%BIN%" set "BIN=%BUILD%\bin"

echo.
echo binaries in %BIN%
for %%f in (llama-bench.exe llama-perplexity.exe test-backend-ops.exe) do (
    if exist "%BIN%\%%f" (echo   ok      %%f) else (echo   MISSING %%f)
)
