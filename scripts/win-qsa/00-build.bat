@echo off
setlocal
call "%~dp0_config.bat"

rem the backend flags only turn a backend on, so a cache made for another one would keep
rem it, and its dlls would stay in bin. start over when the backend changed
if not exist "%BUILD%\CMakeCache.txt" goto :configure
set "HAVE=cpu"
findstr /b /i /c:"GGML_VULKAN:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=vulkan"
findstr /b /i /c:"GGML_CUDA:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=cuda"
if /i "%HAVE%"=="%BACKEND%" goto :configure
echo %BUILD% was configured for %HAVE%, wanted %BACKEND%: removing it
rmdir /s /q "%BUILD%"

:configure
echo building into %BUILD%, backend %BACKEND% %CMAKE_BACKEND%
if /i "%BACKEND%"=="cpu" echo   no cuda toolkit and no vulkan sdk found, this is a cpu only build
cmake -S "%~dp0..\.." -B "%BUILD%" %CMAKE_BACKEND% -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
if not "%ERRORLEVEL%"=="0" exit /b 1

cmake --build "%BUILD%" --config Release -j %NUMBER_OF_PROCESSORS%
if not "%ERRORLEVEL%"=="0" exit /b 1

rem the binaries did not exist when _config.bat ran, resolve again
set "BIN=%BUILD%\bin\Release"
if not exist "%BIN%" set "BIN=%BUILD%\bin"

echo.
echo binaries in %BIN%
for %%f in (llama-bench.exe llama-perplexity.exe test-backend-ops.exe) do (
    if exist "%BIN%\%%f" (echo   ok      %%f) else (echo   MISSING %%f)
)
