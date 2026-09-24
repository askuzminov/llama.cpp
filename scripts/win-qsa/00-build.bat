@echo off
setlocal
call "%~dp0_config.bat"

rem msbuild builds cuda through the toolkit's own integration files, and the cuda installer
rem copies them only into a visual studio it finds at install time. without them cmake says
rem "No CUDA toolset found", even with nvcc on PATH. ninja does not need them
set "GENARG="
if /i not "%GENERATOR%"=="auto" goto :gen_pin
if /i not "%BACKEND%"=="cuda" goto :gen_done
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :gen_done
set "VSPATH="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH goto :gen_done
dir /b /s "%VSPATH%\MSBuild\Microsoft\VC\CUDA *.props" >nul 2>&1
if not errorlevel 1 goto :gen_done
echo cuda integration for msbuild is missing in %VSPATH%
where ninja >nul 2>&1
if errorlevel 1 goto :gen_nomsbuild
where cl >nul 2>&1
if errorlevel 1 goto :gen_nomsbuild
set "GENERATOR=Ninja"
echo building with ninja instead, it goes to nvcc directly
goto :gen_pin

:gen_nomsbuild
echo.
echo msbuild cannot build cuda without those files, and ninja needs cl on PATH. pick one:
echo   1. copy them in, as administrator:
echo      copy "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions\*" "%VSPATH%\MSBuild\Microsoft\VC\v170\BuildCustomizations\"
echo   2. run the cuda installer again and select visual studio integration
echo   3. run this script from "x64 Native Tools Command Prompt" with ninja installed
exit /b 1

:gen_pin
if not defined GENERATOR goto :gen_done
if /i "%GENERATOR%"=="auto" goto :gen_done
set "GENARG=-G "%GENERATOR%""
:gen_done

rem the backend flags only turn a backend on, so a cache made for another one would keep
rem it, and its dlls would stay in bin. a cache also remembers its generator. start over
rem when either changed
if not exist "%BUILD%\CMakeCache.txt" goto :configure
set "HAVE=cpu"
findstr /b /i /c:"GGML_VULKAN:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=vulkan"
findstr /b /i /c:"GGML_CUDA:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=cuda"
if /i not "%HAVE%"=="%BACKEND%" goto :wipe_backend
if not defined GENARG goto :configure
findstr /b /i /c:"CMAKE_GENERATOR:INTERNAL=%GENERATOR%" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 goto :configure
echo %BUILD% was configured for another generator, wanted %GENERATOR%: removing it
rmdir /s /q "%BUILD%"
goto :configure

:wipe_backend
echo %BUILD% was configured for %HAVE%, wanted %BACKEND%: removing it
rmdir /s /q "%BUILD%"

:configure
echo building into %BUILD%, backend %BACKEND% %CMAKE_BACKEND% %GENARG%
if /i "%BACKEND%"=="cpu" echo   no cuda toolkit and no vulkan sdk found, this is a cpu only build
cmake -S "%~dp0..\.." -B "%BUILD%" %GENARG% %CMAKE_BACKEND% -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
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
