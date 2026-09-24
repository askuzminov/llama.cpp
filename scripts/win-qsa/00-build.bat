@echo off
setlocal
call "%~dp0_config.bat"

rem msbuild builds cuda through the toolkit's own integration files, and the cuda installer
rem copies them only into a visual studio it finds at install time, so a studio newer than the
rem toolkit never has them. cmake takes the newest studio, then says "No CUDA toolset found"
rem even with nvcc on PATH. take the studio that does have them, or ninja, which needs none
set "GENARG="
if /i not "%GENERATOR%"=="auto" goto :gen_pin
if /i not "%BACKEND%"=="cuda" goto :gen_done
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :gen_done
set "VSNEW="
set "VSCUDA="
set "VSHOST="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -products * -sort -property installationPath`) do call :gen_scan "%%i"
if not defined VSCUDA goto :gen_ninja
rem cmake already takes the newest studio, name one only when the files are in another
if /i "%VSCUDA%"=="%VSNEW%" goto :gen_done
set "VSMAJ="
set "VSYEAR="
for /f "usebackq tokens=1 delims=." %%v in (`"%VSWHERE%" -path "%VSCUDA%" -property installationVersion`) do set "VSMAJ=%%v"
for /f "usebackq delims=" %%v in (`"%VSWHERE%" -path "%VSCUDA%" -property catalog_productLineVersion`) do set "VSYEAR=%%v"
if not defined VSYEAR if "%VSMAJ%"=="17" set "VSYEAR=2022"
if not defined VSYEAR if "%VSMAJ%"=="16" set "VSYEAR=2019"
if not defined VSMAJ goto :gen_done
if not defined VSYEAR goto :gen_done
echo cuda integration for msbuild is only in %VSCUDA%
set "GENERATOR=Visual Studio %VSMAJ% %VSYEAR%"
goto :gen_pin

:gen_ninja
echo no visual studio here has the cuda integration for msbuild
where ninja >nul 2>&1
if errorlevel 1 goto :gen_nomsbuild
where cl >nul 2>&1
if not errorlevel 1 goto :gen_ninja_ok
rem ninja calls cl itself, so it has to be on PATH. take it from the studio, as the ci does
if not defined VSHOST set "VSHOST=%VSNEW%"
if not defined VSHOST goto :gen_nomsbuild
if not exist "%VSHOST%\VC\Auxiliary\Build\vcvarsall.bat" goto :gen_nomsbuild
echo taking the compiler from %VSHOST%
call "%VSHOST%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
where cl >nul 2>&1
if errorlevel 1 goto :gen_nomsbuild
:gen_ninja_ok
set "GENERATOR=Ninja"
echo building with ninja instead, it goes to nvcc directly
goto :gen_pin

:gen_nomsbuild
echo.
echo msbuild cannot build cuda without those files, and ninja needs cl. pick one:
echo   1. install cuda 13.4, it knows visual studio 2026 and is what upstream ci builds
echo   2. copy the files in, as administrator:
echo      copy "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions\*" "%VSNEW%\MSBuild\Microsoft\VC\v170\BuildCustomizations\"
echo   3. run the cuda installer again and select visual studio integration
exit /b 1

:gen_pin
if not defined GENERATOR goto :gen_done
if /i "%GENERATOR%"=="auto" goto :gen_done
set "GENARG=-G "%GENERATOR%""
:gen_done

rem the backend flags only turn a backend on, so a cache made for another one would keep
rem it, and its dlls would stay in bin. a cache also remembers its generator and its nvcc.
rem start over when any of them changed
if not exist "%BUILD%\CMakeCache.txt" goto :configure
set "HAVE=cpu"
findstr /b /i /c:"GGML_VULKAN:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=vulkan"
findstr /b /i /c:"GGML_CUDA:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=cuda"
if /i not "%HAVE%"=="%BACKEND%" goto :wipe_backend
rem a new toolkit does not replace the nvcc in the cache, and a configure that died on a
rem missing toolset leaves a cache with no nvcc at all. the cache stores the path with /
if /i not "%BACKEND%"=="cuda" goto :check_gen
if not defined CUDA_PATH goto :check_gen
set "CP=%CUDA_PATH%"
if "%CP:~-1%"=="\" set "CP=%CP:~0,-1%"
findstr /b /i /c:"CMAKE_CUDA_COMPILER:FILEPATH=%CP:\=/%/bin/nvcc.exe" "%BUILD%\CMakeCache.txt" >nul
if errorlevel 1 goto :wipe_toolkit

:check_gen
if not defined GENARG goto :configure
findstr /b /i /c:"CMAKE_GENERATOR:INTERNAL=%GENERATOR%" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 goto :configure
echo %BUILD% was configured for another generator, wanted %GENERATOR%: removing it
rmdir /s /q "%BUILD%"
goto :configure

:wipe_toolkit
echo %BUILD% was configured for another cuda toolkit, wanted %CP%: removing it
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

exit /b 0

rem %1 = one visual studio install, newest first
:gen_scan
if not defined VSNEW set "VSNEW=%~1"
if not defined VSHOST call :gen_host "%~1"
if defined VSCUDA goto :eof
dir /b /s "%~1\MSBuild\Microsoft\VC\CUDA *.props" >nul 2>&1
if errorlevel 1 goto :eof
set "VSCUDA=%~1"
goto :eof

rem ninja needs a compiler the toolkit accepts. a toolkit with no integration for the newest
rem studio usually refuses its cl too, so prefer 2022
:gen_host
for /f "usebackq tokens=1 delims=." %%v in (`"%VSWHERE%" -path "%~1" -property installationVersion`) do set "VSMAJ=%%v"
if "%VSMAJ%"=="17" set "VSHOST=%~1"
goto :eof
