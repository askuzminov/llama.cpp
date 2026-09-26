@echo off
setlocal
call "%~dp0_config.bat"

rem msbuild does not build cuda on its own: it needs the integration files that the toolkit
rem ships in extras\visual_studio_integration. the cuda installer copies them into a visual
rem studio it finds at install time, so a studio newer than the toolkit never gets them and
rem cmake says "No CUDA toolset found", even with nvcc on PATH. -T cuda=<toolkit> sends
rem msbuild to the toolkit's own copy, and then it does not matter which studio cmake takes
set "GENARG="
set "TOOLSET="
set "CP=%CUDA_PATH%"
if defined CP if "%CP:~-1%"=="\" set "CP=%CP:~0,-1%"
if not defined GENERATOR set "GENERATOR=auto"
if /i not "%GENERATOR%"=="auto" set "GENARG=-G "%GENERATOR%""
if /i not "%BACKEND%"=="cuda" goto :gen_done
if /i "%GENERATOR%"=="Ninja" goto :gen_done
if not defined CP goto :gen_ninja
if not exist "%CP%\extras\visual_studio_integration\MSBuildExtensions" goto :gen_ninja
set "TOOLSET=-T "cuda=%CP%""
goto :gen_done

:gen_ninja
rem the toolkit itself has no integration files, so msbuild is out. ninja goes to nvcc directly
if defined CP echo %CP% has no extras\visual_studio_integration\MSBuildExtensions
if not defined CP echo CUDA_PATH is not set, so the integration files cannot be found
where ninja >nul 2>&1
if errorlevel 1 goto :gen_nonvcc
where cl >nul 2>&1
if not errorlevel 1 goto :gen_ninja_ok
rem ninja calls cl itself, so it has to be on PATH. take it from the studio, as the ci does.
rem vswhere sits under "program files (x86)", and for /f hands its command to cmd /c, which
rem strips the quotes when the path holds ( or ). so the list goes through a file
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :gen_nonvcc
set "VSTMP=%~dp0vswhere.tmp"
set "VSHOST="
"%VSWHERE%" -products * -sort -property installationPath > "%VSTMP%" 2>nul
set /p VSHOST=<"%VSTMP%"
del "%VSTMP%" >nul 2>&1
if not defined VSHOST goto :gen_nonvcc
if not exist "%VSHOST%\VC\Auxiliary\Build\vcvarsall.bat" goto :gen_nonvcc
echo taking the compiler from %VSHOST%
call "%VSHOST%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
where cl >nul 2>&1
if errorlevel 1 goto :gen_nonvcc

:gen_ninja_ok
set "GENERATOR=Ninja"
set "GENARG=-G Ninja"
echo building with ninja instead
goto :gen_done

:gen_nonvcc
echo.
echo without those files msbuild cannot build cuda, and ninja is not usable here. pick one:
echo   1. run the cuda installer again and select visual studio integration
echo   2. install ninja, then run this again
exit /b 1
:gen_done
rem for ninja cmake takes the first nvcc on PATH, and it can be from another toolkit. give it
rem the one in CUDA_PATH, as -T does for msbuild. when it changes, cmake makes a new cache
set "NVCCARG="
if /i "%BACKEND%"=="cuda" if not defined TOOLSET if defined CP if exist "%CP%\bin\nvcc.exe" set "NVCCARG="-DCMAKE_CUDA_COMPILER=%CP:\=/%/bin/nvcc.exe""

rem the backend flags only turn a backend on, so a cache made for another one would keep
rem it, and its dlls would stay in bin. a cache also remembers its generator and its
rem toolset. start over when any of them changed
if not exist "%BUILD%\CMakeCache.txt" goto :configure
set "HAVE=cpu"
findstr /b /i /c:"GGML_VULKAN:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=vulkan"
findstr /b /i /c:"GGML_CUDA:BOOL=ON" "%BUILD%\CMakeCache.txt" >nul
if not errorlevel 1 set "HAVE=cuda"
if /i not "%HAVE%"=="%BACKEND%" goto :wipe_backend
rem msbuild gets nvcc from the -T toolset. cmake refuses to change the toolset of a cache it
rem made, and compares the text as it is: match case too. /l keeps the \ in the path literal
if not defined TOOLSET goto :check_gen
findstr /b /l /c:"CMAKE_GENERATOR_TOOLSET:INTERNAL=cuda=%CP%" "%BUILD%\CMakeCache.txt" >nul
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
echo building into %BUILD%, backend %BACKEND% %CMAKE_BACKEND% %GENARG% %TOOLSET% %NVCCARG%
if /i "%BACKEND%"=="cpu" echo   no cuda toolkit and no vulkan sdk found, this is a cpu only build
cmake -S "%~dp0..\.." -B "%BUILD%" %GENARG% %TOOLSET% %NVCCARG% %CMAKE_BACKEND% -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
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
