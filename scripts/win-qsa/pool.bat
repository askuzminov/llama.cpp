@echo off
rem the pool of servers (llama-server --pool): one address for the agents in front of the servers of all machines.
rem every machine runs serve.bat (router mode, the same model and context); this one also runs the pool.
rem the nodes and the options are in pool.ini next to this file (copy pool.example.ini and set the addresses).
rem the agents (qwen code) take http://<this machine>:12346/v1 as the OpenAI base url, the dashboard is
rem http://<this machine>:12346/pool. the pool loads no model, stopping it leaves the nodes running
setlocal

rem git bash / msys puts its own find, sort and friends ahead of the windows ones on PATH
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%PATH%"

rem ---- edit these, or set them in the environment before calling this file ----

if not defined LLAMA_SERVER  if exist "%~dp0..\..\build-win\bin\Release\llama-server.exe" set "LLAMA_SERVER=%~dp0..\..\build-win\bin\Release\llama-server.exe"
if not defined LLAMA_SERVER  set "LLAMA_SERVER=C:\Users\AI\Documents\github\llamacpp\llama.cpp\build-win\bin\Release\llama-server.exe"
if not defined POOL_INI      set "POOL_INI=%~dp0pool.ini"
if not defined POOLARGS      set "POOLARGS=--host 0.0.0.0 --port 12346"

rem ---------------------------------------------------------------------------

if not exist "%LLAMA_SERVER%" (
    echo ERROR: llama-server.exe not found
    echo %LLAMA_SERVER%
    pause
    exit /b 1
)

if not exist "%POOL_INI%" (
    echo ERROR: pool.ini not found, copy pool.example.ini next to this file and set the addresses
    echo %POOL_INI%
    pause
    exit /b 1
)

echo pool: %POOL_INI%
echo the agents: http://this-machine:12346/v1, the dashboard: http://this-machine:12346/pool
"%LLAMA_SERVER%" --pool "%POOL_INI%" %POOLARGS%
exit /b %ERRORLEVEL%
