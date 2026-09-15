@echo off
rem how far is the row cache read path from what the drive can do? our own path is swept
rem first, then diskspd gives the same random pattern straight to the drive, then the same
rem diskspd run repeats with a buffered reader on the file and once more after that reader has
rem ended, and our sweep runs a second time on the now warm file.
rem diskspd reads the model file itself, random (-r) at several queue depths. it reads
rem unbuffered (-Suw), so its numbers are the drive alone with no page cache in the way, which
rem is the ceiling our buffered path is measured against. the row cache issues one blocking
rem read per reader thread, so "-o1 -tN" is its pattern and "-oN -t4" is the same depth reached
rem asynchronously: if only the second one scales, the readers have to keep several in flight.
rem read-only: -w0 and no -c, so the file is never written, created or extended.
rem if diskspd shows the same numbers, the wall is the disk and there is nothing to fix.
rem in the output look at the "total:" line of the Read IO block: I/O per s and MiB/s.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem variants are "blockKiB queueDepth threads", one diskspd run each
if not defined DISKVARIANTS set "DISKVARIANTS="4 1 1" "4 1 8" "4 1 32" "4 8 4" "4 32 4" "256 32 4""
if not defined DISKSECS    set "DISKSECS=15"

rem diskspd has to read the same file the row cache reads, or the two are not comparable. a
rem split model keeps the table in one part, and the row cache now prints which one. set
rem DISKFILE in _local.bat if that is not the first part
if not defined DISKFILE set "DISKFILE=%MODEL%"

rem how many times the read path sweep runs back to back at the start. a file nothing has read
rem for a while is slower than one a few GB have just gone through, so a second load with nothing
rem in between tells apart "opening the model does it" from "diskspd does it"
if not defined READPATHRUNS set "READPATHRUNS=2"

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

rem DISKSPD in _local.bat wins, then a copy next to these scripts, then a download
if not defined DISKSPD set "DISKSPD=%~dp0diskspd\amd64\diskspd.exe"
if not exist "%DISKSPD%" call :fetch
if not exist "%DISKSPD%" exit /b 1

set "LOG=%LOGS%\13-disk-iops-%TS%.log"
echo writing %LOG%

echo ### FILE=%MODEL% > "%LOG%"
echo ### DISKSPD=%DISKSPD% >> "%LOG%"
echo ### DISKVARIANTS=%DISKVARIANTS% DISKSECS=%DISKSECS% READPATHRUNS=%READPATHRUNS% >> "%LOG%"

rem a real time scanner reads a file it is asked about before the caller gets it, which is the
rem best explanation left for a cold file being slow, so record whether one is on
powershell -NoProfile -Command "try { $s = Get-MpComputerStatus; 'realtime=' + $s.RealTimeProtectionEnabled } catch { 'realtime=unknown' }" >> "%LOG%" 2>&1

set "RC=0"

rem our own read path first: the same handles, readers and pool slots a gather uses, over
rem the same request sizes and queue depths diskspd is asked for below
rem only the first pass is cold: it leaves the pages it read in the cache for the next one
call :readpath cold
for /l %%r in (2,1,%READPATHRUNS%) do call :readpath "repeat %%r"

for %%v in (%DISKVARIANTS%) do for /f "tokens=1-3" %%a in (%%v) do call :run %%a %%b %%c

rem the same unbuffered run, but with a buffered reader on the file at the same time. our
rem readers are unbuffered while the loader reads the header of the same file through the
rem cache, and windows keeps a non cached read coherent with the cache. if this run also
rem collapses to the single thread number, that is where our missing queue depth goes
call :mixed

rem the same unbuffered run once the buffered reader has ended. the pages it left behind stay
rem in the cache, so this tells apart "a reader running now" from "pages cached earlier": only
rem the second one would explain a row cache that is slow with nothing else on the file
call :after

rem and the row cache read path again, on a file that is now warm in the page cache
call :readpath warm

echo.
echo === summary
echo. >> "%LOG%"
echo ### summary >> "%LOG%"
set "PICK=/C:"### -b" /C:"### row cache" /C:"read path:" /C:"stays on disk" /C:"total:" /C:"realtime=" /C:"### exit=""
findstr %PICK% "%LOG%"
findstr %PICK% "%LOG%" >> "%LOG%"

echo.
echo done, %LOG%
exit /b %RC%

rem the row cache read path sweep runs inside the loader, so any run that opens the model
rem with -lzm dio prints it. one model load, a few seconds of reads
:readpath
if not exist "%BIN%\llama-bench.exe" (
    echo skipping the read path sweep: %BIN%\llama-bench.exe is not built
    goto :eof
)
if not defined BENCHREQS set "BENCHREQS=2048"
echo === row cache read path (%~1), %BENCHREQS% blocks per point
echo. >> "%LOG%"
echo ### row cache read path (%~1), BENCHREQS=%BENCHREQS% blocks per point >> "%LOG%"
set "LLAMA_ROW_CACHE_BENCH=%BENCHREQS%"
"%BIN%\llama-bench.exe" -m "%MODEL%" -v -fa on -p 1 -n 0 -r 1 -lm dio -lzm dio %EXTRA% -o md >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
set "LLAMA_ROW_CACHE_BENCH="
goto :eof

:after
echo === -b4K -o1 -t32 after the buffered reader has ended
echo. >> "%LOG%"
echo ### -b4K -o1 -t32 -r -w0 -Suw -d%DISKSECS%, after the buffered reader has ended >> "%LOG%"
"%DISKSPD%" -b4K -o1 -t32 -r -w0 -Suw -L -d%DISKSECS% "%DISKFILE%" >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
goto :eof

:mixed
echo === -b4K -o1 -t32 with a buffered reader on the same file
echo. >> "%LOG%"
echo ### -b4K -o1 -t32 -r -w0 -Suw -d%DISKSECS%, buffered -Sb -o1 -t4 running on the same file >> "%LOG%"
set "NOISELOG=%LOGS%\13-buffered-noise-%TS%.log"
set /a "NOISESECS=%DISKSECS%+15" >nul
start "" /b "%DISKSPD%" -b4K -o1 -t4 -r -w0 -Sb -d%NOISESECS% "%DISKFILE%" > "%NOISELOG%" 2>&1
powershell -NoProfile -Command "Start-Sleep -Seconds 5"
"%DISKSPD%" -b4K -o1 -t32 -r -w0 -Suw -L -d%DISKSECS% "%DISKFILE%" >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
powershell -NoProfile -Command "Start-Sleep -Seconds 15"
goto :eof

rem %1 = block KiB, %2 = outstanding requests per thread, %3 = threads
:run
echo === -b%~1K -o%~2 -t%~3
echo. >> "%LOG%"
echo ### -b%~1K -o%~2 -t%~3 -r -w0 -Suw -d%DISKSECS% >> "%LOG%"
"%DISKSPD%" -b%~1K -o%~2 -t%~3 -r -w0 -Suw -L -d%DISKSECS% "%DISKFILE%" >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
goto :eof

:fetch
if not defined DISKSPDURL set "DISKSPDURL=https://github.com/microsoft/diskspd/releases/download/v2.2/DiskSpd.ZIP"
set "ZIP=%~dp0DiskSpd.ZIP"
if exist "%ZIP%" goto :unpack

echo downloading %DISKSPDURL%
where curl.exe >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%DISKSPDURL%' -OutFile '%ZIP%'"
) else (
    curl.exe -L --fail --retry 2 -o "%ZIP%" "%DISKSPDURL%"
)
set "EC=!ERRORLEVEL!"
if not "!EC!"=="0" if exist "%ZIP%" del "%ZIP%"

if not exist "%ZIP%" (
    echo.
    echo download failed, exit=!EC!
    echo get %DISKSPDURL% by hand, put DiskSpd.ZIP next to these scripts and run this again,
    echo or unpack it anywhere and set DISKSPD to the full path of amd64\diskspd.exe in _local.bat
    goto :eof
)

:unpack
powershell -NoProfile -Command "Expand-Archive -Force -Path '%ZIP%' -DestinationPath '%~dp0diskspd'"
if not "%ERRORLEVEL%"=="0" goto :eof
del "%ZIP%"

if not exist "%DISKSPD%" echo unpacked, but %DISKSPD% is still missing
goto :eof
