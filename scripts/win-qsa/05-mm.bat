@echo off
rem the dense q8_0 matmuls of a prompt on the device: the default path and the tile variants of mul_mm_f16reg.comp
rem (GGML_VK_MM_F16REG), each first through the MUL_MAT tests, then perf of the qwen4exp shapes (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" mm
exit /b %ERRORLEVEL%
