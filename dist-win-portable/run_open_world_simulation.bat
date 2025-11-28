@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "PY_EMBED=%SCRIPT_DIR%python-3.10.11-embed-amd64\\python.exe"
if not exist "%PY_EMBED%" (
    echo [ERRO] Runtime Python portatil nao encontrado em "python-3.10.11-embed-amd64". Extraia o pacote completo.
    popd
    endlocal
    exit /b 1
)

set "PYTHONUTF8=1"
"%PY_EMBED%" "05 - open_world_simulation.py"

popd
endlocal
