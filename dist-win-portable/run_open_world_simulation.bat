@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "VENV_PY=%SCRIPT_DIR%.venv\\Scripts\\python.exe"
set "PY_EMBED=%SCRIPT_DIR%python-3.10.11-embed-amd64\\python.exe"

rem Preferimos a .venv (ja vem com dependencias e SDK FBX)
if exist "%VENV_PY%" (
    set "PY_CMD=%VENV_PY%"
) else if exist "%PY_EMBED%" (
    set "PY_CMD=%PY_EMBED%"
    if not exist "%SCRIPT_DIR%.venv\\Lib\\site-packages" (
        echo [ALERTA] Ambiente .venv nao encontrado; veja README_portable.txt para recriar.
    )
) else (
    echo [ERRO] Nenhum Python portatil encontrado (.venv ou python-3.10.11-embed-amd64).
    echo Instale Python 3.10.11 64-bit e rode: python -m venv .venv ^& pip install -r requirements.txt
    popd
    endlocal
    exit /b 1
)

set "PYTHONUTF8=1"
"%PY_CMD%" "05 - open_world_simulation.py"
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% NEQ 0 (
    echo(
    echo [ERRO] A execucao falhou (codigo %EXIT_CODE%). Veja README_portable.txt para diagnostico/instalacao manual.
)

popd
endlocal & exit /b %EXIT_CODE%
