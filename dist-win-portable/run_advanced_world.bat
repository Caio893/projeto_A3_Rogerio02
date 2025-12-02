@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "VENV_PY=%SCRIPT_DIR%.venv\\Scripts\\python.exe"
set "PY_EMBED=%SCRIPT_DIR%python-3.10.11-embed-amd64\\python.exe"
set "SDK_WHEEL_DIR=%SCRIPT_DIR%Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe\\2020.3.7"

rem Preferimos a .venv (ja vem com dependencias e SDK FBX)
if exist "%VENV_PY%" (
    set "PY_CMD=%VENV_PY%"
) else if exist "%PY_EMBED%" (
    set "PY_CMD=%PY_EMBED%"
    if not exist "%SCRIPT_DIR%.venv\\Lib\\site-packages" (
        echo [ALERTA] Ambiente .venv nao encontrado; veja README_portable.txt para recriar usando o wheel em:
        echo   "%SDK_WHEEL_DIR%"
    )
) else (
    echo [ERRO] Nenhum Python portatil encontrado (.venv ou python-3.10.11-embed-amd64).
    echo Instale Python 3.10.11 64-bit e rode:
    echo   python portable_env_builder.py --offline --wheel-dir "%SDK_WHEEL_DIR%"
    echo (ou) python -m venv .venv ^& pip install --no-index --find-links "%SDK_WHEEL_DIR%" -r requirements.txt
    popd
    endlocal
    exit /b 1
)

set "PYTHONUTF8=1"
"%PY_CMD%" "05 - advanced_world.py"
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% NEQ 0 (
    echo(
    echo [ERRO] A execucao falhou (codigo %EXIT_CODE%). Veja README_portable.txt para diagnostico/instalacao manual.
)

popd
endlocal & exit /b %EXIT_CODE%
