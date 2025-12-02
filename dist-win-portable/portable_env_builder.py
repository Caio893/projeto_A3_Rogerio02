"""
Ferramenta para montar a ".venv" usada pelos .BAT portáteis.

O script cria/atualiza um ambiente virtual localizado em
``dist-win-portable/.venv`` e instala todos os pacotes listados em
``dist-win-portable/requirements.txt``. Ele foi pensado para ser executado
em uma máquina Windows com Python 3.10 e acesso à Internet ou a um diretório
local contendo os wheels necessários (especialmente o SDK do FBX).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
DEFAULT_WHEEL_DIR = PROJECT_ROOT / "fbx_sdk" / "FBX Python SDK" / "2020.3.7"
VENV_DIR = PROJECT_ROOT / ".venv"


class CommandError(RuntimeError):
    """Erro de execução ao chamar subprocessos."""


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, check=False)  # noqa: S603,S607
    if result.returncode != 0:
        raise CommandError(f"Comando falhou (código {result.returncode}): {' '.join(cmd)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monta a .venv portátil usada pelos .BAT.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Caminho do python 3.10 a ser usado para criar a venv (padrao: interpretador atual).",
    )
    parser.add_argument(
        "--wheel-dir",
        default=str(DEFAULT_WHEEL_DIR),
        help="Pasta com wheels locais (incluindo o fbx-2020.3.7-cp310-none-win_amd64.whl).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Instala apenas a partir de wheels locais (usa --no-index e --find-links).",
    )
    return parser


def _validate_python(python_cmd: str) -> None:
    version_cmd = [python_cmd, "-c", "import sys; print(sys.platform); print(sys.version)"]
    completed = subprocess.run(version_cmd, capture_output=True, text=True, check=False)  # noqa: S603,S607
    if completed.returncode != 0:
        raise CommandError("Nao foi possivel executar o interpretador informado.")

    platform, version = completed.stdout.strip().splitlines()[:2]
    if platform != "win32":
        print("[aviso] Recomenda-se executar em Windows 64 bits para gerar binaries corretos.")
    if not version.startswith("3.10"):
        print("[aviso] Use Python 3.10 para compatibilidade com o SDK do FBX.")


def _ensure_venv(python_cmd: str) -> Path:
    VENV_DIR.mkdir(exist_ok=True)
    if os.name == "nt":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "Scripts" / "python"
    if not venv_python.exists():
        print(f"[info] Criando venv em {VENV_DIR}...")
        _run([python_cmd, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"[info] Reutilizando venv existente em {VENV_DIR}...")
    return venv_python


def _install_requirements(venv_python: Path, wheel_dir: Path, offline: bool) -> None:
    pip_cmd = [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]
    _run(pip_cmd)

    install_cmd = [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
    if offline:
        install_cmd.extend(["--no-index", "--find-links", str(wheel_dir)])
    elif wheel_dir.exists():
        install_cmd.extend(["--find-links", str(wheel_dir), "--prefer-binary"])

    print(f"[info] Instalando dependencias ({'offline' if offline else 'online/offline'})...")
    _run(install_cmd)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    python_cmd = str(Path(args.python).expanduser())
    wheel_dir = Path(args.wheel_dir).expanduser()

    try:
        _validate_python(python_cmd)
        venv_python = _ensure_venv(python_cmd)
        _install_requirements(venv_python, wheel_dir, args.offline)
    except CommandError as exc:
        print(f"[erro] {exc}")
        return 1

    print("[ok] Ambiente portatil montado. Os .BAT devem encontrar as dependencias em .venv/Lib/site-packages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
