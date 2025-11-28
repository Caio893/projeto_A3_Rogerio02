"""
Guarantee that the project runs with an interpreter that already has the FBX SDK.

If the current Python cannot import ``fbx``, we simply re-launch the script using
one of the interpreters that are known to have the SDK preinstalled.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def _has_fbx() -> bool:
    try:
        import fbx  # type: ignore # pylint: disable=unused-import
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _same_executable(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return a.resolve() == b.resolve()


def _iter_env_candidates() -> Iterable[Path]:
    env_value = os.environ.get("FBX_PYTHON_EXECUTABLES", "")
    for raw in env_value.split(os.pathsep):
        raw = raw.strip()
        if not raw:
            continue
        yield Path(raw)


def _iter_default_candidates(current: Path) -> Iterable[Path]:
    # Prefer an explicit 3.10 install because the bundled wheel targets cp310/win_amd64.
    win_default_roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python",
        Path("C:/Python310"),
        Path(os.environ.get("ProgramFiles", "")) / "Python310",
    ]
    if sys.platform == "win32":
        for root in win_default_roots:
            candidate = root / "python.exe"
            yield candidate
    # Generic fallbacks resolved via PATH.
    for name in ("python3.10", "python3"):
        located = shutil.which(name)
        if located:
            yield Path(located)
    # Finally, try a sibling python in the same folder (portable installs).
    yield current.parent / "python.exe"
    yield current.parent / "python3.10"


def _resolve_existing(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        try:
            candidate = path.resolve()
        except OSError:
            continue
        if candidate.exists() and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def ensure_supported_runtime() -> None:
    if _has_fbx():
        return

    current = Path(sys.executable).resolve()
    candidates = _resolve_existing(
        list(_iter_env_candidates()) + list(_iter_default_candidates(current))
    )

    for candidate in candidates:
        if _same_executable(candidate, current):
            continue

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        cmd = [str(candidate), *sys.argv]
        try:
            exit_code = subprocess.call(cmd, env=env)  # noqa: S603
        except OSError:
            # Candidate resolved/existed but falhou ao criar processo; tenta o proximo.
            continue
        os._exit(exit_code)

    raise RuntimeError(
        "FBX SDK nao esta disponivel no interpretador atual e nenhum Python configurado "
        "foi encontrado. Instale o wheel (fbx-2020.3.7-cp310-none-win_amd64.whl) com "
        "`pip install \"fbx_sdk/FBX Python SDK/2020.3.7/fbx-2020.3.7-cp310-none-win_amd64.whl\"` "
        "ou defina FBX_PYTHON_EXECUTABLES com o caminho do Python 3.10 que possui o SDK."
    )


__all__ = ["ensure_supported_runtime"]
