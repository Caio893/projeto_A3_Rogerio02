Pacote portavel (Windows 64 bits)
---------------------------------
1) Extraia o ZIP em uma pasta sem espacos protegidos (ex.: C:\Games\A3World).
2) Execute um dos atalhos:
   - run_open_world_simulation.bat
   - run_advanced_world.bat
3) Os .bat usam .venv\Scripts\python.exe e, se faltar, o runtime embed python-3.10.11-embed-amd64\python.exe. As dependencias (incluindo SDK FBX) ja estao em .venv\Lib\site-packages.

Conteudo offline incluso (FBX SDK):
- Instalador: dist-win-portable/fbx202037_fbxpythonsdk_win.exe
- Arquivos extraidos: dist-win-portable/Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7/
  - Wheel: fbx-2020.3.7-cp310-none-win_amd64.whl

Se a .venv nao existir ou quebrar:
- Instale Python 3.10.11 64-bit com "Add python.exe to PATH".
- No terminal dentro de dist-win-portable:
  python portable_env_builder.py --offline --wheel-dir "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7"
  (ou manual)
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install --no-index --find-links "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7" -r requirements.txt
- Se a pasta de arquivos extraidos nao existir, rode o instalador fbx202037_fbxpythonsdk_win.exe nesta pasta e extraia novamente.

Rodar pelo VSCode (fallback):
- Selecione o interpretador .venv\Scripts\python.exe no VSCode.
- Terminal na raiz:
  python "05 - open_world_simulation.py"
  python "05 - advanced_world.py"
- Para depurar, crie um launch.json apontando para o script e o Python da .venv.

Notas:
- Requisitos: Windows 64 bits, driver OpenGL 3.3+, GPU razoavel.
- runtime_bootstrap.py tenta relancar com um Python que tenha o SDK; opcionalmente defina FBX_PYTHON_EXECUTABLES com outros pythons.
- Nao renomeie/mova as pastas FBX models, Textures, Heightmaps, python-3.10.11-embed-amd64 ou .venv.
