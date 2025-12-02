# projeto_A3_Rogerio02
Trabalho de A3

Pacote portavel (Windows 64 bits)
---------------------------------
1) Extraia o ZIP em uma pasta sem espacos protegidos (ex.: C:\Games\A3World).
2) Execute um dos atalhos:
   - `run_open_world_simulation.bat`
   - `run_advanced_world.bat`
3) Os .bat usam `.venv\Scripts\python.exe` e, se faltar, o runtime embed `python-3.10.11-embed-amd64\python.exe`. As dependencias (incluindo SDK FBX) ja estao em `.venv\Lib\site-packages`.

Conteudo offline incluso (FBX SDK):
- Instalador: `dist-win-portable/fbx202037_fbxpythonsdk_win.exe`
- Arquivos extraidos: `dist-win-portable/Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7/`
  - Wheel: `fbx-2020.3.7-cp310-none-win_amd64.whl`

Se a .venv nao existir ou quebrar:
- Instale Python 3.10.11 64-bit com "Add python.exe to PATH".
- No terminal dentro de `dist-win-portable`:
  ```bash
  python portable_env_builder.py --offline --wheel-dir "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7"
  ```
  (ou manual)
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install --no-index --find-links "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7" -r requirements.txt
  ```
- Se a pasta de arquivos extraidos nao existir, rode o instalador `fbx202037_fbxpythonsdk_win.exe` nesta pasta e extraia novamente.

Rodar pelo VSCode (fallback)
- Selecione o interpretador `.venv\Scripts\python.exe` no VSCode.
- Terminal na raiz:
  ```bash
  python "05 - open_world_simulation.py"
  python "05 - advanced_world.py"
  ```
- Para depurar, crie um `launch.json` apontando para o script e o Python da .venv.

Notas
- Requisitos: Windows 64 bits, driver OpenGL 3.3+, GPU razoavel.
- `runtime_bootstrap.py` tenta relancar com um Python que tenha o SDK; opcionalmente defina `FBX_PYTHON_EXECUTABLES` com outros pythons.
- Nao renomeie/mova as pastas FBX models, Textures, Heightmaps, python-3.10.11-embed-amd64 ou .venv.



Caso os .bat falhem siga esse guia:

    Nao mova/renomeie as pastas internas (FBX models, Textures, Heightmaps, python-3.10.11-embed-amd64, .venv).
    O runtime portatil adiciona ~8-9 MB ao zip distribuido (e ~16 MB apos extraido).
    Se o arquivo .bat nao iniciar, clique com botao direito e escolha "Executar como administrador" (em alguns ambientes bloqueados).


    Criar um ambiente do zero (se a .venv quebrar ou em outra maquina)

    Instale o Python 3.10.11 64-bit com a opcao "Add python.exe to PATH".
    No terminal na pasta do projeto: python -m venv .venv ..venv\Scripts\activate python -m pip install --upgrade pip python -m pip install -r requirements.txt
    Sobre o FBX: o wheel esperado e já esta na pasta o .exe onde extrai o fbx-2020.3.7-cp310-none-win_amd64.whl. python -m pip install "fbx_sdk/FBX Python SDK/2020.3.7/fbx-2020.3.7-cp310-none-win_amd64.whl" // Local onde o .exe estrair o .whl
        Caso queira baixar diretamente do site Link: https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxpythonsdk_win.exe e instale apontando para o .whl correspondente.
    No VSCode, selecione o interpretador .venv\Scripts\python.exe depois da instalacao. //python310

    Como rodar pelo VSCode

    Mantenha o terminal aberto na raiz do projeto (onde estao as pastas Textures, Heightmaps, FBX models).
    Com o interpretador correto selecionado, execute: python "05 - open_world_simulation.py" python "05 - advanced_world.py"

