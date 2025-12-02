Pacote portavel (Windows 64 bits)
---------------------------------
1) Extraia todo o ZIP para uma pasta sem espacos protegidos (ex.: C:\Games\A3World).
2) Execute um dos atalhos na raiz:
   - run_open_world_simulation.bat
   - run_advanced_world.bat
3) Os .bat procuram primeiro `.venv\Scripts\python.exe` e usam o runtime embed `python-3.10.11-embed-amd64\python.exe` como reserva. As dependencias (inclusive FBX) ja estao em `.venv\Lib\site-packages`. Nao precisa instalar Python no Windows.

Requisitos rapidos:
- Windows 64 bits com driver OpenGL 3.3+ ativo.
- GPU dedicada ou integrada recente.
- Permissao para capturar o mouse enquanto a janela roda.

Se der erro ao abrir:
- Confirme que a pasta `.venv` veio no ZIP; se nao veio, siga a secao "Recriar o ambiente".
- Rode manualmente pelo terminal (na raiz):
  .\.venv\Scripts\python.exe "05 - open_world_simulation.py"
  .\.venv\Scripts\python.exe "05 - advanced_world.py"
- Erro de `fbx` ausente: instale o wheel `fbx-2020.3.7-cp310-none-win_amd64.whl` (veja abaixo). O script tambem tenta relancar usando `FBX_PYTHON_EXECUTABLES`.

Recriar o ambiente do zero (caso a .venv falhe ou nao esteja no ZIP)
- Instale Python 3.10.11 64-bit com "Add python.exe to PATH".
- No terminal na pasta do projeto:
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
- Instale o SDK FBX (wheel `fbx-2020.3.7-cp310-none-win_amd64.whl`):
  * Se tiver a pasta `fbx_sdk/FBX Python SDK/2020.3.7/`, rode:
    python -m pip install "fbx_sdk/FBX Python SDK/2020.3.7/fbx-2020.3.7-cp310-none-win_amd64.whl"
  * Se nao tiver a pasta, baixe o "FBX Python SDK 2020.3.7" para CPython 3.10 (Windows) no site da Autodesk e instale apontando para o wheel.
- No VSCode, selecione o interpretador `.venv\Scripts\python.exe`.

Rodar pelo VSCode (opcao de depuracao)
- Terminal na raiz (onde estao Textures/Heightmaps/FBX models).
- Com a .venv selecionada: `python "05 - open_world_simulation.py"` ou `python "05 - advanced_world.py"`.
- Para depurar, crie um `launch.json` apontando para o script escolhido e o Python da .venv.

Observacoes:
- Nao mova/renomeie as pastas FBX models, Textures, Heightmaps, python-3.10.11-embed-amd64 ou .venv.
- O runtime embed adiciona ~8-9 MB ao ZIP (e ~16 MB apos extraido).
- Se o .bat nao abrir, tente "Executar como administrador" em ambientes bloqueados.
