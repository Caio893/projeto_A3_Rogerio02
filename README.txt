Pacote portavel (Windows 64 bits)
---------------------------------
1) Extraia o ZIP em uma pasta sem espacos protegidos (ex.: C:\Games\A3World).
2) Execute um dos atalhos:
   - run_open_world_simulation.bat
   - run_advanced_world.bat
3) Os .bat usam .venv\Scripts\python.exe e, se preciso, o runtime embed python-3.10.11-embed-amd64\python.exe. As dependencias ja estao em .venv\Lib\site-packages (inclui SDK FBX); nao precisa instalar Python.

Se der erro:
- Verifique se a pasta .venv esta presente; se nao estiver, recrie o ambiente (abaixo).
- Rode manualmente:
  .\.venv\Scripts\python.exe "05 - open_world_simulation.py"
  .\.venv\Scripts\python.exe "05 - advanced_world.py"
- Se aparecer "No module named 'fbx'", instale o wheel fbx-2020.3.7-cp310-none-win_amd64.whl (veja "Recriar o ambiente").

Recriar o ambiente do zero (caso a .venv falhe ou falte no ZIP)
- Instale Python 3.10.11 64-bit com "Add python.exe to PATH".
- No terminal na raiz do projeto:
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
- SDK FBX: instale o wheel fbx-2020.3.7-cp310-none-win_amd64.whl.
  * Se tiver a pasta fbx_sdk/FBX Python SDK/2020.3.7/, use:
    python -m pip install "fbx_sdk/FBX Python SDK/2020.3.7/fbx-2020.3.7-cp310-none-win_amd64.whl"
  * Se nao tiver a pasta, baixe o "FBX Python SDK 2020.3.7" para CPython 3.10 (Windows) no site da Autodesk e instale apontando para o wheel.
- No VSCode, selecione o interpretador .venv\Scripts\python.exe.

Rodar pelo VSCode (opcao de depuracao)
- Terminal na raiz (onde estao Textures/Heightmaps/FBX models).
- Com a .venv selecionada:
  python "05 - open_world_simulation.py"
  python "05 - advanced_world.py"
- Para depurar, crie um launch.json apontando para o script escolhido e o Python da .venv.

Porque o .bat falhava em outra maquina?
- O repositorio publico nao inclui a pasta .venv nem o wheel do SDK FBX (fbx_sdk/.../fbx-2020.3.7-cp310-none-win_amd64.whl). Sem eles, o Python embed nao encontra o pacote fbx.
- Os .bat agora priorizam a .venv e avisam quando ela estiver ausente, facilitando o diagnostico.
