# projeto_A3_Rogerio02
Trablho de A3

Pacote portavel (Windows 64 bits)
---------------------------------
1) Extraia o zip inteiro para uma pasta sem espacos protegidos (ex.: C:\Games\A3World).
2) Abra a pasta extraida e execute um dos atalhos:
   - run_open_world_simulation.bat
   - run_advanced_world.bat
3) Os atalhos usam o Python 3.10 portatil incluso (pasta python-3.10.11-embed-amd64) e o conjunto de dependencias pre-instaladas em .venv\Lib\site-packages (incluindo FBX SDK). Nao precisa de Python instalado no Windows.

Requisitos:
- Windows 64 bits com driver OpenGL 3.3+ funcional.
- GPU dedicada ou integrada recente (melhor desempenho).
- Permissao para usar o mouse em modo "capturado" (a janela trava o cursor enquanto roda).

Observacoes:
- Nao mova/renomeie as pastas internas (FBX models, Textures, Heightmaps, python-3.10.11-embed-amd64, .venv).
- O runtime portatil adiciona ~8-9 MB ao zip distribuido (e ~16 MB apos extraido).
- Se o arquivo .bat nao iniciar, clique com botao direito e escolha "Executar como administrador" (em alguns ambientes bloqueados).

Guia para rodar pelo VSCode (fallback aos .bat)
-----------------------------------------------
Objetivo: iniciar os scripts caso os .bat falhem, usando o VSCode ou um terminal comum.

1) Requisitos de sistema
- Windows 64 bits com driver OpenGL 3.3+ funcionando e GPU razoavel.
- VSCode com extensao "Python".
- Python 3.10.x 64 bits (CPython). O wheel do FBX so existe para cp310/win_amd64.

2) Usar o ambiente portatil que ja veio no zip (mais rapido)
- A pasta traz `.venv` com dependencias e `python-3.10.11-embed-amd64` com runtime configurado.
- No VSCode: Ctrl+Shift+P -> "Python: Select Interpreter" -> escolha `.venv\Scripts\python.exe`.
- Rode no terminal integrado, estando na pasta raiz:
  .\.venv\Scripts\python.exe "05 - open_world_simulation.py"
  .\.venv\Scripts\python.exe "05 - advanced_world.py"
- Se preferir acionar o runtime embed diretamente (sem .venv):
  set PYTHONUTF8=1 & .\python-3.10.11-embed-amd64\python.exe "05 - open_world_simulation.py"
  set PYTHONUTF8=1 & .\python-3.10.11-embed-amd64\python.exe "05 - advanced_world.py"
- Opcional: defina `FBX_PYTHON_EXECUTABLES` com o caminho do Python que possui o SDK FBX para forcar o uso dele.

3) Criar um ambiente do zero (se a .venv quebrar ou em outra maquina)
- Instale o Python 3.10.11 64-bit com a opcao "Add python.exe to PATH".
- No terminal na pasta do projeto:
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
- Sobre o FBX: o wheel esperado e `fbx-2020.3.7-cp310-none-win_amd64.whl`.
  * Se o zip tiver a pasta `fbx_sdk/FBX Python SDK/2020.3.7/`, instale direto com:
    python -m pip install "fbx_sdk/FBX Python SDK/2020.3.7/fbx-2020.3.7-cp310-none-win_amd64.whl"
  * Caso a pasta nao exista, baixe o "FBX Python SDK 2020.3.7" para CPython 3.10 (Windows) Link: https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxpythonsdk_win.exe e instale apontando para o .whl correspondente.
- No VSCode, selecione o interpretador `.venv\Scripts\python.exe` depois da instalacao.

4) Como rodar pelo VSCode
- Mantenha o terminal aberto na raiz do projeto (onde estao as pastas Textures, Heightmaps, FBX models).
- Com o interpretador correto selecionado, execute:
  python "05 - open_world_simulation.py"
  python "05 - advanced_world.py"
