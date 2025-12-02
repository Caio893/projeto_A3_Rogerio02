Guia de instalação de ambiente
---------------------------------

Criar um ambiente usando o .venv dentro do projeto
- Com vsCode aberto no terminal na pasta do projeto:
  `python -m venv .venv`
  `.\.venv\Scripts\activate` ou `\.venv\Scripts\activate` Essa parte eu me confundo professor.
  `python -m pip install --upgrade pip`
  `python -m pip install -r requirements.txt`

Adicione o python.exe dentro de python-3.10.11-embed-amd64 como seu interpretador.
- Dentro do vsode aperte Control + Shift + P
- Vá em ou digite "Escolher Interpretador"
- Clique em Insira o caminho do interpretador
- E escolha o python.exe pasta python-3.10.11-embed-amd64:

Tentei fazer de forma simples apernas clicando no .bat mas aparentemente não funciona direto em outros pcs, então essa parte aqui para baixo pode ser desconsiderada, apenas o guia acima funcionou em outra maquina além da minha.


Guia data e mais técnico para instalar o ambiente sem o .venv

1) Extraia o ZIP inteiro para uma pasta sem espaços protegidos.
2) Execute um dos atalhos:
   - run_open_world_simulation.bat
   - run_advanced_world.bat
3) Os .bat usam primeiro `.venv\Scripts\python.exe` e, se faltar, o runtime embed `python-3.10.11-embed-amd64\python.exe`. As dependências (incluindo FBX) já estão em `.venv\Lib\site-packages`.

Conteúdo offline incluso (FBX SDK):
- Instalador original: `fbx202037_fbxpythonsdk_win.exe` esse é o SDK necessário para rodar os FBX mas já deixei extraído dentro de uma pasta chamada `Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7/`

- Instale Python 3.10.11 64-bit com "Add python.exe to PATH".
- No terminal (nesta pasta):
  ```
  python portable_env_builder.py --offline --wheel-dir "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7"
  ```
  (ou manual)
  ```
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install --no-index --find-links "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7" -r requirements.txt
  ```
- Se a pasta de arquivos extraidos nao existir, rode `fbx202037_fbxpythonsdk_win.exe` nesta pasta e extrai-los novamente.

Requisitos rapidos:
- Windows 64 bits com driver OpenGL 3.3+ ativo.
- GPU dedicada ou integrada recente.
- Permissao para capturar o mouse enquanto a janela roda.

Rodar pelo VSCode (fallback):
- Selecione o interpretador `.venv\Scripts\python.exe` no VSCode.
- Terminal na raiz:
  ```
  python "05 - open_world_simulation.py"
  python "05 - advanced_world.py"
  ```
- Para depurar, use um `launch.json` apontando para o script e o Python da .venv.

Dicas:
- Import `fbx` falhou? Verifique se esta em Python 3.10 e instale o wheel local (ver comandos acima).
- `runtime_bootstrap.py` ja tenta relancar com um Python que tenha o SDK; opcionalmente defina `FBX_PYTHON_EXECUTABLES` com outros pythons.
- Nao renomeie/mova as pastas FBX models, Textures, Heightmaps, python-3.10.11-embed-amd64 ou .venv.
