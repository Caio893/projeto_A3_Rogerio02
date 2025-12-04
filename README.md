Guia de instalação de ambiente
## Esse projeto já tem um ambiente virtual .venv com todas as dependencias necessárias
---------------------------------

Criar um ambiente usando o .venv dentro do projeto
- Com vsCode aberto no terminal na pasta do projeto:

  `python -m venv .venv`
  
  `.\.venv\Scripts\activate`

  `py.venv\Scripts\activate`

  `python -m pip install --upgrade pip`

  `python -m pip install -r requirements.txt`

Rodar pelo terminal do VSCode usando o python e libs da .venv:
- Selecione o interpretador `.venv\Scripts\python.exe` no VSCode.
- OPCIONAL ABAIXO
- Adicione o python.exe dentro de python-3.10.11-embed-amd64 como seu interpretador.
- Dentro do vsode aperte Control + Shift + P
- Vá em ou digite "Escolher Interpretador"
- Clique em Insira o caminho do interpretador
- E escolha o python.exe na pasta python-3.10.11-embed-amd64:

- Terminal na raiz:
  ```
  python "primeira pessoa.py"
  python "3º pessoa.py"
  pyton "GLB Mundo Virtual.py"
  ```

## Essa parte abaixo foi feita no Pc da faculdade, apenas o guia acima funcionou em outra maquina além da minha de casa que já tinha Python.


Guia mais técnico para instalar o ambiente com .venv

1) Extraia o ZIP inteiro para uma pasta sem espaços protegidos.

Conteúdo offline incluso OPCIONAL!!!!!!!(FBX SDK):
- Instalador original: `fbx202037_fbxpythonsdk_win.exe` esse é o SDK necessário para rodar os FBX mas já deixei extraído dentro de uma pasta chamada `Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7/`

- Instale Python 3.10.11 64-bit com "Add python.exe to PATH".
- No terminal (nesta pasta):
  ```
  python portable_env_builder.py --offline --wheel-dir "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7"
  ```
  (Guia abaixo feito no PC da faculdade)
  ```
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install --no-index --find-links "Arquivos SDK para extraidos do fbx202037_fbxpythonsdk_win.exe/2020.3.7" -r requirements.txt
  ```
Rodar pelo terminal do VSCode usando o python e libs da .venv:
- Selecione o interpretador `.venv\Scripts\python.exe` no VSCode.
- Terminal na raiz:
  ```
  python "primeira pessoa.py"
  python "3º pessoa.py"
  python "GLB Mundo Virtual.py"
  ```

Requisitos rapidos:
- Windows 64 bits com driver OpenGL 3.3+ ativo.
- GPU dedicada ou integrada recente.
- Permissao para capturar o mouse enquanto a janela roda.

