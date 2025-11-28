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
