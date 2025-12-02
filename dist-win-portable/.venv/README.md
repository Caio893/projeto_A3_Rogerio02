# Ambiente virtual portátil (placeholder)

Esta pasta é o destino dos pacotes usados pelo runtime portátil.
Para preenchê-la automaticamente, execute `portable_env_builder.py`
com Python 3.10 em uma máquina Windows e acesso às dependências
(listadas em `dist-win-portable/requirements.txt`).

> Como o ambiente desta execução não tem acesso à internet nem aos wheels
> do SDK do FBX, a `site-packages` permanece vazia aqui. Após baixar os
> wheels (incluindo `fbx-2020.3.7-cp310-none-win_amd64.whl`), rode o builder
> para instalar tudo e deixar os `.bat` funcionais.
