# ((-)) biotimbre

*[English](README.md) · Português (Brasil)*

O Biotimbre é uma ferramenta para criar e gerenciar bibliotecas de templates de
sons de animais, com busca por similaridade baseada em embeddings de áudio. Os
embeddings são calculados com modelos em formato ONNX por meio do `onnxruntime`; o Perch v2 e
o BirdNET 2.4 são incluídos automaticamente.

Cada biblioteca armazena as gravações, seus metadados e seus vetores de
embedding em uma única pasta. As bibliotecas podem ser utilizadas por meio de
uma interface web ou da linha de comando, pesquisadas por som e por metadados,
exportadas para análise e compartilhadas, item por item, com outras
bibliotecas.

## Instalação

**Sem Python.** O instalador configura tudo o que é necessário e adiciona um
atalho **biotimbre** que abre a interface web. Executá-lo novamente atualiza a
instalação.

No Windows, abra o PowerShell e execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
```

No macOS e no Linux, abra um terminal e execute:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
```

O instalador lista o que vai fazer e pede confirmação antes de alterar
qualquer coisa.

**Desinstalação.** Execute o instalador novamente com `BIOTIMBRE_UNINSTALL`
definida. No Windows, feche antes a janela do biotimbre e execute no
PowerShell:

```powershell
$env:BIOTIMBRE_UNINSTALL=1; irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex
```

No macOS e no Linux:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | BIOTIMBRE_UNINSTALL=1 sh
```

As bibliotecas nunca são removidas.

**Com uv ou pip.** A interface web é um extra opcional, de modo que uma
instalação sem interface ou apenas de linha de comando continua pequena:

```bash
uv tool install --python 3.12 "biotimbre[gui]"   # com a interface web
uv tool install --python 3.12 biotimbre          # apenas linha de comando
pip install "biotimbre[gui]"                     # ou com pip, no ambiente atual
```

[Instalação](docs/pt-BR/installation.md) descreve o que o instalador faz, suas
variáveis de ambiente, o que a desinstalação remove e o Visual C++
Redistributable de que os modelos precisam no Windows.

## Início Rápido

```bash
biotimbre init --name "Minha Biblioteca"     # cria uma biblioteca em ./data
biotimbre gui                                # a interface web, em http://127.0.0.1:8080
biotimbre add canto.wav --sidecar canto.txt  # adiciona uma gravação com seus metadados
biotimbre search consulta.wav -k 5           # os cinco itens mais semelhantes
```

A primeira inicialização baixa os modelos Perch e BirdNET (cerca de 70 MB) e os
armazena em cache; depois disso, nenhuma conexão de rede é necessária.
`biotimbre --help` lista todos os comandos, e `biotimbre COMANDO --help`
explica cada uma de suas opções em detalhe, com exemplos.

## Documentação

| Página | Conteúdo |
|---|---|
| [Instalação](docs/pt-BR/installation.md) | O instalador, a desinstalação, uv e pip, a primeira execução |
| [Bibliotecas](docs/pt-BR/libraries.md) | Onde uma biblioteca é armazenada, como abri-la, a estrutura dos dados |
| [Interface Web](docs/pt-BR/web-interface.md) | As páginas Biblioteca e Ferramenta de busca, o modo somente busca, a escolha da biblioteca, o mapa, as opções de exibição |
| [Interface de Linha de Comando](docs/pt-BR/cli.md) | Todos os comandos, com exemplos |
| [Metadados](docs/pt-BR/metadata.md) | Campos, identificadores, um biotimbre por espécie, arquivos sidecar, tabelas |
| [Preparação do Áudio](docs/pt-BR/audio-preparation.md) | Recorte, filtragem e normalização; gravações ultrassônicas e o parâmetro de velocidade |
| [Funcionamento da Busca](docs/pt-BR/search.md) | O limite da consulta, os filtros, a ordenação por um modelo |
| [Exportação de Dados](docs/pt-BR/exporting.md) | Áudio, embeddings e tabelas de metadados |
| [Compartilhamento de Itens entre Bibliotecas](docs/pt-BR/sharing.md) | Pacotes, regras de importação, importação de outra biblioteca |
| [Integração com o iNaturalist](docs/pt-BR/inaturalist.md) | Adicionar e buscar com a gravação de uma observação |
| [Configuração](docs/pt-BR/configuration.md) | O `biotimbre.toml`, com cada configuração e seu padrão |
| [Adição de um Modelo](docs/pt-BR/adding-a-model.md) | Modelos de embedding ONNX personalizados |
| [Desenvolvimento](docs/pt-BR/development.md) | Trabalhar no próprio biotimbre, testes e CI |

## Licença

[MIT](LICENSE)
