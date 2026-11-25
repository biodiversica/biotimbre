# Instalação

*[English](../en/installation.md) · Português (Brasil)*

**Sem Python.** O instalador configura tudo o que é necessário e adiciona um
atalho **biotimbre** que abre a [interface web](web-interface.md). Executá-lo
novamente atualiza a instalação.

No Windows, abra o PowerShell e execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
```

No macOS e no Linux, abra um terminal e execute:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
```

O instalador primeiro lista o que vai fazer e pede confirmação antes de
alterar qualquer coisa. Em seguida, executa as seguintes etapas:

1. Instala o [uv](https://docs.astral.sh/uv/) em `~/.local/bin`, caso ainda
   não esteja presente. O uv baixa o próprio Python, de modo que nenhuma
   instalação prévia é necessária, e qualquer instalação de Python existente
   permanece inalterada.
2. No Windows, instala o Microsoft Visual C++ Redistributable (x64) atual, se
   o presente for anterior à versão 14.40, a exigida pelos modelos. O Windows
   pede permissão de administrador para esta etapa.
3. Executa `uv tool install --python 3.12 "biotimbre[gui]"`, que instala o
   biotimbre com a interface web e adiciona `biotimbre` ao PATH.
4. Cria uma pasta de trabalho, `~/biotimbre`, na qual a interface é iniciada e
   oferece a criação da primeira biblioteca.
5. Adiciona um atalho **biotimbre** que executa `biotimbre gui` nessa pasta:
   um atalho na Área de Trabalho e no menu Iniciar no Windows,
   `biotimbre.command` na Mesa do macOS e uma entrada no menu de aplicativos
   no Linux. O atalho abre uma janela de terminal ao lado do navegador. Essa
   janela executa o servidor, e fechá-la (ou pressionar Ctrl+C) encerra o
   servidor.

Duas variáveis de ambiente alteram os padrões do instalador:
`BIOTIMBRE_PACKAGE` (o pacote a instalar, por exemplo `biotimbre` sem a
interface) e `BIOTIMBRE_WORKDIR` (a pasta de trabalho). `BIOTIMBRE_YES=1`
dispensa a confirmação, para instalar ou desinstalar a partir de um script.

**Desinstalação.** Execute o instalador novamente com `BIOTIMBRE_UNINSTALL`
definida. Ele lista o que vai remover e pede confirmação antes. No Windows,
feche antes a janela do biotimbre e execute no PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c '$env:BIOTIMBRE_UNINSTALL=1; irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex'
```

No macOS e no Linux:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | BIOTIMBRE_UNINSTALL=1 sh
```

Isso remove o biotimbre e o atalho. Com `BIOTIMBRE_UNINSTALL=all` no lugar de
`1`, remove também os modelos baixados e o que a interface lembra (idioma,
tema, bibliotecas recentes). As bibliotecas nunca são removidas, nem o uv, que
outros programas podem usar. Um simples `uv tool uninstall biotimbre` remove
apenas o programa e deixa um atalho que não funciona mais: apague-o à mão (no
Windows, **biotimbre** na Área de Trabalho e no menu Iniciar).

**Com uv ou pip.** A interface web é um extra opcional, de modo que uma
instalação sem interface ou apenas de linha de comando continua pequena:

```bash
uv tool install --python 3.12 "biotimbre[gui]"   # com a interface web
uv tool install --python 3.12 biotimbre          # apenas linha de comando
pip install "biotimbre[gui]"                     # ou com pip, no ambiente atual
```

No Windows, os modelos exigem um
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
atual (14.40 ou mais recente). Sem ele, o biotimbre continua funcionando, mas
os modelos não carregam, e a interface e a linha de comando informam o motivo.

`uv tool upgrade biotimbre` atualiza e `uv tool uninstall biotimbre`
desinstala. Para trabalhar no próprio biotimbre, consulte
[Desenvolvimento](development.md); a partir de uma cópia do código-fonte,
prefixe os comandos deste documento com `uv run`.

**Primeira execução.** A primeira inicialização baixa os modelos Perch e
BirdNET (cerca de 70 MB) e os armazena em cache; depois disso, nenhuma conexão
de rede é necessária. Uma biblioteca pode ser criada pela interface ou pela
linha de comando:

```bash
biotimbre init --name "Minha Biblioteca"
```

A opção `--name` é obrigatória na primeira inicialização de uma biblioteca.
Cada biblioteca possui um nome e um UUID, gerado uma única vez e estável por
toda a sua existência, de modo que ela continue identificável após ser
compartilhada: copiar o diretório de dados transfere ambos. O nome e o UUID são
exibidos no cabeçalho da interface e por `biotimbre stats`. Executar
`init --name "..."` novamente renomeia a biblioteca.
