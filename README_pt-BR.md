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
atalho **biotimbre** que abre a [interface web](#interface-web). Executá-lo
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
[Desenvolvimento](#desenvolvimento); a partir de uma cópia do código-fonte,
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

## Localização da Biblioteca

Por padrão, a biblioteca é armazenada em `./data`, relativo ao diretório atual.
A localização pode ser definida de três maneiras:

```bash
# 1. Para uma única execução
biotimbre --data-dir /media/drive/biotimbre list

# 2. Com um arquivo de configuração no diretório de trabalho (ou --config /caminho/para/ele)
echo '[library]
data_dir = "/media/drive/biotimbre"' > biotimbre.toml

# 3. Com uma variável de ambiente apontando para um arquivo de configuração
export BIOTIMBRE_CONFIG=/media/drive/biotimbre/biotimbre.toml
```

`--data-dir` tem precedência sobre o arquivo de configuração. Caminhos
relativos em um arquivo de configuração são resolvidos em relação ao diretório
que contém esse arquivo, e não ao diretório de trabalho, de modo que um arquivo
de configuração e sua biblioteca podem ser movidos juntos.

**Abrir uma biblioteca existente.** Nenhuma etapa adicional é necessária: todo
comando abre o banco de dados e cria o que estiver faltando.

```bash
biotimbre --data-dir /media/drive/biotimbre list
```

`init` pode ser executado com segurança em uma biblioteca já populada; ele
apenas cria o que estiver ausente e baixa os modelos. Os caminhos dos áudios
armazenados são relativos ao diretório de dados, portanto **a pasta inteira
pode ser movida ou renomeada sem invalidar nenhuma referência.**

## Interface Web

```bash
biotimbre gui        # http://127.0.0.1:8080
```

Por padrão, o servidor escuta apenas na interface de loopback; disponibilizá-lo
a outras máquinas exige um `--host` explícito. Os modelos são carregados em
segundo plano, de modo que a interface pode ser utilizada enquanto o cabeçalho
informa o progresso. `--no-map` desativa o mapa de localização, que é o único
componente que acessa a rede por conta própria.

A interface possui duas páginas, **Biblioteca** e **Ferramenta de busca**,
apresentadas como abas. Ambas permanecem carregadas durante a troca entre
elas, de modo que valores digitados, gravações preparadas e resultados de busca
são preservados. A barra de endereços acompanha a página ativa, portanto o
recarregamento, os favoritos e os botões de voltar e avançar do navegador
funcionam normalmente.

### Página Biblioteca

A página Biblioteca é dividida em três seções.

- **Itens da biblioteca** — um mapa com todos os itens e a
  lista de itens. São exibidos também os totais da biblioteca, as velocidades
  presentes e os táxons, cada um com seu número de itens; quando há muitos
  táxons, os mais frequentes são exibidos primeiro e os demais podem ser
  expandidos. Selecionar um item abre uma janela com todos os seus
  metadados, seu espectrograma ou forma de onda, um reprodutor e botões para
  exportá-lo, editá-lo ou excluí-lo. A edição emite um aviso antes de salvar
  uma alteração de `speed`, pois isso exige o recálculo dos embeddings.
  **Exportar CSV** baixa os metadados de todos os itens em uma tabela (consulte
  [Exportação de Dados](#exportação-de-dados)). Marcar itens na lista (a caixa
  do cabeçalho marca uma página inteira) exibe **Exportar N como pacote**, que
  os baixa como um único `.btz` compartilhável, com ou sem os embeddings.
- **Adicionar um item** — envia uma gravação e seus metadados. O espectrograma
  e a taxa de amostragem são exibidos antes que algo seja armazenado, com o
  limite de frequência de cada modelo indicado. A gravação pode ser recortada,
  filtrada e normalizada antes de ser armazenada (consulte
  [Preparação do Áudio](#preparação-do-áudio)). Erros de validação são
  exibidos no campo correspondente.
- **Importar itens** — carrega um pacote `.btz` de outra biblioteca, ou
  adiciona todos os itens de outra biblioteca no mesmo computador, com uma
  prévia do resultado que pode ser ajustada antes que qualquer item seja
  adicionado (consulte
  [Compartilhamento de Itens entre Bibliotecas](#compartilhamento-de-itens-entre-bibliotecas)).

O local da gravação é informado em um único campo, no formato
`latitude, longitude` em graus decimais. Vírgula, ponto e vírgula ou espaços
são aceitos como separadores, de modo que coordenadas copiadas de serviços de
mapas, aplicativos de GPS ou cadernos de campo podem ser coladas diretamente.
**Escolher no mapa** permite definir o local clicando em um mapa.

### Página Ferramenta de busca

A página Ferramenta de busca recebe uma gravação de consulta e retorna os itens
mais semelhantes da biblioteca, ordenados por pontuação.

- A consulta é preparada com os mesmos controles do formulário de adição da
  página Biblioteca (recorte, filtragem, normalização e velocidade), e é a
  cópia preparada que é utilizada na busca. A seleção é limitada a 15 s; para
  um arquivo mais longo, um segmento de no máximo 15 s é selecionado no
  espectrograma.
- **Filtros** restringem a busca por táxons, intervalo de datas, distância de
  um ponto central e ID do biotimbre. Uma busca também pode ser realizada
  apenas com filtros, sem gravação; nesse caso, os itens correspondentes são
  listados sem pontuação, dos mais recentes para os mais antigos.
- **Opções** definem a velocidade da consulta, o número máximo de resultados,
  a métrica de similaridade, a forma de combinar as pontuações dos modelos e
  uma pontuação mínima.

Cada resultado exibe sua pontuação, o detalhamento por modelo, um reprodutor
próprio e um espectrograma no qual está marcada a janela que gerou a
pontuação. A consulta permanece disponível para reprodução no topo da página,
para comparação auditiva. Cada resultado oferece ainda:

- **Detalhes** — uma janela somente leitura com todos os metadados do item,
  como na página Biblioteca, porém sem as opções de edição e exclusão.
- **Exportar** — o áudio do item, seus embeddings ou um pacote para
  compartilhamento.
- Um ID de biotimbre abreviado que, ao ser clicado, copia o UUID completo para
  a área de transferência, para uso no filtro de ID do biotimbre ou em outra
  biblioteca.

**Exportar CSV**, ao lado da contagem de resultados, baixa os resultados
exibidos em uma tabela. Marcar resultados (ou **Selecionar tudo**) exibe
**Exportar N como pacote**, que baixa os marcados como um único `.btz`
compartilhável; uma nova busca começa sem nada marcado. **Adicionar à
biblioteca…** abre o formulário de
adição da página Biblioteca com a gravação da consulta e todas as suas
configurações (e, para uma gravação obtida do iNaturalist, os metadados da
observação), para que uma nova gravação possa ser adicionada à biblioteca.
Nada é armazenado até que o formulário seja salvo. Essa opção não está
disponível no modo somente busca.

### Modo Somente Busca

```bash
biotimbre --data-dir ./library gui --search-only
```

Esse modo permite que outras pessoas pesquisem uma biblioteca sem poder
modificá-la. A página Biblioteca não é adicionada ao GUI (`/` redireciona, portanto a
página e seus manipuladores nunca são criados), o seletor de bibliotecas não
fica disponível e **o banco de dados é aberto somente para leitura**, o que
torna isso uma garantia, e não apenas uma aba oculta. O cabeçalho exibe
`Somente busca` para indicar o modo.

A biblioteca precisa já existir: `--search-only` exige uma biblioteca indicada
com `--data-dir` (ou `library.data_dir`) e não cria nada, de modo que um
caminho digitado incorretamente é tratado como erro, e não como uma nova
biblioteca vazia. Uma biblioteca cujo esquema precisa de atualização também é
recusada, pois a migração é uma escrita; essa biblioteca deve primeiro ser
aberta uma vez sem a opção.

Todos os recursos da página Ferramenta de busca permanecem disponíveis,
inclusive o download do áudio de um resultado ou sua exportação como pacote. O
modo impede que a biblioteca seja *alterada*; ele não restringe a leitura do
seu conteúdo. Para torná-la acessível a partir de outra máquina, combine-o com
`--host`:

```bash
biotimbre --data-dir ./library gui --search-only --host 0.0.0.0
```

A configuração equivalente, para uma máquina que deve apenas atender buscas:

```toml
[ui]
search_only = true
```

### Seleção da Biblioteca na Inicialização

Quando `biotimbre gui` é iniciado sem `--data-dir`, ele não abre nem cria uma
biblioteca por conta própria. A primeira página pergunta, com duas opções:

- **Criar uma nova biblioteca** recebe um nome e um local. A biblioteca é
  criada em uma pasta própria dentro desse local, com o nome dela (*Cigarras
  da Mata Atlântica* vira `cigarras-da-mata-atlântica`), e o caminho completo
  é exibido antes que qualquer coisa seja criada. Uma pasta que já existe e
  não está vazia é recusada.
- **Abrir uma biblioteca existente** navega até uma biblioteca; as pastas que
  contêm uma são sinalizadas. Uma pasta sem biblioteca é informada, em vez de
  ser transformada em uma nova, de modo que um caminho digitado errado não
  produz uma biblioteca vazia por acidente.

As bibliotecas abertas a partir do mesmo navegador são oferecidas novamente no
acesso seguinte. Um `data_dir` em um arquivo de configuração não pula esta
página: a biblioteca indicada é oferecida primeiro ou, se ainda não existir,
oferecida para ser criada nesse caminho. Apenas `--data-dir` abre uma
biblioteca diretamente. Os demais comandos, e `gui --search-only`, que não tem
seletor, continuam usando o `data_dir` configurado.

A navegação não se restringe a `$HOME`, pois as bibliotecas são frequentemente
mantidas em unidades externas e arquivos montados. Como essa listagem é a única
parte da interface em que uma requisição indica um caminho do sistema de
arquivos, ela existe apenas enquanto nenhuma biblioteca está aberta e é
removida assim que uma biblioteca é aberta.

## Interface de Linha de Comando

### Referência de Comandos

```bash
biotimbre init [--name NOME] [--skip-models]   # --name é obrigatório até que um nome seja definido
biotimbre add AUDIO [--sidecar ARQUIVO | opções de metadados] [opções de processamento]
biotimbre add --inat OBSERVAÇÃO [opções de metadados] [opções de processamento]
biotimbre import-dir DIR                       # todo arquivo acompanhado de um sidecar
biotimbre import-table TABELA.csv [--dry-run]  # uma linha por gravação ou observação
biotimbre list [--biotimbre-id ID] [--limit N]
biotimbre show ITEM_ID
biotimbre edit ITEM_ID [opções de metadados] [--clear CAMPO...]
biotimbre delete ITEM_ID [-y] [--keep-audio]
biotimbre search [AUDIO | --inat OBSERVAÇÃO] [-k N] [filtros]   # sem nenhum: só pelos filtros
biotimbre models                               # modelos registrados e contagem de vetores
biotimbre backfill MODEL_ID                    # calcula embeddings dos itens que não os têm para este modelo
biotimbre export-npy ITEM_ID MODEL_ID SAIDA [--with-audio]
biotimbre export-csv SAIDA                     # metadados de todos os itens em CSV
biotimbre export-bundle [ITEM_ID...] [--items-from CSV] -o ARQUIVO [--no-embeddings]
biotimbre import-bundle ARQUIVO [--dry-run] [opções]
biotimbre import-library DIR [--dry-run] [opções]    # adiciona os itens de outra biblioteca a esta
biotimbre stats
biotimbre gui [--host H] [--port P] [--no-browser] [--no-map] [--search-only]
```

Opções globais: `--data-dir`, `--config`, `--models`. Elas vêm antes do
comando, por exemplo `biotimbre --data-dir ~/minha-biblioteca list`.

`biotimbre --help` lista todos os comandos, e `biotimbre COMANDO --help`
explica em detalhe cada um de seus argumentos e opções, com exemplos (em
inglês).

### Exemplos

```bash
# Adicionar uma gravação com metadados explícitos. As coordenadas são
# informadas em opções separadas, pois o argparse não aceita
# "--location -26,-48" (o valor começa com um sinal de menos).
biotimbre add katydid.wav \
    --date 2021-03-05 --time 22:10:00 \
    --latitude -26.9 --longitude -48.7 \
    --window-start 12 --window-end 17 \
    --taxa orthoptera insecta --device Audiomoth \
    --highpass 8000 --lowpass 40000 --speed 0.25

# Buscar, com filtros aplicados antes da pontuação
biotimbre search query.wav -k 5 \
    --taxa cicadidae --latitude -26 --longitude -48 --radius-km 50 \
    --date-from 2020-01-01

# Uma consulta mais longa que o limite exige um segmento explícito
biotimbre search dawn_chorus.wav --start-time 92 --end-time 99

# Filtrar a consulta antes de calcular seus embeddings (a biblioteca não muda)
biotimbre search noisy.wav --apply-highpass 1500

# Buscar só pelos metadados: sem gravação de consulta, mais recentes primeiro
biotimbre search --taxa cicadidae --date-from 2024-01-01 -k 50

# Armazenar só parte de uma gravação, filtrada e normalizada, como no formulário
biotimbre add dawn.wav --sidecar dawn.txt --trim 92 99 --apply-highpass 1500 --normalise

# Alterar metadados armazenados; os campos não informados são mantidos
biotimbre edit 12 --taxa "guyalna bonaerensis" --clear device
```

Outras opções de `search`: `--metric {cosine,dot,euclidean}`,
`--fusion {max,mean,MODEL_ID}`, `--min-score`, `--query-speed`, `--taxa-all`,
`--date-to`, `--biotimbre-id`, `--exclude`, `--apply-lowpass`, `--apply-filter-order` e,
numa busca só pelos filtros, `--speed` (itens armazenados nessas velocidades).

`--highpass`/`--lowpass` em `add` e `edit` *registram* uma filtragem já feita na
gravação; `--apply-highpass`/`--apply-lowpass` *filtram*. Todas as opções são
explicadas em detalhe por `biotimbre COMANDO --help` (em inglês).

## Metadados

### Campos

**Obrigatórios:** `date` (AAAA-MM-DD), `time` (HH:MM:SS) e o local (latitude e
longitude em graus decimais).

**Opcionais:**

| Campo | Descrição |
|---|---|
| `biotimbre_id` | Omita para criar um novo biotimbre; informe um ID existente para registrar o item nele. |
| `taxa` | Etiquetas de táxons. Vazio significa desconhecido. |
| `sample_window` | Início e fim, em segundos, do item *dentro da gravação completa*. Só tem significado em conjunto com `link_to_full_recording`. As duas extremidades devem ser informadas, ou nenhuma. |
| `device`, `microphone`, `recorded_by` | Equipamento de gravação e responsável pela gravação. |
| `filtered_band`, `filter_type`, `filter_order` | Filtragem aplicada à gravação. |
| `temperature`, `humidity` | Condições no momento da gravação. |
| `speed` | Velocidade de reprodução utilizada no cálculo dos embeddings (consulte [Gravações Ultrassônicas](#gravações-ultrassônicas-e-o-parâmetro-de-velocidade)). |
| `link_to_full_recording` | URL da gravação completa. |
| `notes` | Texto livre. |

### Identificadores

`item_id` é atribuído automaticamente, nunca é reutilizado e é específico de
uma biblioteca; trata-se apenas de um identificador de linha.

Um **biotimbre** é um ser sonoro: um sonotipo cuja espécie pode ser
desconhecida. Por isso, `biotimbre_id` é um UUID, e não um contador local da
biblioteca, de modo que o mesmo sonotipo seja identificado de forma consistente
quando bibliotecas forem compartilhadas ou mescladas. Vários itens podem
compartilhar um mesmo `biotimbre_id` — a mesma espécie ou indivíduo, com
conteúdos sonoros diferentes.

### Uma Espécie, Um Biotimbre

Uma espécie é uma identificação completa; por isso, a biblioteca atribui a ela
uma única identidade. Um item etiquetado com uma espécie já presente na
biblioteca — por exemplo, `Guyalna bonaerensis` — é registrado sob o
`biotimbre_id` já existente dessa espécie, em vez de iniciar um segundo
biotimbre para o mesmo animal. As etiquetas são armazenadas em letras
minúsculas, portanto o uso de maiúsculas não importa.

Essa regra se aplica apenas a binômios. Uma categoria acima de espécie
(`cicadidae`) não determina qual ser foi gravado, e a nomenclatura aberta
(`Guyalna sp.`, `cf.`, `aff.`) declara explicitamente que a espécie é
desconhecida; em ambos os casos, um novo biotimbre é iniciado. Um
`biotimbre_id` informado explicitamente sempre tem precedência.

### Arquivos Sidecar

Um arquivo sidecar contém um campo por linha: o nome do campo seguido de seu
valor, separados por tabulações ou espaços. Um arquivo chamado `GUYVIR.txt` é
associado a `GUYVIR.wav`.

```
taxa	cicadidae
date	2020-01-14
time	09:40:21
sample_window 5,10
location -20,-46
device	Audiomoth
filtered_band	2000,10000
```

O local também pode ser informado em duas linhas, `latitude -20` e
`longitude -46`, os nomes que uma tabela usa. `filtered_band` aceita `-` para um
limite aberto, por exemplo `2000,-`. Linhas
iniciadas por `#` ou `;` são comentários. Um nome de campo não reconhecido é
tratado como erro, e não ignorado, de modo que um erro de digitação não pode
descartar silenciosamente um valor obrigatório.

Um arquivo sidecar também pode pedir que o áudio seja processado antes de ser
armazenado, como fazem os controles do formulário. Estas linhas são instruções,
não descrição:

```
trim                5,10      # armazenar só 5-10 s do arquivo
apply_highpass      2000      # passa-altas Butterworth de fase zero, em Hz
apply_lowpass       10000     # e passa-baixas
apply_filter_order  4         # opcional; padrão 4
normalise           yes       # levar o pico da cópia armazenada a -1 dBFS
```

Os metadados, que descrevem o arquivo como foi entregue, são corrigidos para
descrever o que é armazenado: um recorte move `sample_window` (para `5,10`, ou
5 s adiante de onde o arquivo já estava numa gravação mais longa); um filtro
estreita `filtered_band` e define `filter_type` e `filter_order` se não forem
informados; a normalização acrescenta uma linha às `notes`. `biotimbre add`
aceita as mesmas escolhas como `--trim`, `--apply-highpass`, `--apply-lowpass`,
`--apply-filter-order` e `--normalise`, que substituem a linha do arquivo sidecar para
a mesma escolha. `import-dir` aplica as de cada arquivo sidecar.

### Tabelas

`import-table` adiciona muitos itens a partir de uma tabela CSV, em vez de um
arquivo sidecar por gravação. Cada linha é um item. As colunas são os nomes que
um sidecar aceita, seus campos e linhas de processamento (`latitude` e
`longitude` ou `location`), mais as colunas próprias da tabela, `file`, `inat` e
`inat_sound`:

```
file,inat,date,time,latitude,longitude,taxa,trim,normalise
GUYVIR.wav,,2020-01-12,09:40:21,-26.9,-48.7,cicadidae,"1,4",yes
,400139850,,,,,cicadidae,,
```

```bash
biotimbre import-table gravacoes/tabela.csv --dry-run   # verificar, sem alterar nada
biotimbre import-table gravacoes/tabela.csv
```

- **`file`**: um nome sozinho é procurado na pasta da tabela, que fica junto às
  gravações; um caminho relativo a essa pasta também funciona, e um caminho
  absoluto é usado como está. Um caminho escrito em outro sistema operacional
  (`C:\...` lido no Linux, `/home/...` no Windows) é apontado como tal.
- **`inat`**: o ID ou a URL de uma observação do iNaturalist no lugar de
  `file`. A gravação é baixada, e os metadados da observação preenchem o que a
  linha deixa vazio, como em `add --inat`; `inat_sound` escolhe uma das
  gravações. As observações são pedidas de 200 em 200 e as gravações uma a uma,
  com no máximo uma requisição por segundo e menos de 5 GB de áudio por hora,
  seguindo as práticas recomendadas pelo iNaturalist; uma requisição recusada
  por excesso é repetida após a espera indicada. Esperas longas são anunciadas.
- A tabela pode ser separada por vírgulas, ponto e vírgula ou tabulações. Numa
  tabela separada por ponto e vírgula, um número sozinho pode usar vírgula
  decimal (`-26,9`); pares como `trim` sempre exigem ponto decimal.
- Uma coluna desconhecida é um erro. Uma célula vazia não informa nada. Todas
  as linhas são tentadas, e cada uma é informada com o número da sua linha.
- Uma gravação que já está na biblioteca é **ignorada**, de modo que uma
  importação interrompida pode simplesmente ser repetida; `--keep-duplicates`
  a armazena de novo.

## Preparação do Áudio

### Recorte, Filtragem e Normalização

O formulário de adição processa a gravação enviada antes de armazená-la. Um
segmento é selecionado arrastando sobre o espectrograma; um corte passa-altas
e/ou passa-baixas aplica um filtro Butterworth de fase zero; e a normalização
ajusta a gravação para que sua amostra mais alta atinja um nível escolhido. Um
segundo espectrograma mostra exatamente o que será armazenado, e o reprodutor
acompanha o segmento selecionado. A página Ferramenta de busca oferece os
mesmos controles para a consulta, com a seleção limitada a 15 s.

A biblioteca armazena o áudio processado, e os metadados são preenchidos de
acordo: o segmento se torna `sample_window` (a posição do item na gravação
original), e os cortes se tornam `filtered_band`, com `filter_type` definido
como "Butterworth" e a ordem utilizada. Ambos permanecem editáveis, pois
somente o usuário sabe se o arquivo enviado já era, ele próprio, um trecho. A
normalização é registrada nas notas.

O áudio armazenado é sempre mono, pois é isso que todos os modelos recebem e o
que um espectrograma mostra. Uma gravação com vários canais é, portanto,
convertida para mono pela média dos canais, tanto pelo formulário de adição
quanto por `biotimbre add`, e o reprodutor apresenta a versão mono. O áudio
armazenado também é sempre WAV, como declaram seu nome, seus pacotes e sua
reprodução: uma gravação em outro formato (FLAC, OGG, MP3, AIFF) é convertida,
uma sem perdas mantendo cada amostra na sua própria profundidade de bits e uma
com perdas decodificada, o que não custa mais qualidade. Um arquivo WAV mono
que não é recortado nem filtrado é armazenado exatamente como foi recebido.

### Gravações Ultrassônicas e o Parâmetro de Velocidade

Antes do cálculo dos embeddings, o áudio é reamostrado para a taxa de cada
modelo — 32 kHz para o Perch e 48 kHz para o BirdNET —, o que remove todo o
conteúdo acima de 16 kHz e 24 kHz, respectivamente. Uma esperança (katydid) a
35 kHz seria representada como quase silêncio e nunca corresponderia a nada.

`--speed` reinterpreta a taxa de amostragem de origem antes da reamostragem.
Com `0.25`, o conteúdo em 40 kHz é deslocado para 10 kHz, dentro da faixa de
ambos os modelos:

```bash
biotimbre add katydid.wav --sidecar katydid.txt --speed 0.25
```

O áudio armazenado nunca é modificado, e a reprodução não é afetada.

**Não é necessário conhecer a velocidade de um item ao realizar uma busca.** A
consulta é processada automaticamente em todas as velocidades presentes na
biblioteca, de modo que uma gravação em sua velocidade natural ainda encontra
um item desacelerado. Os resultados informam a velocidade em que houve
correspondência. Definir uma velocidade da consulta (`search --query-speed`, ou
*Velocidade da busca* na página de Busca) calcula os embeddings da consulta
só nessa velocidade e busca apenas os itens armazenados nela.

Se uma `filtered_band` declarada exceder o limite de um modelo, `add` sugere um
valor de velocidade. Uma velocidade adequada leva a frequência mais alta de
interesse para abaixo de 16 kHz; um fator de 1/4 ou 1/8 é adequado para a
maioria das gravações de insetos e morcegos. Utilizar o mesmo valor em
gravações semelhantes é mais importante do que o valor exato escolhido.

## Funcionamento da Busca

Uma busca utiliza no máximo **15 s** de áudio de consulta. A pontuação
considera o par de janelas com maior correspondência, de modo que uma consulta
longa permitiria que um fragmento arbitrário determinasse o resultado.
Arquivos dentro do limite são utilizados por inteiro; arquivos mais longos
exigem `--start-time` e `--end-time` (ou uma seleção na interface) e são
recusados, em vez de truncados. Apenas os quadros selecionados são
decodificados.

Os filtros são aplicados antes da pontuação, de modo que os resultados são
ordenados dentro do conjunto filtrado, e não filtrados posteriormente. Repetir
a mesma consulta com filtros diferentes reutiliza os embeddings em cache e
retorna imediatamente.

## Exportação de Dados

### Áudio e Embeddings

A janela de detalhes do item e cada resultado de busca oferecem um menu
**Exportar** com o arquivo de áudio armazenado, os embeddings de cada modelo e
um pacote para compartilhamento.

Cada arquivo exportado contém no máximo uma janela, pois a janela — um vetor
sobre um intervalo de tempo — é a unidade com significado. Um modelo com uma
única janela (por exemplo, o Perch em um item curto) é exportado como um único
`item_12_perch_v2.npy`, contendo um vetor unidimensional. Várias janelas são
exportadas como `item_12_birdnet_v2.4.zip`, com um `.npy` por janela, cada um
nomeado com seu índice e intervalo de tempo
(`item_12_birdnet_v2.4_w00_0.00-3.00s.npy`). `biotimbre export-npy` gera
exatamente os mesmos arquivos, de modo que um download pela interface e uma
exportação pela linha de comando são intercambiáveis.

### Embeddings com o Áudio de Entrada do Modelo

A opção **embeddings + trechos de áudio** do menu Exportar, ou
`biotimbre export-npy --with-audio`, inclui também o áudio de cada janela tal
como o modelo o recebeu: com a velocidade ajustada, reamostrado para a taxa de
amostragem do modelo e completado até o comprimento da janela. O resultado é
sempre um arquivo zip (`item_12_birdnet_v2.4_with_audio.zip`) no qual cada
`.npy` é acompanhado de um arquivo WAV com o mesmo nome
(`item_12_birdnet_v2.4_w00_0.00-3.00s.wav`). Se as janelas armazenadas de um
item não corresponderem mais ao seu áudio, a exportação é recusada e o item
precisa ter seus embeddings recalculados antes.

### Tabelas de Metadados (CSV)

Os metadados dos itens podem ser exportados em uma tabela CSV, para uso em uma
planilha ou em um dataframe, com uma linha por item e uma coluna por valor
armazenado. Valores compostos são divididos em colunas separadas (por exemplo,
`latitude` e `longitude`), valores ausentes ficam vazios e uma coluna para cada
modelo registrado indica o número de janelas de embedding. O arquivo utiliza a
codificação UTF-8 e vírgula como separador.

- **Página Biblioteca:** **Exportar CSV**, na seção *Itens da biblioteca*, ou
  `biotimbre export-csv SAIDA`, exporta todos os itens
  (`biotimbre_items.csv`).
- **Página Ferramenta de busca:** **Exportar CSV**, ao lado da contagem de
  resultados, exporta apenas os resultados exibidos, na ordem de
  classificação (`biotimbre_search_results.csv`). Cada linha é precedida pela
  posição, pela pontuação combinada, pelo melhor modelo, pela janela
  correspondente, pela distância até o centro do filtro, pela velocidade da
  consulta e pela pontuação de cada modelo. Em uma busca sem gravação, as
  colunas de pontuação ficam vazias.

## Compartilhamento de Itens entre Bibliotecas

`biotimbre_id` é um UUID para que um ser sonoro tenha a mesma identidade em
todas as bibliotecas que o contêm. Os itens são transferidos como **pacotes**:
um único arquivo `.btz` contendo os metadados de um item, seu áudio armazenado
e seus vetores de embedding.

### Criação e Importação de Pacotes

Os pacotes podem ser criados pelo menu Exportar da interface ou pela linha de
comando:

```bash
biotimbre export-bundle 12 -o guyalna.btz          # um item, com seus vetores
biotimbre export-bundle 12 13 14 -o batch.btz      # vários itens
biotimbre export-bundle 12 -o audio-only.btz --no-embeddings   # só metadados e áudio
```

Resultados de busca podem ser compartilhados da mesma forma. `search --csv`
grava os resultados como a página de busca os exporta; apague numa planilha as
linhas das correspondências erradas, e `--items-from` usa os itens que
restarem, desse arquivo ou de qualquer CSV com uma coluna `item_id`:

```bash
biotimbre search query.wav --taxa cicadidae --csv hits.csv
biotimbre export-bundle --items-from hits.csv -o checked.btz
```

O destinatário pode importar o pacote na seção *Importar itens*
da página Biblioteca, que exibe primeiro uma prévia, ou pela linha de comando.
Uma simulação (`--dry-run`) informa o que aconteceria sem alterar nada:

```bash
biotimbre import-bundle guyalna.btz --dry-run
biotimbre import-bundle guyalna.btz
```

Um arquivo `.btz` é um arquivo zip comum. Ele contém `item.json` (os metadados
de referência), `item.txt` (os mesmos campos no formato sidecar, para
leitura), o áudio como arquivo WAV mono e os vetores como arrays `.npz`.

### Regras de Importação

**Gravações já presentes na biblioteca.** As gravações são comparadas pelo
resumo criptográfico (digest) do arquivo armazenado, de modo que duas cópias
são reconhecidas independentemente do nome atribuído por cada biblioteca. Sob
o mesmo biotimbre, o item recebido é o mesmo item e é ignorado. Sob um
biotimbre *diferente*, as duas bibliotecas divergem quanto a qual ser se trata;
isso é informado e o item é ignorado. `--duplicates add` altera esse
comportamento.

**Espécies que já possuem um biotimbre.** Quando duas bibliotecas catalogaram
*Guyalna bonaerensis* separadamente, há dois UUIDs para um mesmo animal.
Seguindo a regra de uma espécie, um biotimbre, o item recebido é incorporado ao
biotimbre local, e o ID do remetente é informado. `--species keep-incoming`
mantém o ID do remetente, deixando duas identidades para uma mesma espécie até
que sejam mescladas.

**Embeddings.** Os vetores só são comparáveis quando toda a configuração de
pré-processamento coincide. Por isso, o pacote inclui a assinatura completa de
cada modelo — origem, arquivo, revisão, taxa de amostragem, duração da janela,
sobreposição e dimensão —, e a biblioteca receptora a compara campo a campo.
Vetores compatíveis são armazenados como estão, poupando uma etapa de
inferência; vetores incompatíveis são recusados, o campo divergente é
indicado e os embeddings são recalculados. Vetores de um modelo que a
biblioteca receptora não utiliza são sempre recusados. `--distrust-embeddings`
descarta todos os vetores recebidos e os recalcula a partir do áudio, que é a
única parte de um pacote que pode ser verificada.

Tudo o que o pacote não fornece é calculado na importação, de modo que um item
importado pode ser encontrado em buscas imediatamente. `--no-embeddings` deixa
essas lacunas para serem preenchidas posteriormente com `backfill`, o que é
útil para importações em grande volume em uma máquina sem os arquivos dos
modelos.

### Importando de Outra Biblioteca

Todos os itens de outra biblioteca no mesmo computador podem ser importados de
uma só vez, sem criar um pacote antes. Na seção *Importar itens* da página
Biblioteca, **Importar de outra biblioteca** recebe o diretório da outra
biblioteca (o que contém seu `library.db`), digitado ou escolhido com
**Procurar**, e **Prévia da importação** abre a mesma prévia e as mesmas
configurações de um pacote. Pela linha de comando, `import-library` aceita as
mesmas opções que `import-bundle`:

```bash
biotimbre import-library ~/campo-2024 --dry-run
biotimbre import-library ~/campo-2024
```

A outra biblioteca é aberta somente para leitura e não é alterada. Cada um de
seus itens é tratado como uma entrada de um pacote vindo dessa biblioteca, de
modo que as [regras de importação](#regras-de-importação) acima se aplicam sem
alteração: gravações já presentes são ignoradas, o que torna seguro repetir uma
importação; espécies são incorporadas ao biotimbre local; e os vetores só são
reaproveitados quando as assinaturas dos modelos coincidem. Itens cujo áudio
armazenado não está no disco ficam de fora e são listados. A outra biblioteca
precisa estar na versão atual do esquema; uma mais antiga é atualizada ao ser
aberta uma vez.

### Segurança

Um pacote recebido de outra pessoa é tratado como não confiável. Apenas os
membros indicados em seu manifesto são lidos, caminhos que escapariam do
arquivo são recusados, tamanhos e taxas de compressão são limitados, arrays são
carregados com o pickling desativado, todo resumo de áudio é verificado e todo
campo passa pela mesma validação de um valor digitado no formulário.

## Integração com o iNaturalist

As duas páginas aceitam o ID ou a URL de uma observação do iNaturalist, em um
campo abaixo do envio de arquivos, e preparam a gravação exatamente como um
arquivo enviado, com a mesma prévia e os mesmos controles.

Na página **Biblioteca**, os metadados também são preenchidos a partir da
observação: a data e a hora no fuso horário da própria observação, as
coordenadas, o táxon, o observador como `recorded_by` e a página da observação
como `link_to_full_recording` (à qual o `sample_window` de um item recortado
passa a se referir). A licença e a atribuição são acrescentadas a `notes`.
Todos os campos permanecem editáveis, e as notas já existentes são
complementadas, e não substituídas.

Na página **Ferramenta de busca**, a gravação se torna a consulta, e as
coordenadas da observação (com um raio de 25 km) e o táxon são inseridos no
painel de filtros, que é aberto para que a restrição fique visível e possa ser
removida. A data não é utilizada, pois a data de uma observação registra
quando aquela gravação foi feita, e não quando o animal vocaliza.

Em ambos os casos, as ressalvas relevantes são informadas. Um local ocultado é
deixado em branco, em vez de preenchido com o ponto aleatorizado do
iNaturalist; uma gravação com todos os direitos reservados é sinalizada; e
observações abaixo do grau de pesquisa são indicadas, pois sua identificação
ainda pode mudar. Uma barra de progresso é exibida durante o download da
gravação.

Quando uma observação tem várias gravações, a usada é escolhida numa lista, e o
crédito em `notes` traz a licença dessa gravação.

Na linha de comando, `--inat` toma o lugar do arquivo de áudio:

```bash
biotimbre add --inat 400139850                    # baixar e preencher os metadados
biotimbre add --inat 400139850 --trim 2 7 --taxa cicadidae   # as opções têm precedência
biotimbre search --inat 400139850 --start-time 2 --end-time 7
biotimbre search --inat 400139850 --inat-filters  # também restringir pelo táxon e local
```

`add --inat` preenche os mesmos campos que a página Biblioteca; qualquer opção
de metadados informada substitui o que a observação diz, e um local ocultado
precisa ser informado com `--latitude` e `--longitude`. `search --inat` usa a
gravação apenas como consulta: ao contrário da página de busca, só aplica o
táxon e o raio de 25 km da observação com `--inat-filters`, já que um filtro
definido na linha de comando passaria despercebido. `--inat-sound N` escolhe
outra das gravações da observação. As mesmas ressalvas são exibidas como
avisos.

A maior parte do áudio do iNaturalist está em AAC, em um contêiner `.m4a`, que
a libsndfile não consegue ler; por isso, a decodificação exige o **ffmpeg**.
Sem o ffmpeg, o download falha com uma mensagem explicativa, e todos os demais
recursos continuam funcionando. Um envio típico de 128 kbps não contém nada
acima de aproximadamente 16 kHz, independentemente da taxa de amostragem
declarada: adequado para o Perch, abaixo dos 24 kHz do BirdNET e inadequado
para trabalhos com ultrassom.

A integração pode ser desativada em uma biblioteca off-line:

```toml
[sources.inaturalist]
enabled = false
```

## Mapa de Localização

Todo item possui coordenadas (as colunas são `NOT NULL`), portanto cada página
exibe **um único mapa**: a página Biblioteca mostra todos os itens, e a página
Ferramenta de busca mostra os resultados atuais. Cada ponto representa um item;
selecioná-lo identifica o item e oferece um link para seus detalhes. Um mapa
separado para cada item é deliberadamente evitado, pois exigiria uma instância
de mapa e um conjunto de requisições de blocos (tiles) por resultado.

Na página Ferramenta de busca, o mapa também pode ser utilizado para definir o
filtro de localização: clicar em outro ponto do mapa define o centro (abrindo
o painel de filtros para que a restrição fique visível), e o centro e o raio
são desenhados como um ponto e um círculo que acompanham os campos do filtro.

O Leaflet é distribuído com o NiceGUI e servido pela própria aplicação,
portanto nenhum script é carregado de uma CDN. **Os blocos do mapa base são a
única exceção** e o único motivo pelo qual o mapa acessa a rede. Para uma
biblioteca mantida off-line, o mapa pode ser desativado; as coordenadas
continuam sendo utilizadas nos filtros e exibidas em todos os resultados:

```toml
[map]
enabled = false
```

Para uma única execução, `biotimbre gui --no-map` tem o mesmo efeito. As opções
de linha de comando têm precedência sobre o arquivo de configuração e
permanecem em vigor quando outra biblioteca é aberta pelo navegador.

Definir `tile_url` para um servidor de blocos próprio mantém as requisições do
mapa fora do servidor público. Qualquer que seja o servidor utilizado,
`attribution` deve creditá-lo, e `max_zoom` deve corresponder ao nível de zoom
mais profundo que ele oferece, para evitar blocos em branco.

## Opções de Exibição

### Espectrograma e Forma de Onda

Toda imagem de uma gravação — na janela de detalhes do item, no formulário de
adição, na consulta de busca e em cada resultado — pode ser exibida como
espectrograma ou como forma de onda. O botão ao lado de cada imagem altera
apenas aquela imagem. A escolha mais recente é memorizada por navegador,
juntamente com o idioma e o tema, e é utilizada nas imagens exibidas a seguir.

A forma de onda mostra dois valores por coluna de pixels: a faixa de **pico**
(a amplitude máxima do sinal, para que cliques e batidas de asas continuem
visíveis), como uma faixa clara, e o nível **RMS** (a intensidade da coluna,
que corresponde à percepção auditiva do volume), desenhado dentro dela. As
amplitudes são escaladas em relação à coluna de maior intensidade, assim como
a faixa em decibéis do espectrograma é escalada em relação ao seu ponto mais
intenso, de modo que uma gravação de baixo volume continua sendo exibida com
clareza, e não como uma linha plana.

A forma de onda é desenhada como SVG embutido, e não como uma imagem
renderizada. Isso permite que ela utilize as cores do tema (uma imagem é
renderizada no servidor, onde a preferência do usuário por tema claro ou
escuro não é conhecida) e se ajuste a qualquer largura sem uma nova
requisição.

### Tema de Cores

O cabeçalho oferece um controle de três estados: **seguir o sistema** (o
padrão), **claro** e **escuro**. A escolha é memorizada por navegador,
juntamente com o idioma, de modo que várias pessoas que utilizam uma mesma
instância local podem escolher cada uma o seu; o arquivo de configuração não é
alterado.

Cada cor é declarada uma única vez, como `light-dark(<claro>, <escuro>)`, em
conjunto com um `color-scheme` de CSS, de modo que as duas paletas não podem
divergir, e os controles nativos que as folhas de estilo não alcançam (barras
de rolagem, o reprodutor de áudio, seletores de data) também acompanham o
tema. A troca entra em vigor sem recarregar a página.

## Configuração

O arquivo de configuração `biotimbre.toml` é opcional, e toda configuração
possui um valor padrão.

**Caminhos no Windows.** Em TOML, uma barra invertida entre aspas duplas
inicia uma sequência de escape, de modo que `"C:\Users\eu\biotimbre"` é
recusado como inválido. Use aspas simples, que tomam o texto literalmente
(`'C:\Users\eu\biotimbre'`), ou barras normais (`"C:/Users/eu/biotimbre"`).

```toml
[library]
data_dir = "data"
# db_path   = "/fast/library.db"      # pode ser armazenado separadamente do áudio
# audio_dir = "/bulk/audio"           # armazenado como caminho absoluto se estiver fora de data_dir
# model_cache_dir = "/path/to/cache"  # padrão: o cache do Hugging Face

[models]
active = ["perch_v2", "birdnet_v2.4"]   # padrão: todos os modelos conhecidos
inference_batch_size = 1   # janelas por chamada do modelo; 1 é o mais rápido aqui
inference_threads = 0      # 0 deixa o onnxruntime decidir
# tabelas [models.custom.<id>] adicionam modelos: veja Adição de um Modelo

[search]
top_k = 10
metric = "cosine"          # cosine | dot | euclidean
fusion = "max"             # max | mean | ID de um modelo
batch_size = 8192
query_cache_entries = 16
max_query_duration_s = 15

[ui]
language = "en"            # en | pt_BR
search_only = false        # serve apenas a página de busca, com a biblioteca somente leitura
dominant_frequency = true  # linha tracejada e rotulada na frequência de maior intensidade

[map]
# O mapa de localização das duas páginas. É a única parte da interface que
# acessa a rede simplesmente por ser exibida: o Leaflet é servido por este
# processo, mas os blocos do mapa base vêm da URL abaixo. Uma biblioteca
# off-line deve definir enabled = false. As coordenadas continuam sendo
# filtradas e exibidas como texto em ambos os casos.
enabled = true
# tile_url    = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# attribution = "&copy; OpenStreetMap contributors"   # deve creditar a fonte
# max_zoom    = 19        # deve corresponder ao zoom mais profundo do servidor de blocos

[ui.theme]
# A paleta clara. Todas as chaves são opcionais; os valores mostrados são os
# padrões. As cores são valores CSS (hexadecimal, rgb(a) ou nomeadas).
# primary..warning definem a paleta do Quasar; as demais chaves estilizam o
# cabeçalho diretamente.
primary = "#20460a"
secondary = "#fcfee5"
accent = "#e9b44c"
dark = "#20460a"
positive = "#118b50"
negative = "#c62828"
info = "#2f6f9f"
warning = "#e9b44c"
button_text = "#fcfee5"        # cor do texto em botões preenchidos com a cor primária
page_bg = "#f4f6ef"            # fundo da página abaixo do cabeçalho
text = "#1a2712"               # cor padrão dos textos
header_bg = "#1b2516"
header_text = "#fcfee5"
header_inactive_text = "rgba(255, 255, 255, 0.65)"
header_active_bg = "rgba(255, 255, 255, 0.2)"

# As chaves a seguir estilizam componentes cujas cores o Quasar e o Tailwind
# fixam independentemente de ui.colors():
card = "#ffffff"               # fundo dos cartões (janela do item, formulários)
link = "#118b50"               # hiperlinks (por exemplo, "abrir gravação completa")
muted = "#5b6b52"              # texto secundário (dicas, horários, IDs)
notice = "#92400e"             # avisos em linha (velocidade alterada, limite da consulta etc.)
surface = "#eef1e6"            # fundo dos blocos de estatísticas na página Biblioteca
chip = "#dde4d1"               # fundo das pequenas etiquetas de ID
marker_border = "rgba(252, 254, 229, 0.8)"  # contorno da janela correspondente no espectrograma
marker_fill = "rgba(252, 254, 229, 0.25)"

# Cores das próprias imagens de espectrograma. Pontos #rrggbb igualmente
# espaçados, do mais silencioso (primeiro) ao mais intenso (último); são
# necessários pelo menos dois. A sequência deve aumentar de brilho de forma
# monotônica, para que a intensidade continue legível para pessoas com
# daltonismo. A sequência padrão vai do azul-escuro ao amarelo-claro,
# passando por magenta e laranja.
spectrogram_colors = ["#0c0826", "#400c60", "#941a60", "#de5234", "#fca83c", "#fff4c8"]

[ui.dark_theme]
# A paleta escura, com as mesmas chaves e significados. Chaves omitidas
# recorrem aos padrões escuros embutidos, e nunca aos valores claros acima, de
# modo que alterar uma cor não traz a paleta clara para o tema escuro.
# primary = "#6f9a7a"
# page_bg = "#0e100f"
# text = "#d6cdb8"
# card = "#161a18"
# spectrogram_colors não é lido aqui: as imagens são renderizadas e mantidas em
# cache no servidor, onde o tema do usuário não é conhecido, portanto a
# sequência de [ui.theme] é utilizada para ambos.
```

## Estrutura dos Dados

```
data/
├── library.db                        # metadados e vetores de embedding
└── audio/<biotimbre_id>/<item_id>.wav
```

Para fazer o backup de uma biblioteca, basta copiar sua pasta. Os vetores são
armazenados no banco de dados para que os filtros de metadados possam reduzir
o conjunto de candidatos antes que qualquer vetor seja pontuado; `export-npy`
fornece os vetores como arquivos quando necessário para outras ferramentas.

## Adição de um Modelo

Além dos dois modelos incluídos (`perch_v2` e `birdnet_v2.4`), qualquer modelo
de embeddings ONNX pode ser adicionado no arquivo de configuração, uma tabela
por modelo. Ele é baixado de um repositório do Hugging Face ou de uma URL
`http(s)`; arquivos locais não são aceitos, para que uma biblioteca sempre
possa obter, em outro computador, os modelos com que foi construída.

```toml
[models.custom.meu_modelo]               # o ID: minúsculas, dígitos, _ . -
repo_id = "dono/repositorio"
filename = "model.onnx"                  # caminho do arquivo no repositório
revision = "3f2a9c1"                     # opcional: fixa um commit
sample_rate = 32000                      # Hz esperados pelo modelo
window_duration_s = 5.0                  # segundos por janela
window_overlap = 0.5                     # opcional, de 0 a 0.95; padrão 0.5
description = "Meu modelo"               # opcional

[models.custom.outro_modelo]
url = "https://example.org/models/encoder.onnx"
sha256 = "9b74c9897bac770ffc029102a200c5de..."  # opcional: fixa o arquivo
sample_rate = 16000
window_duration_s = 1.0
```

O modelo deve receber áudio mono bruto, como um array `(janelas, amostras)` na
sua taxa de amostragem, e devolver um embedding por janela. O biotimbre não calcula espectrogramas;
qualquer etapa desse tipo deve fazer parte do grafo.

A menos que `[models] active` liste os modelos explicitamente, todo modelo
definido assim é usado junto com os incluídos. Para calcular seus embeddings
para os itens já presentes na biblioteca:

```bash
biotimbre backfill meu_modelo
```

Os itens existentes são lidos novamente a partir do áudio armazenado; nenhum
item precisa ser importado novamente.

**A biblioteca se lembra dos seus modelos.** No primeiro uso, a definição do
modelo é gravada em `library.db`, junto com o SHA-256 do arquivo de um modelo
baixado por URL. Uma biblioteca aberta depois sem aquele arquivo de
configuração — por outra pessoa, ou com `--search-only` em outro computador —
ainda baixa e usa o modelo. Uma URL cujo arquivo tenha mudado é recusada, em
vez de misturada aos vetores armazenados.

**Uma definição não pode mudar depois que existem vetores.** Vetores só são
comparáveis quando vêm do mesmo arquivo, com a mesma taxa de amostragem,
janela e sobreposição. Se algum desses itens mudar sob um ID que já tem
vetores, o modelo é recusado e o campo divergente é indicado. Defina o modelo
alterado sob um novo ID e execute o backfill.

## Desenvolvimento

```bash
git clone git@github.com:biodiversica/biotimbre.git
cd biotimbre
uv sync --extra dev --extra gui
uv run biotimbre --help
uv run pytest                    # suíte completa, aproximadamente 2 minutos
uv run pytest -m "not models"    # ignora os testes que baixam e executam modelos reais
```

`uv sync --extra X` substitui o conjunto de extras ativos em vez de
acrescentar a ele, de modo que todos os extras necessários devem ser indicados
a cada execução.

A CI (`.github/workflows/ci.yml`) executa a suíte completa no Linux, no macOS
e no Windows a cada push para `main` e a cada pull request. Ela também gera o
pacote, executa os dois instaladores com ele e inicia a interface a partir da
pasta de trabalho do instalador.

## Licença

[MIT](https://github.com/biodiversica/biotimbre/blob/main/LICENSE)
