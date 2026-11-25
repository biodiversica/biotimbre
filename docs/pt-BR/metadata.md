# Metadados

*[English](../en/metadata.md) · Português (Brasil)*

## Campos

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
| `speed` | Velocidade de reprodução utilizada no cálculo dos embeddings (consulte [Gravações Ultrassônicas](audio-preparation.md#gravações-ultrassônicas-e-o-parâmetro-de-velocidade)). |
| `link_to_full_recording` | URL da gravação completa. |
| `notes` | Texto livre. |

## Identificadores

`item_id` é atribuído automaticamente, nunca é reutilizado e é específico de
uma biblioteca; trata-se apenas de um identificador de linha.

Um **biotimbre** é um ser sonoro: um sonotipo cuja espécie pode ser
desconhecida. Por isso, `biotimbre_id` é um UUID, e não um contador local da
biblioteca, de modo que o mesmo sonotipo seja identificado de forma consistente
quando bibliotecas forem compartilhadas ou mescladas. Vários itens podem
compartilhar um mesmo `biotimbre_id` — a mesma espécie ou indivíduo, com
conteúdos sonoros diferentes.

## Uma Espécie, Um Biotimbre

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

## Arquivos Sidecar

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

## Tabelas

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
