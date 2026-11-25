# Exportação de Dados

*[English](../en/exporting.md) · Português (Brasil)*

## Áudio e Embeddings

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

## Embeddings com o Áudio de Entrada do Modelo

A opção **embeddings + trechos de áudio** do menu Exportar, ou
`biotimbre export-npy --with-audio`, inclui também o áudio de cada janela tal
como o modelo o recebeu: com a velocidade ajustada, reamostrado para a taxa de
amostragem do modelo e completado até o comprimento da janela. O resultado é
sempre um arquivo zip (`item_12_birdnet_v2.4_with_audio.zip`) no qual cada
`.npy` é acompanhado de um arquivo WAV com o mesmo nome
(`item_12_birdnet_v2.4_w00_0.00-3.00s.wav`). Se as janelas armazenadas de um
item não corresponderem mais ao seu áudio, a exportação é recusada e o item
precisa ter seus embeddings recalculados antes.

## Tabelas de Metadados (CSV)

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
  posição, pela pontuação do modelo que ordena, por esse modelo, pela janela
  correspondente, pela distância até o centro do filtro, pela velocidade da
  consulta e pela pontuação e posição de cada modelo. Em uma busca sem gravação, as
  colunas de pontuação ficam vazias.
