# Preparação do Áudio

*[English](../en/audio-preparation.md) · Português (Brasil)*

## Recorte, Filtragem e Normalização

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

## Gravações Ultrassônicas e o Parâmetro de Velocidade

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
