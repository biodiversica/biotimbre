# Interface de Linha de Comando

*[English](../en/cli.md) · Português (Brasil)*

## Referência de Comandos

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
biotimbre search [AUDIO | --inat OBSERVAÇÃO | --item ITEM_ID] [-k N] [filtros]   # sem nenhum: pelos filtros
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

## Exemplos

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

# Buscar com um item que já está na biblioteca: o que mais soa como o item 12
biotimbre search --item 12

# Buscar só pelos metadados: sem gravação de consulta, mais recentes primeiro
biotimbre search --taxa cicadidae --date-from 2024-01-01 -k 50

# Armazenar só parte de uma gravação, filtrada e normalizada, como no formulário
biotimbre add dawn.wav --sidecar dawn.txt --trim 92 99 --apply-highpass 1500 --normalise

# Alterar metadados armazenados; os campos não informados são mantidos
biotimbre edit 12 --taxa "guyalna bonaerensis" --clear device
```

Outras opções de `search`: `--metric {cosine,dot,euclidean}`,
`--rank-by MODEL_ID` (padrão `perch_v2`), `--min-score`, `--query-speed`, `--taxa-all`,
`--date-to`, `--biotimbre-id`, `--exclude`, `--apply-lowpass`, `--apply-filter-order` e,
numa busca só pelos filtros, `--speed` (itens armazenados nessas velocidades).

`--highpass`/`--lowpass` em `add` e `edit` *registram* uma filtragem já feita na
gravação; `--apply-highpass`/`--apply-lowpass` *filtram*. Todas as opções são
explicadas em detalhe por `biotimbre COMANDO --help` (em inglês).

Os resultados de busca e `show` informam o áudio de cada item pelo caminho
completo do arquivo armazenado. Num terminal que aceita links (GNOME Terminal,
Konsole, Windows Terminal, iTerm2, kitty, WezTerm, VS Code), basta clicar no
caminho para tocar a gravação no reprodutor de áudio do sistema; nos demais, ele
pode ser copiado para um.
