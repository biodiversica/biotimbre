# Configuração

*[English](../en/configuration.md) · Português (Brasil)*

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
rank_by = "perch_v2"       # o modelo cujas pontuações ordenam os resultados
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
