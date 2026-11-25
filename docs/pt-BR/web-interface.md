# Interface Web

*[English](../en/web-interface.md) · Português (Brasil)*

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

## Página Biblioteca

A página Biblioteca é dividida em três seções.

- **Itens da biblioteca** — um mapa com todos os itens e a
  lista de itens. São exibidos também os totais da biblioteca, as velocidades
  presentes e os táxons, cada um com seu número de itens; quando há muitos
  táxons, os mais frequentes são exibidos primeiro e os demais podem ser
  expandidos. Selecionar um item abre uma janela com todos os seus
  metadados, seu espectrograma ou forma de onda, um reprodutor e botões para
  exportá-lo, editá-lo ou excluí-lo, ou para **Buscar semelhantes** (veja a
  página de busca abaixo). A edição emite um aviso antes de salvar
  uma alteração de `speed`, pois isso exige o recálculo dos embeddings.
  **Exportar CSV** baixa os metadados de todos os itens em uma tabela (consulte
  [Exportação de Dados](exporting.md)). Marcar itens na lista (a caixa
  do cabeçalho marca uma página inteira) exibe **Exportar N como pacote**, que
  os baixa como um único `.btz` compartilhável, com ou sem os embeddings.
- **Adicionar um item** — envia uma gravação e seus metadados. O espectrograma
  e a taxa de amostragem são exibidos antes que algo seja armazenado, com o
  limite de frequência de cada modelo indicado. A gravação pode ser recortada,
  filtrada e normalizada antes de ser armazenada (consulte
  [Preparação do Áudio](audio-preparation.md)). Erros de validação são
  exibidos no campo correspondente.
- **Importar itens** — carrega um pacote `.btz` de outra biblioteca, ou
  adiciona todos os itens de outra biblioteca no mesmo computador, com uma
  prévia do resultado que pode ser ajustada antes que qualquer item seja
  adicionado (consulte
  [Compartilhamento de Itens entre Bibliotecas](sharing.md)).

O local da gravação é informado em um único campo, no formato
`latitude, longitude` em graus decimais. Vírgula, ponto e vírgula ou espaços
são aceitos como separadores, de modo que coordenadas copiadas de serviços de
mapas, aplicativos de GPS ou cadernos de campo podem ser coladas diretamente.
**Escolher no mapa** permite definir o local clicando em um mapa.

## Página Ferramenta de busca

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
- **Opções** definem o número máximo de resultados, a métrica de
  similaridade, o modelo que ordena os resultados (Perch v2 por padrão) e uma
  pontuação mínima na escala desse modelo. As pontuações de cada modelo têm
  linha de base e dispersão próprias, então um 0,6 de um não equivale a um 0,6
  de outro, e elas nunca são combinadas. A **Velocidade da busca** da consulta começa vazia, o que busca em
  todas as velocidades da biblioteca; um valor busca apenas os itens
  armazenados nele.

Cada resultado exibe a pontuação do modelo que ordena, as pontuações dos
outros modelos com a posição que cada um deu ao item (posições altas em ambos
indicam que os modelos concordam), um reprodutor próprio e um espectrograma no qual está marcada a janela que gerou a
pontuação. A consulta permanece disponível para reprodução no topo da página,
para comparação auditiva. Cada resultado oferece ainda:

- **Detalhes** — uma janela somente leitura com todos os metadados do item,
  como na página Biblioteca, porém sem as opções de edição e exclusão.
- **Exportar** — o áudio do item, seus embeddings ou um pacote para
  compartilhamento.
- **Buscar semelhantes** — busca na biblioteca com a gravação desse item.
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

**Buscar semelhantes**, num resultado ou na janela de um item, faz de um item
da biblioteca a consulta: sua gravação armazenada é preparada como um arquivo
enviado, pronta para ser recortada ou filtrada, e a área da consulta indica o
item. O item fica fora dos próprios resultados, onde viria primeiro com
pontuação 1, a menos que **Incluir o próprio item** esteja marcado. A
velocidade da consulta é a do próprio item, de modo que um item armazenado
desacelerado é comparado com os outros armazenados nessa velocidade. Na linha
de comando, o equivalente é `biotimbre search --item ITEM_ID`.

## Modo Somente Busca

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

## Seleção da Biblioteca na Inicialização

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

Para trocar de biblioteca depois, o botão **⇄** ao lado do nome da biblioteca,
no cabeçalho, a fecha, após confirmar, e volta a esta página, com a biblioteca
recém-fechada oferecida em primeiro lugar para ser aberta de novo. O que não foi
salvo nas páginas — uma gravação sendo adicionada, uma consulta — é descartado;
a biblioteca em si não muda. As outras abas do navegador no mesmo servidor
também vão para esta página. Uma execução somente busca não tem esse botão:
ela serve uma única biblioteca.

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
