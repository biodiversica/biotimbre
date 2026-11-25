# Integração com o iNaturalist

*[English](../en/inaturalist.md) · Português (Brasil)*

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
