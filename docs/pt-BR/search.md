# Funcionamento da Busca

*[English](../en/search.md) · Português (Brasil)*

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

## Ordenação

Todos os modelos ativos pontuam todos os candidatos, mas os resultados são
ordenados por um único modelo: o Perch v2, a menos que `search --rank-by`,
*Ordenar por* na página Ferramenta de busca ou `search.rank_by` na
[configuração](configuration.md) indique outro. As pontuações de cada modelo
têm linha de base e dispersão próprias, então um 0,6 de um não equivale a um
0,6 de outro, e elas nunca são combinadas. Em vez disso, cada resultado mostra
também as pontuações dos outros modelos e a posição que cada um deu ao item:
um item perto do topo das duas ordenações é um em que os modelos concordam.
Uma pontuação mínima é lida na escala do modelo que ordena.
