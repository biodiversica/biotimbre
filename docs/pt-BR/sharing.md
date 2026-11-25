# Compartilhamento de Itens entre Bibliotecas

*[English](../en/sharing.md) · Português (Brasil)*

`biotimbre_id` é um UUID para que um ser sonoro tenha a mesma identidade em
todas as bibliotecas que o contêm. Os itens são transferidos como **pacotes**:
um único arquivo `.btz` contendo os metadados de um item, seu áudio armazenado
e seus vetores de embedding.

## Criação e Importação de Pacotes

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

## Regras de Importação

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

## Importando de Outra Biblioteca

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

## Segurança

Um pacote recebido de outra pessoa é tratado como não confiável. Apenas os
membros indicados em seu manifesto são lidos, caminhos que escapariam do
arquivo são recusados, tamanhos e taxas de compressão são limitados, arrays são
carregados com o pickling desativado, todo resumo de áudio é verificado e todo
campo passa pela mesma validação de um valor digitado no formulário.
