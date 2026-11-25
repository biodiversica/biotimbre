# Adição de um Modelo

*[English](../en/adding-a-model.md) · Português (Brasil)*

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
