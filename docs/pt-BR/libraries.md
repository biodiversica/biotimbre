# Bibliotecas

*[English](../en/libraries.md) · Português (Brasil)*

## Localização da Biblioteca

Por padrão, a biblioteca é armazenada em `./data`, relativo ao diretório atual.
A localização pode ser definida de três maneiras:

```bash
# 1. Para uma única execução
biotimbre --data-dir /media/drive/biotimbre list

# 2. Com um arquivo de configuração no diretório de trabalho (ou --config /caminho/para/ele)
echo '[library]
data_dir = "/media/drive/biotimbre"' > biotimbre.toml

# 3. Com uma variável de ambiente apontando para um arquivo de configuração
export BIOTIMBRE_CONFIG=/media/drive/biotimbre/biotimbre.toml
```

`--data-dir` tem precedência sobre o arquivo de configuração. Caminhos
relativos em um arquivo de configuração são resolvidos em relação ao diretório
que contém esse arquivo, e não ao diretório de trabalho, de modo que um arquivo
de configuração e sua biblioteca podem ser movidos juntos.

**Abrir uma biblioteca existente.** Nenhuma etapa adicional é necessária: todo
comando abre o banco de dados e cria o que estiver faltando.

```bash
biotimbre --data-dir /media/drive/biotimbre list
```

`init` pode ser executado com segurança em uma biblioteca já populada; ele
apenas cria o que estiver ausente e baixa os modelos. Os caminhos dos áudios
armazenados são relativos ao diretório de dados, portanto **a pasta inteira
pode ser movida ou renomeada sem invalidar nenhuma referência.**

## Estrutura dos Dados

```
data/
├── library.db                        # metadados e vetores de embedding
└── audio/<biotimbre_id>/<item_id>.wav
```

Para fazer o backup de uma biblioteca, basta copiar sua pasta. Os vetores são
armazenados no banco de dados para que os filtros de metadados possam reduzir
o conjunto de candidatos antes que qualquer vetor seja pontuado; `export-npy`
fornece os vetores como arquivos quando necessário para outras ferramentas.
