# Desenvolvimento

*[English](../en/development.md) · Português (Brasil)*

```bash
git clone git@github.com:biodiversica/biotimbre.git
cd biotimbre
uv sync --extra dev --extra gui
uv run biotimbre --help
uv run pytest                    # suíte completa, aproximadamente 2 minutos
uv run pytest -m "not models"    # ignora os testes que baixam e executam modelos reais
```

`uv sync --extra X` substitui o conjunto de extras ativos em vez de
acrescentar a ele, de modo que todos os extras necessários devem ser indicados
a cada execução.

A CI (`.github/workflows/ci.yml`) executa a suíte completa no Linux, no macOS
e no Windows a cada push para `main` e a cada pull request. Ela também gera o
pacote, executa os dois instaladores com ele e inicia a interface a partir da
pasta de trabalho do instalador.
