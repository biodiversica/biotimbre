# Development

*English · [Português (Brasil)](../pt-BR/development.md)*

```bash
git clone git@github.com:biodiversica/biotimbre.git
cd biotimbre
uv sync --extra dev --extra gui
uv run biotimbre --help
uv run pytest                    # full suite, approximately 2 minutes
uv run pytest -m "not models"    # skips the tests that download and run real models
```

`uv sync --extra X` replaces the active set of extras rather than adding to
it, so every required extra must be named on each run.

CI (`.github/workflows/ci.yml`) runs the full suite on Linux, macOS and
Windows for every push to `main` and every pull request. It also builds the
package and runs both installers with it, then starts the interface from the
installer's working folder.
