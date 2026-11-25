# biotimbre documentation

- **English** — [`docs/en/`](en/), indexed from the [README](../README.md)
- **Português (Brasil)** — [`docs/pt-BR/`](pt-BR/), indexado pelo
  [README_pt-BR](../README_pt-BR.md)

Each page exists in both languages under the same file name, and each links to
its translation at the top. A change to one should be made to the other in the
same commit.

The READMEs keep only what biotimbre is, how to install it and a first run.
CI runs the Windows install and uninstall commands exactly as `README.md`
writes them (`.github/scripts/readme-command.ps1`), so those two lines must
stay there. `README.md` is also the package description on PyPI, where
relative links do not resolve: its links are absolute GitHub URLs.
