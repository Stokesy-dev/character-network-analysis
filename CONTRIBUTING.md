# Contributing to Character Network Analysis

Thank you for your interest in contributing!

## How to contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes following the code style below
4. Run tests: `pytest tests/ -v`
5. Run linting: `flake8 src/ tests/ --max-line-length=100`
6. Commit: `git commit -m "feat: description"`
7. Push and open a Pull Request

## Code style

- PEP8 compliant (enforced by flake8)
- Black formatted (`black src/ tests/`)
- Type hints on all functions
- Docstrings on all public functions (Google style)
- Max line length: 100 characters

## Commit convention

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructure
- `test:` tests only
- `chore:` build/config changes

## Suggestions welcome

- New datasets (Marvel, GOT, Harry Potter)
- Additional Graph ML models (GAT, GIN)
- Link prediction task
- Temporal graph analysis
