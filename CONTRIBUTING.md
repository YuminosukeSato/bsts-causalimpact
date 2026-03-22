# Contributing to bsts-causalimpact

Thank you for considering contributing to bsts-causalimpact.

## Development Setup

### Prerequisites

- Python 3.10+
- Rust toolchain (stable)
- uv (recommended) or pip

### Getting Started

```bash
git clone https://github.com/YuminosukeSato/bsts-causalimpact.git
cd bsts-causalimpact

# Install all dependencies including Rust extension
uv sync --all-extras

# Set up git hooks
git config core.hooksPath .githooks
```

### Running Tests

```bash
# Full test suite
uv run pytest tests/ -v

# Rust unit tests
cargo test

# Lint check
uv run ruff check .
```

## Pull Request Workflow

1. Create a feature branch from `main`
2. Write tests first (TDD)
3. Implement changes
4. Ensure all tests pass and ruff reports no errors
5. Open a PR with a clear description of changes

### Commit Messages

Use conventional commit style:

- `feat:` new features
- `fix:` bug fixes
- `test:` test additions or changes
- `docs:` documentation changes
- `refactor:` code restructuring without behavior change

### Test Requirements

- All new code must have tests
- Boundary values and edge cases must be covered
- Existing tests must not be weakened to make them pass
- Numerical equivalence tests (±3% tolerance with R) must stay green

## Architecture Overview

```
python/causal_impact/   # Python package
src/                    # Rust Gibbs sampler (PyO3)
tests/                  # pytest test suite
benchmarks/             # Performance benchmarks
```

The Gibbs sampler runs in Rust via PyO3 bindings. Python handles data preparation, post-processing, plotting, and summary formatting.

## Reporting Issues

Open an issue on GitHub with:

- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
