# Contributing

We warmly welcome contributions!

## How You Can Help

- **Add a plugin:** Extend `jax2onnx` by writing a simple Python file in [`jax2onnx/plugins`](https://github.com/enpasos/jax2onnx/tree/main/jax2onnx/plugins):
  a primitive or an example. The [Plugin Quickstart](../developer_guide/plugin_system.md) walks through the process step-by-step.

- **Bug fixes & improvements:** PRs and issues are always welcome on [GitHub](https://github.com/enpasos/jax2onnx).

## Getting Started

1. **Fork** the repository and clone it locally.
2. **Install** dependencies with [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   ```
3. **Install pre-commit hooks** (essential for linting):
   ```bash
   poetry run pre-commit install
   ```

## Development Workflow

### Linting & Formatting

We use `ruff` (linting), `black` (formatting), and `mypy` (typing).

```bash
# Run all checks (recommended)
poetry run pre-commit run --all-files

# Or run individually
poetry run ruff check .
poetry run black .
poetry run mypy .
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run a specific test file
poetry run pytest tests/test_my_feature.py
```

### Documentation

To preview documentation changes locally:

```bash
poetry run mkdocs serve
```

See the [Plugin System](../developer_guide/plugin_system.md) documentation for details on adding new operators.
