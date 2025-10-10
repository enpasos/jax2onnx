# Using the `onnx_ir` Type Information (PEP 561)

The `onnx_ir` package ships inline types and a `py.typed` marker so static type checkers know the annotations are part of the public API. Downstream projects can opt into these types without copying stubs or tweaking search paths—the package already complies with [PEP 561](https://peps.python.org/pep-0561/).

## What the PEP 561 marker gives you

- `onnx_ir` publishes `src/onnx_ir/py.typed`, signalling to type checkers that the runtime package contains type information.
- All exports rebind their `__module__` to `onnx_ir` (see `src/onnx_ir/__init__.py`), so hover information and error messages reference the public surface instead of private modules.
- Because the marker file is empty, the project advertises itself as “fully typed”. If you discover gaps, report them or mark them with `typing.cast` to keep the promise tidy.

## Install the package as usual

```bash
pip install onnx-ir
```

You do not need extra stub packages. If you vendor the source, be sure to keep the `py.typed` file alongside the `onnx_ir` directory.

For pyproject-based projects, declare the dependency in `pyproject.toml`:

```toml
[project]
dependencies = [
  "onnx-ir>=0.1.11",
]
```

Pin to at least the version you tested with so the exported types stay stable.

## Configure your type checker

### mypy

```ini
# mypy.ini
[mypy]
python_version = 3.11
strict = True

[mypy-onnx_ir.*]
ignore_missing_imports = False  # default, kept for clarity
```

Run it with:

```bash
mypy src
```

Because `py.typed` lives next to the package, mypy discovers the types automatically.

### Pyright / Pylance

Add the package to the `venv` you use for analysis. Pyright picks up the marker automatically; no `"typeCheckingMode": "strict"` toggle is required but is recommended:

```json
// pyproject.toml or pyrightconfig.json (whichever you use)
{
  "pythonVersion": "3.11",
  "typeCheckingMode": "strict",
  "reportMissingTypeStubs": "warning"
}
```

### Other tools

- **Pytype**: as long as `onnx_ir` is importable, Pytype reads the inline annotations.
- **Ruff** (`ruff check --select PYI`): uses the same metadata when linting.

## Working with the API in a typed codebase

### Constructing a model

```python
from __future__ import annotations

from onnx_ir import Model, Node, tensor, val


def build_constant_add() -> Model:
    weight = tensor([1.0, 2.0, 3.0], name="weight")
    bias = tensor([0.1, 0.2, 0.3], name="bias")
    add_node: Node = Node(
        op_type="Add",
        inputs=[weight.value, bias.value],
        outputs=[val("sum", elem_type=weight.type.elem_type, shape=weight.type.shape)],
    )
    return Model.from_nodes([add_node])
```

Key points:

- `tensor()`, `node()`, and `val()` return fully typed objects so attribute access is safe.
- Structural protocols such as `TensorProtocol` can type-annotate third-party tensor objects accepted by the API.

### Leveraging protocols for interop

```python
from typing import Iterable

from onnx_ir import TensorProtocol


def unwrap_tensor(tensor: TensorProtocol) -> Iterable[float]:
    return list(tensor.tolist())
```

If you pass a NumPy array that implements `TensorProtocol`, the checker confirms compatibility without extra casts.

## Best practices for downstream libraries

- **Adopt future annotations**: add `from __future__ import annotations` at the top of new modules to reduce runtime typing overhead.
- **Preserve type information**: if you re-export `onnx_ir` objects, rebind them in your own `__all__` so IDEs surface the right module path.
- **Avoid untyped opt-outs**: instead of `type: ignore[assignment]`, prefer `typing.cast` with explanatory comments; this keeps you aligned with the library’s declared typing coverage.
- **Test your types**: run `mypy --strict` or `pyright --level strict` in CI to detect API changes early when you upgrade `onnx_ir`.

## Troubleshooting

- **Checker cannot find the package**: confirm the environment where the checker runs has `onnx_ir` installed—`python -m site` helps locate `site-packages`.
- **Marker missing after vendoring**: ensure `onnx_ir/py.typed` ships with your wheel or editable install. Without it, consumers see “Missing type stubs” warnings even though annotations exist.
- **Encountering `Any` leaks**: file an issue with a minimal example; the maintainers mark problematic sections using `typing.cast` or `Protocol` adjustments rather than dropping the marker.

## Further reading

- [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
- [Typing best practices in the Python docs](https://docs.python.org/3/howto/typing.html)
- [mypy configuration reference](https://mypy.readthedocs.io/en/stable/config_file.html)
