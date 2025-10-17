# tests/conftest.py

from __future__ import annotations

from pathlib import Path

import pytest

from jax2onnx.quickstart import export_quickstart_model
from jax2onnx.quickstart_functions import export_quickstart_functions_model

_DOCS_ONNX_DIR = Path(__file__).resolve().parents[1] / "docs" / "onnx"


@pytest.fixture(scope="session", autouse=True)
def ensure_docs_quickstart_models() -> None:
    """Regenerate quickstart ONNX artifacts if a developer removed them locally."""
    _DOCS_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    my_callable = _DOCS_ONNX_DIR / "my_callable.onnx"
    if not my_callable.exists():
        export_quickstart_model(my_callable)
    functions_model = _DOCS_ONNX_DIR / "model_with_function.onnx"
    if not functions_model.exists():
        export_quickstart_functions_model(functions_model)
