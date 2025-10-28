# tests/conftest.py

from __future__ import annotations

from pathlib import Path

from tests._initializer_guard import install_initializer_guard

import pytest

try:
    from jax2onnx.quickstart import export_quickstart_model
    from jax2onnx.quickstart_functions import export_quickstart_functions_model
except ImportError:  # pragma: no cover - optional dependency
    export_quickstart_model = None
    export_quickstart_functions_model = None

_DOCS_ONNX_DIR = Path(__file__).resolve().parents[1] / "docs" / "onnx"


@pytest.fixture(scope="session", autouse=True)
def ensure_docs_quickstart_models() -> None:
    """Regenerate quickstart ONNX artifacts if a developer removed them locally."""
    if export_quickstart_model is None or export_quickstart_functions_model is None:
        return
    _DOCS_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    my_callable = _DOCS_ONNX_DIR / "my_callable.onnx"
    if not my_callable.exists():
        export_quickstart_model(my_callable)
    functions_model = _DOCS_ONNX_DIR / "model_with_function.onnx"
    if not functions_model.exists():
        export_quickstart_functions_model(functions_model)


@pytest.fixture(scope="session", autouse=True)
def _install_initializer_guard_fixture():
    """Install the initializer guard for every conversion."""
    with install_initializer_guard():
        yield
