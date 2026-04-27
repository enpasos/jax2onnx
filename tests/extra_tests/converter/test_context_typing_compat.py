# tests/extra_tests/converter/test_context_typing_compat.py

from __future__ import annotations

from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins import plugin_system


def test_legacy_irbuildcontext_aliases_use_shared_protocol() -> None:
    assert conversion_api._IRBuildContext is LoweringContextProtocol
    assert plugin_system._IRBuildContext is LoweringContextProtocol


def test_ir_context_satisfies_lowering_context_protocol() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])

    assert isinstance(ctx, LoweringContextProtocol)
