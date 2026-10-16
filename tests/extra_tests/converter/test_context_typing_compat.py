# tests/extra_tests/converter/test_context_typing_compat.py

from __future__ import annotations

import onnx_ir as ir

from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.typing_support import (
    ConverterContext,
    LoweringContextProtocol,
    NativeLoweringContext,
)
from jax2onnx.plugins import plugin_system


def test_legacy_irbuildcontext_aliases_use_shared_protocol() -> None:
    assert LoweringContextProtocol is ConverterContext
    assert conversion_api._IRBuildContext is ConverterContext
    assert plugin_system._IRBuildContext is ConverterContext


def test_ir_context_satisfies_lowering_context_protocol() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])

    assert isinstance(ctx, NativeLoweringContext)
    assert isinstance(ctx, ConverterContext)
    assert isinstance(ctx, LoweringContextProtocol)


def test_native_and_legacy_contexts_emit_into_the_same_graph() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False)
    native: NativeLoweringContext = ctx
    legacy: LoweringContextProtocol = ctx
    value = ir.val("input", dtype=ir.DataType.FLOAT, shape=[2])
    ctx.builder.inputs.append(value)

    first = native.builder.op("Neg", [value])
    second = legacy.builder.Neg(first, _outputs=["legacy_result"])

    assert isinstance(first, ir.Value)
    assert isinstance(second, ir.Value)
    assert [node.op_type for node in ctx.builder.graph] == ["Neg", "Neg"]
    assert list(ctx.builder.graph[0].inputs) == [value]
    assert list(ctx.builder.graph[1].inputs) == [first]
    assert second.producer() is ctx.builder.graph[1]
    assert second.name == "legacy_result"
