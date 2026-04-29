# tests/extra_tests/converter/test_ir_context_nodes.py

from __future__ import annotations

import onnx_ir as ir

from jax2onnx.converter.ir_builder import (
    JAX_CALLSITE_METADATA_KEY,
    PLUGIN_METADATA_KEY,
)
from jax2onnx.converter.ir_context import IRContext


def test_ir_context_add_node_routes_through_builder_metadata() -> None:
    ctx = IRContext(
        opset=21,
        enable_double_precision=False,
        input_specs=[],
        stacktrace_metadata=True,
    )
    ctx.builder.set_current_jax_traceback("user_model.py:12 (forward)")
    ctx.builder.set_current_plugin_identifier("test.module.Plugin.lower", "34")
    input_value = ir.Value(
        name="input",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((1,)),
    )
    output_value = ir.Value(
        name="output",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((1,)),
    )
    node = ir.Node(
        "",
        "Identity",
        [input_value],
        outputs=[output_value],
        name="identity",
    )

    ctx.add_node(node)

    assert ctx.builder.nodes[-1] is node
    assert node.metadata_props[JAX_CALLSITE_METADATA_KEY] == "forward:12"
    assert node.metadata_props[PLUGIN_METADATA_KEY] == "Plugin.lower:34"
