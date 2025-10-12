# tests/extra_tests/converter/test_jaxpr_converter_interaction_with_builder.py

from __future__ import annotations

import onnx_ir as ir

from jax2onnx.converter.function_scope import FunctionScope
from jax2onnx.converter.ir_context import IRContext


def test_function_scope_constants_emit_constant_nodes() -> None:
    parent = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    parent_input = ir.Value(
        name="x0",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((1,)),
    )

    scope = FunctionScope(parent, name="Fn")
    fn_inputs = scope.begin([parent_input])

    assert fn_inputs and fn_inputs[0] is not parent_input
    assert fn_inputs[0].name.startswith("f_in_")

    child_ctx = scope.ctx
    constant_value = child_ctx.builder.add_initializer_from_scalar(
        name="const", value=1.0
    )

    assert not child_ctx.builder.initializers
    constant_nodes = [n for n in child_ctx.builder.nodes if n.op_type == "Constant"]
    assert constant_nodes, "Expected Constant node in function scope"
    assert constant_nodes[0].outputs[0] is constant_value

    fn_def = scope.end([constant_value])

    assert fn_def.outputs == [constant_value]
    assert any(node.op_type == "Constant" for node in fn_def.nodes)
