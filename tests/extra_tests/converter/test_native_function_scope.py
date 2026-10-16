# tests/extra_tests/converter/test_native_function_scope.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.conversion_api import _apply_late_ir_attr_overrides
from jax2onnx.converter.function_scope import (
    FunctionKey,
    FunctionRegistry,
    FunctionScope,
)
from jax2onnx.converter.ir_context import IRContext


def test_function_finalization_owns_its_body_and_preserves_metadata():
    parent = IRContext(opset=21, enable_double_precision=False)
    parent_input = ir.Value(
        name="x", type=ir.TensorType(ir.DataType.FLOAT16), shape=ir.Shape(("B", 3))
    )
    parent.builder.inputs.append(parent_input)
    scope = FunctionScope(parent, name="Activation", domain="custom.Activation.1")
    formal = scope.begin([parent_input])[0]
    formal.metadata_props["formal"] = "preserved"
    staging_graph = scope.ctx.builder.graph
    staging_graph.doc_string = "Function body documentation"
    staging_graph.metadata_props["body"] = "preserved"
    output = ir.Value(name="activated", type=formal.type, shape=formal.shape)
    output.metadata_props["result"] = "preserved"
    scope.ctx.builder.op(
        "LeakyRelu",
        [formal],
        {"alpha": 0.2},
        name="activation",
        doc_string="Node documentation",
        metadata_props={"node": "preserved"},
        output=output,
    )
    scope.ctx.add_node_attr_override("activation", {"alpha": 0.4})

    function = scope.end([output])

    assert function.graph is not staging_graph
    assert function.inputs[0] is not formal
    assert function.inputs[0].graph is function.graph
    assert function.outputs[0] is not output
    assert function.outputs[0].graph is function.graph
    assert formal.graph is staging_graph and output.graph is staging_graph
    assert parent_input.graph is parent.builder.graph
    assert not tuple(parent.builder.graph)
    assert function.inputs[0].dtype == function.outputs[0].dtype == ir.DataType.FLOAT16
    assert function.inputs[0].shape == function.outputs[0].shape == ir.Shape(("B", 3))
    assert function.inputs[0].metadata_props["formal"] == "preserved"
    assert function.outputs[0].metadata_props["result"] == "preserved"
    assert function.graph.doc_string == "Function body documentation"
    assert function.graph.metadata_props["body"] == "preserved"
    (node,) = tuple(function.graph)
    assert node.graph is function.graph and node.inputs[0] is function.inputs[0]
    assert node.outputs[0] is function.outputs[0]
    assert node.doc_string == "Node documentation"
    assert node.metadata_props["node"] == "preserved"
    assert function.opset_imports == {"": 21, "custom.Activation.1": 21}

    # Late overrides apply to the finalized function, without mutating staging IR.
    model = ir.Model(parent.builder.graph, ir_version=10)
    model.functions[function.identifier()] = function
    _apply_late_ir_attr_overrides(model, parent)
    assert node.attributes["alpha"].as_float() == pytest.approx(0.4)
    assert tuple(staging_graph)[0].attributes["alpha"].as_float() == pytest.approx(0.2)
    staging_graph.metadata_props["body"] = "changed after finalization"
    assert function.graph.metadata_props["body"] == "preserved"


def test_function_scope_finalizes_once_and_registry_retains_the_native_definition():
    parent = IRContext(opset=21, enable_double_precision=False)
    scope = FunctionScope(parent, name="Identity", domain="custom.Identity.1")
    with pytest.raises(RuntimeError, match="begin.*before end"):
        scope.end([])
    formal = scope.begin([ir.Value(name="x")])[0]
    function = scope.end([formal])
    with pytest.raises(RuntimeError, match="already sealed"):
        scope.end([formal])
    with pytest.raises(RuntimeError, match="only be called once"):
        scope.begin([])

    key = FunctionKey("Identity", (), ())
    registry = FunctionRegistry()
    registry.put(key, function)
    assert registry.get(key) is function
    assert registry.all() == [function]


@onnx_function(unique=True)
def _shared_native_definition(x):
    return jnp.square(x) + 1.0


def _call_shared_native_definition(x):
    return _shared_native_definition(x), _shared_native_definition(x + 1.0)


def test_repeated_calls_publish_the_same_native_definition_stored_in_the_registry(
    monkeypatch,
):
    definitions = []
    original_put = FunctionRegistry.put

    def record_definition(registry, key, function):
        definitions.append(function)
        original_put(registry, key, function)

    monkeypatch.setattr(FunctionRegistry, "put", record_definition)
    model = to_onnx(
        _call_shared_native_definition,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        return_mode="ir",
    )

    assert len(definitions) == 1
    (function,) = definitions
    assert isinstance(function, ir.Function)
    assert model.functions[function.identifier()] is function
    calls = [node for node in model.graph if node.domain == function.domain]
    assert len(calls) == 2
    assert all(node.op_type == function.name for node in calls)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
