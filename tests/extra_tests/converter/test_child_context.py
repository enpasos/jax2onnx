# tests/extra_tests/converter/test_child_context.py

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.function_scope import FunctionRegistry, FunctionScope
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.jax.lax._control_flow_utils import make_subgraph_context


@pytest.mark.parametrize("scope", ["function", "control_flow"])
def test_child_context_shares_conversion_state_and_owns_its_graph(
    scope: Literal["function", "control_flow"],
) -> None:
    parent = IRContext(
        opset=23,
        enable_double_precision=True,
        normalization_mode="force_decomposed",
        stacktrace_metadata=True,
    )
    registry = FunctionRegistry()
    parent.set_function_registry(registry)
    parent.record_primitive_calls_file = "calls.json"
    parent._call_input_param_names = {"deterministic"}
    parent._call_input_param_literals = {"deterministic": True}
    parent._keep_function_float32 = True
    parent_value = ir.Value(
        name="parent_input",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(("B", 3)),
    )
    parent.builder.inputs.append(parent_value)
    parent.builder.outputs.append(parent_value)
    parent.builder.add_initializer_from_array(
        name="weight", array=np.ones((3,), np.float32)
    )
    parent.record_symbolic_dim_origin("B", parent_value, 0)
    parent._call_param_value_by_name["deterministic"] = parent_value

    child = parent.create_child_context(scope=scope)

    assert child.opset == 23
    assert child.enable_double_precision
    assert child.normalization_mode == "force_decomposed"
    assert child.builder.stacktrace_metadata_enabled
    assert child.record_primitive_calls_file == "calls.json"
    assert child._lowering_record_owner is parent
    assert child._primitive_call_records is parent._primitive_call_records
    assert child.get_function_registry() is registry
    assert child._func_name_counters is parent._func_name_counters
    assert child._call_input_param_names == {"deterministic"}
    assert child._call_input_param_names is not parent._call_input_param_names
    assert child._call_input_param_literals == {"deterministic": True}
    assert child._call_input_param_literals is not parent._call_input_param_literals
    assert child._call_param_value_by_name == {}
    assert child.builder.graph is not parent.builder.graph
    assert not child.builder.inputs
    assert not child.builder.outputs
    assert not child.builder.initializers
    assert not tuple(child.builder.graph)
    assert child._function_mode and child.builder._function_mode

    if scope == "control_flow":
        assert child.ir_functions is parent.ir_functions
        assert child._inside_function_scope
        assert child._keep_function_float32
        origin = child.get_symbolic_dim_origin("B")
        assert origin is not None and origin.value is parent_value
        assert child._sym_origin is not parent._sym_origin
        assert child._sym_origin_str is not parent._sym_origin_str
    else:
        assert child.ir_functions is not parent.ir_functions
        assert child._inside_function_scope
        assert not child._keep_function_float32
        assert child.get_symbolic_dim_origin("B") is None

    child._call_input_param_names.add("child_only")
    child._call_input_param_literals["deterministic"] = False
    assert parent._call_input_param_names == {"deterministic"}
    assert parent._call_input_param_literals == {"deterministic": True}


def test_subgraph_names_preserve_context_and_builder_suffixes() -> None:
    parent = IRContext(opset=21, enable_double_precision=False)
    first = make_subgraph_context(parent, prefix="loop")
    second = make_subgraph_context(parent, prefix="loop")

    assert first.fresh_name("value") == "loop_0/value_0"
    assert first.fresh_name("value") == "loop_0/value_1"
    assert first.builder.fresh_name("value") == "loop_0/value_0"
    assert first.fresh_name("trailing_") == "loop_0/trailing_0"
    assert first.builder.fresh_name("trailing_") == "loop_0/trailing__0"
    assert first.fresh_name("path/") == "loop_0/path/0"
    assert first.builder.fresh_name("path/") == "loop_0/path/_0"
    assert second.fresh_name("value") == "loop_1/value_0"
    assert second.builder.fresh_name("value") == "loop_1/value_0"

    nested = make_subgraph_context(first, prefix="branch")
    assert nested.fresh_name("value") == "loop_0/branch_0/value_0"
    assert nested.builder.fresh_name("value") == "loop_0/branch_0/value_0"
    assert parent.fresh_name("value") == "value_0"


def test_function_scope_remaps_only_origins_of_formal_inputs() -> None:
    parent = IRContext(opset=21, enable_double_precision=False)
    source = ir.Value(
        name="source",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(("B", 3)),
    )
    unrelated = ir.Value(name="unrelated", shape=ir.Shape(("T",)))
    parent.builder.inputs.extend([source, unrelated])
    parent.record_symbolic_dim_origin("B", source, 0)
    parent.record_symbolic_dim_origin("T", unrelated, 0)

    scope = FunctionScope(parent, name="Block")
    assert scope.ctx.get_symbolic_dim_origin("B") is None
    formal = scope.begin([source])[0]

    assert formal is not source
    assert formal.name == "f_in_0"
    assert formal.graph is scope.ctx.builder.graph
    assert source.graph is parent.builder.graph
    assert formal.shape == source.shape
    origin = scope.ctx.get_symbolic_dim_origin("B")
    assert origin is not None and origin.value is formal and origin.axis == 0
    assert scope.ctx.get_symbolic_dim_origin("T") is None
    parent_origin = parent.get_symbolic_dim_origin("B")
    assert parent_origin is not None and parent_origin.value is source
    assert scope.ctx._inside_function_scope

    scope.end([formal])
    assert scope.ctx._inside_function_scope


def test_if_flag_capture_is_lazy_local_and_reused_across_nested_branches() -> None:
    parent = IRContext(opset=21, enable_double_precision=False)
    outer = make_subgraph_context(parent, prefix="outer", implicit_captures=True)
    inner = make_subgraph_context(outer, prefix="inner", implicit_captures=True)
    assert not parent.builder.inputs
    assert not tuple(outer.builder.graph)
    assert not tuple(inner.builder.graph)

    captured = inner.ensure_external_flag("deterministic", None)
    assert inner.ensure_external_flag("deterministic", None) is captured
    root_flag = parent.ensure_external_flag("deterministic", None)
    outer_flag = outer.ensure_external_flag("deterministic", None)

    assert len(parent.builder.inputs) == 1
    assert captured.graph is inner.builder.graph
    assert outer_flag.graph is outer.builder.graph
    assert root_flag.graph is parent.builder.graph
    assert captured is not outer_flag and outer_flag is not root_flag
    assert captured.dtype == outer_flag.dtype == root_flag.dtype == ir.DataType.BOOL
    assert captured.shape == ir.Shape(())
    inner_producer = captured.producer()
    outer_producer = outer_flag.producer()
    assert inner_producer is not None and inner_producer.inputs[0] is outer_flag
    assert outer_producer is not None and outer_producer.inputs[0] is root_flag
    assert len(tuple(inner.builder.graph)) == len(tuple(outer.builder.graph)) == 1


def test_function_and_loop_flags_require_explicit_local_inputs() -> None:
    parent = IRContext(opset=21, enable_double_precision=False)
    parent.ensure_external_flag("deterministic", None)
    function = FunctionScope(parent, name="ExplicitInputs")
    function.begin([])
    loop = make_subgraph_context(parent, prefix="loop")

    for child in (function.ctx, loop):
        with pytest.raises(RuntimeError, match="does not have a dynamic value"):
            child.ensure_external_flag("deterministic", None)
        assert not tuple(child.builder.graph)
        assert not child.builder.inputs


def test_if_variable_capture_preserves_identity_metadata_and_local_bindings() -> None:
    parent = IRContext(opset=21, enable_double_precision=False)
    var = object()
    source = ir.Value(
        name="source", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(("B", 2))
    )
    parent.add_graph_input_value(source)
    parent.bind_value_for_var(var, source)
    outer = make_subgraph_context(parent, prefix="outer", implicit_captures=True)
    inner = make_subgraph_context(outer, prefix="inner", implicit_captures=True)

    assert inner.lookup_value_for_var(var) is None
    captured = inner.require_value_for_var(var)
    outer_capture = outer.require_value_for_var(var)
    assert captured is inner.require_value_for_var(var)
    inner_node, outer_node = captured.producer(), outer_capture.producer()
    assert inner_node is not None and inner_node.inputs[0] is outer_capture
    assert outer_node is not None and outer_node.inputs[0] is source
    assert captured.graph is inner.builder.graph
    assert captured.type == source.type and captured.shape == source.shape
    assert not inner.builder.inputs and not outer.builder.inputs
    assert parent.require_value_for_var(var) is source

    # A local binding wins over the enclosing value for the same variable.
    local = inner.builder.op("Neg", [captured])
    inner.bind_value_for_var(var, local)
    assert inner.require_value_for_var(var) is local

    # Unknown variables fail without fabricating captures in either branch.
    before = (len(outer.builder.graph), len(inner.builder.graph))
    with pytest.raises(KeyError, match="No IR value is bound"):
        inner.require_value_for_var(object())
    assert before == (len(outer.builder.graph), len(inner.builder.graph))


@pytest.mark.parametrize("scope", ["function", "control_flow"])
def test_if_variable_capture_stops_at_explicit_scope_boundaries(
    scope: Literal["function", "control_flow"],
) -> None:
    parent = IRContext(opset=21, enable_double_precision=False)
    var = object()
    source = parent.add_graph_input_value(ir.Value(name="outer"))
    parent.bind_value_for_var(var, source)
    boundary = parent.create_child_context(scope=scope)
    branch = make_subgraph_context(boundary, prefix="if", implicit_captures=True)

    with pytest.raises(KeyError, match="No IR value is bound"):
        branch.require_value_for_var(var)
    assert not tuple(branch.builder.graph)
    assert not branch.builder.inputs


@onnx_function(unique=True)
def _scale_in_conditional(x: jax.Array, *, factor: jax.Array) -> jax.Array:
    return x * factor


def _conditional_function_call(
    x: jax.Array, factor: jax.Array, predicate: jax.Array
) -> jax.Array:
    return jax.lax.cond(
        predicate,
        lambda args: _scale_in_conditional(args[0], factor=args[1]),
        lambda args: args[0] - args[1],
        (x, factor),
    )


def test_function_in_cond_keeps_definition_and_graph_local_inputs() -> None:
    model = to_onnx(
        _conditional_function_call,
        [
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        return_mode="ir",
    )
    function = next(
        fn for fn in model.functions.values() if fn.name == "_scale_in_conditional"
    )
    if_node = next(node for node in model.graph if node.op_type == "If")
    branch = if_node.attributes["then_branch"].as_graph()
    call = next(node for node in branch if node.op_type == function.name)

    assert call.domain == function.domain
    assert len(call.inputs) == len(function.inputs) == 2
    assert all(value is not None and value.graph is branch for value in call.inputs)
    assert all(value.graph is function.graph for value in function.inputs)
    for node in function.graph:
        assert all(
            value is None or value.graph is function.graph for value in node.inputs
        )
    assert all(value.shape == ir.Shape((2,)) for value in function.inputs)

    proto = ir.serde.serialize_model(model)
    onnx.checker.check_model(proto, full_check=True)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    factor = np.asarray([4.0, 5.0], dtype=np.float32)
    names = [value.name for value in model.graph.inputs]
    for predicate in (True, False):
        inputs = (x, factor, np.asarray(predicate, dtype=np.bool_))
        (actual,) = session.run(None, dict(zip(names, inputs)))
        expected = np.asarray(_conditional_function_call(*inputs))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@onnx_function(unique=True)
def _flagged_scale(x: jax.Array, *, deterministic: bool = True) -> jax.Array:
    return jnp.where(deterministic, x, x * 2.0)


def _conditional_flagged_call(
    x: jax.Array, predicate: jax.Array, *, deterministic: bool = True
) -> jax.Array:
    return jax.lax.cond(
        predicate,
        lambda value: _flagged_scale(value, deterministic=deterministic),
        lambda value: value - 1.0,
        x,
    )


@onnx_function(unique=True)
def _flagged_conditional_block(
    x: jax.Array, predicate: jax.Array, *, deterministic: bool = True
) -> jax.Array:
    return _conditional_flagged_call(x, predicate, deterministic=deterministic)


def _call_flagged_conditional_block(
    x: jax.Array, predicate: jax.Array, *, deterministic: bool = True
) -> jax.Array:
    return _flagged_conditional_block(x, predicate, deterministic=deterministic)


@pytest.mark.parametrize("inside_function", [False, True])
def test_function_in_cond_captures_external_bool_parameter(
    inside_function: bool,
) -> None:
    target = (
        _call_flagged_conditional_block
        if inside_function
        else _conditional_flagged_call
    )
    model = to_onnx(
        target,
        [
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        input_params={"deterministic": True},
        return_mode="ir",
    )
    flag = next(value for value in model.graph.inputs if value.name == "deterministic")
    assert flag.dtype == ir.DataType.BOOL and flag.shape == ir.Shape(())
    function = next(
        fn for fn in model.functions.values() if fn.name == "_flagged_scale"
    )
    assert len(function.inputs) == 2
    assert function.inputs[1].dtype == ir.DataType.BOOL
    assert function.inputs[1].shape == ir.Shape(())
    assert all(value.graph is function.graph for value in function.inputs)
    graphs = [model.graph] + [fn.graph for fn in model.functions.values()]
    if_node = next(node for graph in graphs for node in graph if node.op_type == "If")
    branch = if_node.attributes["then_branch"].as_graph()
    assert not branch.inputs
    call = next(node for node in branch if node.op_type == function.name)
    assert len(call.inputs) == 2
    assert all(value is not None and value.graph is branch for value in call.inputs)
    for node in function.graph:
        assert all(
            value is None or value.graph is function.graph for value in node.inputs
        )

    proto = ir.serde.serialize_model(model)
    onnx.checker.check_model(proto, full_check=True)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    x_name, predicate_name = [
        value.name for value in model.graph.inputs if value is not flag
    ]
    for predicate in (True, False):
        for deterministic in (True, False):
            (actual,) = session.run(
                None,
                {
                    x_name: x,
                    predicate_name: np.asarray(predicate, dtype=np.bool_),
                    "deterministic": np.asarray(deterministic, dtype=np.bool_),
                },
            )
            expected = np.asarray(
                _conditional_flagged_call(
                    x, np.asarray(predicate), deterministic=deterministic
                )
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
