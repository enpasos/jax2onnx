# tests/extra_tests/converter/test_conversion_context_helpers.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import jax
import jax.numpy as jnp
import pytest

from jax2onnx.converter.conversion_api import (
    _LayoutAdapter,
    _bind_closed_jaxpr_constants,
    _bind_jaxpr_inputs,
    _create_ir_context,
    _trace_to_jaxpr,
)

_CONST_ARRAY = np.asarray([1.0, 2.0], dtype=np.float32)


def _add_const(x):
    return x + _CONST_ARRAY


def _identity(x):
    return x


def test_create_ir_context_records_call_params_and_file() -> None:
    ctx = _create_ir_context(
        opset=21,
        input_specs=[],
        frozen_params={"training": True},
        record_primitive_calls_file="calls.log",
    )

    assert ctx._call_input_param_names == {"training"}
    assert ctx._call_input_param_literals == {"training": True}
    assert ctx.record_primitive_calls_file == "calls.log"
    assert ctx.get_function_registry() is not None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bind_closed_jaxpr_constants_preserves_typed_payloads(dtype: type) -> None:
    constant = _CONST_ARRAY.astype(dtype)

    def source(x):
        return x + constant

    with jax.enable_x64(dtype is np.float64):
        trace = _trace_to_jaxpr(
            fn=source,
            inputs=[jax.ShapeDtypeStruct((2,), dtype)],
            input_params=None,
            inputs_as_nchw=None,
            outputs_as_nchw=None,
            input_names=["x"],
            output_names=["y"],
        )
    ctx = _create_ir_context(
        opset=21,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )

    _bind_closed_jaxpr_constants(
        ctx,
        trace.jaxpr,
        trace.closed_jaxpr.consts,
    )

    assert trace.jaxpr.constvars
    assert len(ctx.builder.initializers) == len(trace.jaxpr.constvars)
    assert all(var in ctx.builder._var2val for var in trace.jaxpr.constvars)
    for var, payload in zip(trace.jaxpr.constvars, trace.closed_jaxpr.consts):
        value = ctx.require_value_for_var(var)
        actual = value.const_value.numpy()
        assert value.dtype == ir.DataType.from_numpy(np.dtype(dtype))
        assert actual.dtype == np.dtype(dtype)
        assert actual.tobytes() == np.asarray(payload).tobytes()


def test_bind_jaxpr_inputs_creates_nchw_bridge_and_symbol_origin() -> None:
    trace = _trace_to_jaxpr(
        fn=_identity,
        inputs=[jax.ShapeDtypeStruct(("B", 2, 3, 4), jnp.float32)],
        input_params=None,
        inputs_as_nchw=[0],
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )

    _bind_jaxpr_inputs(ctx, trace.jaxpr, inputs_as_nchw=trace.inputs_as_nchw)

    graph_input = ctx.builder.inputs[0]
    assert graph_input.name == "in_0_nchw"
    assert graph_input.shape == ir.Shape(("B", 4, 2, 3))

    bound = ctx.get_value_for_var(trace.jaxpr.invars[0])
    producer = bound.producer()
    assert producer is not None
    assert producer.op_type == "Transpose"
    assert bound.shape == ir.Shape(("B", 2, 3, 4))

    origin = ctx.get_symbolic_dim_origin("B")
    assert origin is not None
    assert origin.value is graph_input
    assert origin.axis == 0


def test_require_value_for_var_does_not_allocate_unbound_outputs() -> None:
    trace = _trace_to_jaxpr(
        fn=_add_const,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params=None,
        inputs_as_nchw=None,
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )
    _bind_closed_jaxpr_constants(
        ctx,
        trace.jaxpr,
        trace.closed_jaxpr.consts,
    )
    _bind_jaxpr_inputs(ctx, trace.jaxpr, inputs_as_nchw=())
    outvar = trace.jaxpr.outvars[0]

    input_value = ctx.lookup_value_for_var(trace.jaxpr.invars[0])
    assert input_value is ctx.builder.inputs[0]
    bindings_before = dict(ctx.builder._var2val)
    nodes_before = tuple(ctx.builder.graph)
    constants_before = tuple(ctx.builder.initializers)
    assert ctx.lookup_value_for_var(outvar) is None
    assert ctx.builder._var2val == bindings_before
    assert tuple(ctx.builder.graph) == nodes_before
    assert tuple(ctx.builder.initializers) == constants_before

    with pytest.raises(KeyError):
        ctx.require_value_for_var(outvar)

    allocated = ctx.allocate_value_for_var(outvar, name_hint="allocated_output")

    assert allocated.name == "allocated_output"
    assert ctx.require_value_for_var(outvar) is allocated
    assert ctx.lookup_value_for_var(outvar) is allocated
    assert ctx.allocate_value_for_var(outvar, name_hint="unused_name") is allocated


@pytest.mark.parametrize("function_mode", [False, True])
def test_literal_lookup_is_pure_and_materialization_preserves_scope(
    function_mode: bool,
) -> None:
    closed = jax.make_jaxpr(lambda x: x + 1.0)(jax.ShapeDtypeStruct((2,), jnp.float32))
    literal = closed.jaxpr.eqns[-1].invars[1]
    ctx = _create_ir_context(
        opset=21,
        input_specs=[],
        frozen_params={},
        record_primitive_calls_file=None,
    )
    ctx._function_mode = function_mode
    ctx.builder._function_mode = function_mode

    assert ctx.lookup_value_for_var(literal) is None
    assert not ctx.builder.initializers
    assert not tuple(ctx.builder.graph)

    value = ctx.require_value_for_var(literal)
    assert value.dtype == ir.DataType.FLOAT
    if function_mode:
        assert not ctx.builder.initializers
        assert value.producer() is not None
        assert value.producer().op_type == "Constant"
    else:
        assert value in ctx.builder.initializers
        assert value.producer() is None


@pytest.mark.parametrize("enable_x64", [False, True])
def test_unbound_float16_outputs_keep_their_traced_dtype(
    enable_x64: bool,
) -> None:
    with jax.enable_x64(enable_x64):
        closed = jax.make_jaxpr(jnp.negative)(jax.ShapeDtypeStruct((2, 3), jnp.float16))
        ctx = _create_ir_context(
            opset=23,
            input_specs=[],
            frozen_params={},
            record_primitive_calls_file=None,
        )
        output = ctx.allocate_value_for_var(closed.jaxpr.outvars[0])

    assert output.dtype == ir.DataType.FLOAT16
    assert output.shape == ir.Shape((2, 3))


def test_layout_adapter_uses_context_symbol_origin_methods() -> None:
    trace = _trace_to_jaxpr(
        fn=_identity,
        inputs=[jax.ShapeDtypeStruct(("B", 2, 3, 4), jnp.float32)],
        input_params=None,
        inputs_as_nchw=[0],
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )

    _LayoutAdapter(ctx).bind_inputs(trace.jaxpr, inputs_as_nchw=trace.inputs_as_nchw)

    origin = ctx.get_symbolic_dim_origin("B")
    assert origin is not None
    assert origin.value is ctx.builder.inputs[0]
    assert origin.axis == 0
