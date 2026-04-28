# tests/extra_tests/converter/test_conversion_context_helpers.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import jax
import jax.numpy as jnp

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
        enable_double_precision=False,
        input_specs=[],
        frozen_params={"training": True},
        record_primitive_calls_file="calls.log",
    )

    assert ctx._call_input_param_names == {"training"}
    assert ctx._call_input_param_literals == {"training": True}
    assert ctx.record_primitive_calls_file == "calls.log"
    assert ctx.get_function_registry() is not None


def test_bind_closed_jaxpr_constants_adds_initializers() -> None:
    trace = _trace_to_jaxpr(
        fn=_add_const,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params=None,
        enable_double_precision=False,
        inputs_as_nchw=None,
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        enable_double_precision=False,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )

    _bind_closed_jaxpr_constants(
        ctx,
        trace.jaxpr,
        trace.closed_jaxpr.consts,
        default_float=np.dtype(np.float32),
        enable_double_precision=False,
    )

    assert trace.jaxpr.constvars
    assert len(ctx.builder.initializers) == len(trace.jaxpr.constvars)
    assert all(var in ctx.builder._var2val for var in trace.jaxpr.constvars)


def test_bind_jaxpr_inputs_creates_nchw_bridge_and_symbol_origin() -> None:
    trace = _trace_to_jaxpr(
        fn=_identity,
        inputs=[jax.ShapeDtypeStruct(("B", 2, 3, 4), jnp.float32)],
        input_params=None,
        enable_double_precision=False,
        inputs_as_nchw=[0],
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        enable_double_precision=False,
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


def test_layout_adapter_uses_context_symbol_origin_methods() -> None:
    trace = _trace_to_jaxpr(
        fn=_identity,
        inputs=[jax.ShapeDtypeStruct(("B", 2, 3, 4), jnp.float32)],
        input_params=None,
        enable_double_precision=False,
        inputs_as_nchw=[0],
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        enable_double_precision=False,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )

    _LayoutAdapter(ctx, enable_double_precision=False).bind_inputs(
        trace.jaxpr, inputs_as_nchw=trace.inputs_as_nchw
    )

    origin = ctx.get_symbolic_dim_origin("B")
    assert origin is not None
    assert origin.value is ctx.builder.inputs[0]
    assert origin.axis == 0
