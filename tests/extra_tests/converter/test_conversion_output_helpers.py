# tests/extra_tests/converter/test_conversion_output_helpers.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import onnx_ir as ir
import pytest

from jax2onnx.converter.conversion_api import (
    _bind_jaxpr_inputs,
    _bind_jaxpr_outputs,
    _create_ir_context,
    _trace_to_jaxpr,
)


def _identity(x):
    return x


def _prepare_identity_context(shape: tuple[object, ...]):
    trace = _trace_to_jaxpr(
        fn=_identity,
        inputs=[jax.ShapeDtypeStruct(shape, jnp.float32)],
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
    _bind_jaxpr_inputs(ctx, trace.jaxpr, inputs_as_nchw=())
    return trace, ctx


def test_bind_jaxpr_outputs_adds_standard_outputs() -> None:
    trace, ctx = _prepare_identity_context((2,))

    _bind_jaxpr_outputs(
        ctx,
        trace.jaxpr,
        outputs_as_nchw=(),
        enable_double_precision=False,
    )

    assert ctx.builder.outputs == [ctx.builder.inputs[0]]


def test_bind_jaxpr_outputs_adds_nchw_transpose() -> None:
    trace, ctx = _prepare_identity_context(("B", 2, 3, 4))

    _bind_jaxpr_outputs(
        ctx,
        trace.jaxpr,
        outputs_as_nchw=(0,),
        enable_double_precision=False,
    )

    assert len(ctx.builder.outputs) == 1
    output = ctx.builder.outputs[0]
    producer = output.producer()
    assert producer is not None
    assert producer.op_type == "Transpose"
    assert output.name == "out_0_nchw_converted"
    assert output.shape == ir.Shape(("B", 4, 2, 3))


def test_bind_jaxpr_outputs_rejects_nchw_for_non_4d_output() -> None:
    trace, ctx = _prepare_identity_context((2,))

    with pytest.raises(
        ValueError,
        match="outputs_as_nchw: output 0 has rank 1, expected 4",
    ):
        _bind_jaxpr_outputs(
            ctx,
            trace.jaxpr,
            outputs_as_nchw=(0,),
            enable_double_precision=False,
        )
