# jax2onnx/plugins/_utils.py

from __future__ import annotations

from typing import Sequence, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    NativeLoweringContext,
)
from jax2onnx.ir_utils import const_value_to_numpy, ir_dtype_to_numpy
from jax2onnx.plugins._loop_extent_meta import propagate_axis0_override


def cast_value_to_dtype(
    ctx: NativeLoweringContext,
    value: ir.Value,
    dtype: ir.DataType,
    *,
    name_hint: str,
) -> ir.Value:
    """Cast this consumer's operand without changing its source binding."""
    if value.dtype == dtype:
        return value
    output = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(dtype),
        shape=value.shape,
    )
    result: ir.Value = ctx.builder.op(
        "Cast", [value], {"to": int(dtype.value)}, output=output
    )
    propagate_axis0_override(value, result)
    return result


def cast_param_like(
    ctx: LoweringContextProtocol,
    param: ir.Value,
    like: ir.Value,
    name_hint: str = "param_cast",
) -> ir.Value:
    """
    If `param`'s dtype differs from `like`'s dtype, insert a CastLike node to cast
    `param` to `like`'s dtype. Returns a consumer-local value without changing
    either original binding or a shared constant's payload.

    Safe to call when dtypes are unknown (no-op). Preserves shape.
    """
    p_dt = param.dtype
    l_dt = like.dtype
    if p_dt is None or l_dt is None or p_dt == l_dt:
        return param

    cast_name = ctx.fresh_name(name_hint)
    out = cast(
        ir.Value,
        ctx.builder.CastLike(
            param,
            like,
            _outputs=[cast_name],
        ),
    )
    out.type = ir.TensorType(l_dt)
    out.shape = param.shape
    return out


def inline_reshape_initializer(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    new_shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    """
    Return a graph-local constant view, reusing the same source/shape pair.

    Returning ``val`` signals that the caller must emit a runtime Reshape. Even
    an unchanged constant shape therefore gets a distinct derived value on its
    first use. Equal payloads from different source values remain independent.
    """
    if val.const_value is None or val.is_graph_input() or val.graph is None:
        return val  # not a constant → caller must insert a Reshape node

    producer = val.producer()
    if not val.is_initializer() and (
        producer is None
        or producer.op_type != "Constant"
        or producer.domain != ""
        or producer.overload
        or producer.inputs
        or any(attribute.is_ref() for attribute in producer.attributes.values())
    ):
        return val

    graph = ctx.builder.graph
    cache = ctx.builder.reshaped_initializers
    key = (val, new_shape)
    cached = cache.get(key)
    if cached is not None and cached.graph is graph and not cached.is_graph_input():
        return cached

    arr = const_value_to_numpy(val)
    if arr is None:
        return val

    np_arr = np.asarray(arr).reshape(new_shape)
    # Preserve dtype from the original value when available
    target_dtype = val.dtype
    target_np_dtype = ir_dtype_to_numpy(target_dtype, default=None)
    if target_np_dtype is not None:
        np_arr = np_arr.astype(target_np_dtype, copy=False)

    initializer: ir.Value = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint), array=np_arr
    )
    tensor = initializer.const_value
    if tensor is not None:
        tensor.metadata_props.update(val.const_value.metadata_props)
        tensor.doc_string = val.const_value.doc_string
    initializer.metadata_props.update(val.metadata_props)
    initializer.doc_string = val.doc_string
    cache[key] = initializer
    return initializer


def normalize_builder_outputs(
    outputs: ir.Value | Sequence[ir.Value],
) -> tuple[ir.Value, ...]:
    """
    Normalize ONNX IR builder outputs to a tuple of values.

    The underlying builder collapses op results to a bare ``ir.Value`` whenever
    the requested output arity is 1, even if the caller passed a one-element
    ``_outputs`` sequence. Multi-result lowerings that iterate over the builder
    result should use this helper so the single-output edge case behaves like
    the general multi-output path.
    """

    if isinstance(outputs, ir.Value):
        return (outputs,)
    return tuple(outputs)
