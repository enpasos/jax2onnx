# jax2onnx/plugins/jax/lax/_control_flow_utils.py

from __future__ import annotations

import onnx_ir as ir
from onnx_ir import Shape as IRShape

from jax2onnx.converter.lowering_dispatch import (
    lower_jaxpr_with_plugins,
)
from jax2onnx.converter.typing_support import IRBuilderAPI, LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx._compat.jax import Jaxpr
from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY


def builder_identity(
    ctx: LoweringContextProtocol, value: ir.Value, *, name_hint: str
) -> ir.Value:
    builder: IRBuilderAPI = ctx.builder
    output_name = ctx.fresh_name(name_hint)
    out = builder.op("Identity", [value])
    out.name = output_name
    out.type = value.type
    out.shape = value.shape
    dims = _extract_shape_dims(value)
    if dims is not None:
        _stamp_type_and_shape(out, dims)
    _ensure_value_metadata(ctx, out)
    return out


def builder_cast(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    target_enum: ir.DataType,
    *,
    name_hint: str,
) -> ir.Value:
    builder: IRBuilderAPI = ctx.builder
    output_name = ctx.fresh_name(name_hint)
    casted = builder.op("Cast", [value], {"to": int(target_enum.value)})
    casted.name = output_name
    casted.type = ir.TensorType(target_enum)
    casted.shape = value.shape
    dims = _extract_shape_dims(value)
    if dims is not None:
        _stamp_type_and_shape(casted, dims)
    _ensure_value_metadata(ctx, casted)
    return casted


def builder_loop(
    ctx: LoweringContextProtocol,
    *inputs: ir.Value,
    body: ir.Graph,
    output_names: list[str],
) -> tuple[ir.Value, ...] | ir.Value:
    builder: IRBuilderAPI = ctx.builder
    results = builder.op_multi_out(
        "Loop",
        inputs,
        {"body": ir.AttrGraph("body", body)},
        num_outputs=len(output_names),
    )
    for result, name in zip(results, output_names):
        result.name = name
    return results[0] if len(results) == 1 else tuple(results)


def lower_jaxpr_eqns(
    ctx: LoweringContextProtocol,
    jaxpr: Jaxpr,
    *,
    source: str = "control_flow",
) -> None:
    """Lower every equation in ``jaxpr`` using the registered plugins."""
    lower_jaxpr_with_plugins(
        ctx=ctx,
        jaxpr=jaxpr,
        registry=PLUGIN_REGISTRY,
        source=source,
    )


def make_subgraph_context(
    parent_ctx: LoweringContextProtocol,
    *,
    prefix: str,
    implicit_captures: bool = False,
) -> LoweringContextProtocol:
    """Create a body context; If branches opt into implicit outer captures."""
    return parent_ctx.create_child_context(
        scope="control_flow", prefix=prefix, implicit_captures=implicit_captures
    )


def relax_value_to_rank_only(val: ir.Value | None) -> None:
    if val is None or val.shape is None or len(val.shape) == 0:
        return
    val.shape = ir.Shape((None,) * len(val.shape))
    if isinstance(val.type, ir.TensorType):
        val.type = ir.TensorType(val.type.dtype)


def _extract_shape_dims(
    value: ir.Value | None,
) -> tuple[int | ir.SymbolicDim, ...] | None:
    if value is None or value.shape is None:
        return None
    return value.shape.dims


def create_loop_header_inputs(
    ctx: LoweringContextProtocol,
    *,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    """Create ``Loop`` iteration/condition inputs for a subgraph context."""

    builder: IRBuilderAPI = ctx.builder
    iter_input = ir.Value(
        name=ctx.fresh_name(f"{prefix}_iter"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=IRShape(()),
    )
    cond_input = ir.Value(
        name=ctx.fresh_name(f"{prefix}_cond_in"),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=IRShape(()),
    )
    builder.inputs.extend([iter_input, cond_input])
    _stamp_type_and_shape(iter_input, ())
    _stamp_type_and_shape(cond_input, ())
    _ensure_value_metadata(ctx, iter_input)
    _ensure_value_metadata(ctx, cond_input)
    return iter_input, cond_input


def clone_value_for_subgraph(
    ctx: LoweringContextProtocol,
    template: ir.Value,
    *,
    name_hint: str,
) -> ir.Value:
    """Create a fresh ``ir.Value`` that mirrors ``template`` for subgraph inputs."""

    dtype = template.dtype
    shape = template.shape
    dims = _extract_shape_dims(template)

    tensor_type = ir.TensorType(dtype) if dtype is not None else template.type

    cloned = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=tensor_type,
        shape=shape,
    )

    if dims is not None:
        _stamp_type_and_shape(cloned, dims)
    elif shape is not None:
        cloned.shape = shape

    _ensure_value_metadata(ctx, cloned)
    return cloned


def clone_input_for_subgraph(
    ctx: LoweringContextProtocol,
    template: ir.Value,
    *,
    name_hint: str,
) -> ir.Value:
    """Clone ``template`` and append it to the subgraph builder inputs."""

    cloned = clone_value_for_subgraph(ctx, template, name_hint=name_hint)
    builder: IRBuilderAPI = ctx.builder
    builder.inputs.append(cloned)
    return cloned


def ensure_bool_value(
    ctx: LoweringContextProtocol, value: ir.Value, *, name_hint: str
) -> ir.Value:
    """Cast ``value`` to BOOL when needed while preserving its shape."""

    if value.dtype == ir.DataType.BOOL:
        return value

    bool_val = builder_cast(
        ctx,
        value,
        ir.DataType.BOOL,
        name_hint=name_hint,
    )
    dims = _extract_shape_dims(value)
    if dims is not None:
        _stamp_type_and_shape(bool_val, dims)
    _ensure_value_metadata(ctx, bool_val)
    return bool_val
