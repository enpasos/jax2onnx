# jax2onnx/plugins/jax/lax/_control_flow_utils.py

from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Any, cast

import onnx_ir as ir
from onnx_ir import Shape as IRShape

from jax import core

from jax2onnx.converter.lowering_dispatch import (
    lower_jaxpr_with_plugins,
)
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY


def _get_builder(ctx: LoweringContextProtocol) -> Any:
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError(
            "IR context missing builder; control-flow lowering requires builder support"
        )
    return builder


def builder_identity(
    ctx: LoweringContextProtocol, value: ir.Value, *, name_hint: str
) -> ir.Value:
    builder = _get_builder(ctx)
    out = cast(
        ir.Value,
        builder.Identity(
            value,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    orig_type = getattr(value, "type", None)
    if orig_type is not None:
        out.type = orig_type
    shape_obj = getattr(value, "shape", None)
    if shape_obj is not None:
        out.shape = shape_obj
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
    builder = _get_builder(ctx)
    casted = cast(
        ir.Value,
        builder.Cast(
            value,
            _outputs=[ctx.fresh_name(name_hint)],
            to=int(target_enum.value),
        ),
    )
    casted.type = ir.TensorType(target_enum)
    shape_obj = getattr(value, "shape", None)
    if shape_obj is not None:
        casted.shape = shape_obj
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
    builder = _get_builder(ctx)
    loop_result = cast(
        tuple[ir.Value, ...] | ir.Value,
        builder.Loop(
            *inputs,
            body=body,
            _outputs=output_names,
        ),
    )
    return loop_result


def lower_jaxpr_eqns(
    ctx: LoweringContextProtocol,
    jaxpr: core.Jaxpr,
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
    parent_ctx: LoweringContextProtocol, *, prefix: str
) -> LoweringContextProtocol:
    """Create a child IR context suitable for Loop/If subgraphs."""
    child_kwargs: dict[str, Any] = {
        "opset": getattr(parent_ctx.builder, "opset", 21),
        "enable_double_precision": getattr(
            parent_ctx.builder, "enable_double_precision", False
        ),
        "input_specs": [],
    }
    stacktrace_metadata = getattr(parent_ctx, "_stacktrace_metadata_enabled", None)
    if stacktrace_metadata is not None:
        child_kwargs["stacktrace_metadata"] = bool(stacktrace_metadata)
    try:
        child_ctx = type(parent_ctx)(**child_kwargs)
    except TypeError:
        child_kwargs.pop("stacktrace_metadata", None)
        child_ctx = type(parent_ctx)(**child_kwargs)
    child_ctx_any = cast(Any, child_ctx)
    child_ctx_any.record_primitive_calls_file = getattr(
        parent_ctx,
        "record_primitive_calls_file",
        None,
    )
    child_ctx_any._lowering_record_owner = getattr(
        parent_ctx,
        "_lowering_record_owner",
        parent_ctx,
    )
    child_ctx_any._primitive_call_records = getattr(
        child_ctx_any._lowering_record_owner,
        "_primitive_call_records",
        [],
    )
    child_ctx_any._function_mode = True
    child_ctx_any._inside_function_scope = True
    # Ensure builder emits constants as nodes (Functions/subgraphs cannot have initializers)
    child_builder = cast(Any, child_ctx_any.builder)
    child_builder._function_mode = True
    child_ctx_any._keep_function_float32 = getattr(
        parent_ctx, "_keep_function_float32", False
    )
    # Inherit known symbolic dimension origins so nested graphs can resolve them.
    child_ctx_any._sym_origin = dict(getattr(parent_ctx, "_sym_origin", {}))
    child_ctx_any._sym_origin_str = dict(getattr(parent_ctx, "_sym_origin_str", {}))

    # Prefix all fresh names so nested graphs remain unique.
    prefix_base = parent_ctx.fresh_name(prefix)
    orig_ctx_fresh = child_ctx_any.fresh_name
    setattr(
        child_ctx_any,
        "fresh_name",
        types.MethodType(
            lambda self, base, _orig=orig_ctx_fresh, _pref=prefix_base: _orig(
                f"{_pref}/{base}"
            ),
            child_ctx_any,
        ),
    )
    orig_builder_fresh = child_builder.fresh_name
    setattr(
        child_builder,
        "fresh_name",
        types.MethodType(
            lambda self, base, _orig=orig_builder_fresh, _pref=prefix_base: _orig(
                f"{_pref}/{base}"
            ),
            child_builder,
        ),
    )
    # Mirror builder bookkeeping so body graphs own independent lists.
    child_builder.inputs = []
    child_builder.outputs = []
    child_builder.nodes = []
    child_builder.initializers = []
    return cast(LoweringContextProtocol, child_ctx_any)


def relax_value_to_rank_only(val: ir.Value | None) -> None:
    if val is None or not isinstance(val, ir.Value):
        return
    shape_obj = getattr(val, "shape", None)
    dims = getattr(shape_obj, "dims", None)
    if dims is None and shape_obj is not None:
        try:
            dims = list(shape_obj) if isinstance(shape_obj, Iterable) else None
        except Exception:
            dims = None
    if dims is None:
        tensor_type = getattr(val, "type", None)
        if isinstance(tensor_type, ir.TensorType):
            shape_obj = getattr(tensor_type, "shape", None)
            dims = getattr(shape_obj, "dims", None)
            if dims is None and shape_obj is not None:
                try:
                    dims = list(shape_obj) if isinstance(shape_obj, Iterable) else None
                except Exception:
                    dims = None
    if not dims or len(dims) == 0:
        return
    if all(dim is None for dim in dims):
        return
    rank_only = ir.Shape(tuple(None for _ in dims))
    try:
        val.shape = rank_only
    except Exception:
        pass
    tensor_type = getattr(val, "type", None)
    if isinstance(tensor_type, ir.TensorType):
        dtype = tensor_type.dtype
        if dtype is not None:
            val.type = ir.TensorType(dtype)


def _extract_shape_dims(value: ir.Value | None) -> tuple[Any, ...] | None:
    if value is None:
        return None
    shape_obj = getattr(value, "shape", None)
    if shape_obj is None:
        return None
    dims = getattr(shape_obj, "dims", None)
    if dims is not None:
        return tuple(dims)
    if isinstance(shape_obj, Iterable):
        return tuple(shape_obj)
    return None


def create_loop_header_inputs(
    ctx: Any,
    *,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    """Create ``Loop`` iteration/condition inputs for a subgraph context."""

    builder = _get_builder(ctx)
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
    ctx: Any,
    template: ir.Value,
    *,
    name_hint: str,
) -> ir.Value:
    """Create a fresh ``ir.Value`` that mirrors ``template`` for subgraph inputs."""

    dtype = getattr(getattr(template, "type", None), "dtype", None)
    shape = getattr(template, "shape", None)
    dims = _extract_shape_dims(template)

    tensor_type = (
        ir.TensorType(dtype) if dtype is not None else getattr(template, "type", None)
    )

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
    ctx: Any,
    template: ir.Value,
    *,
    name_hint: str,
) -> ir.Value:
    """Clone ``template`` and append it to the subgraph builder inputs."""

    cloned = clone_value_for_subgraph(ctx, template, name_hint=name_hint)
    builder = _get_builder(ctx)
    builder.inputs.append(cloned)
    return cloned


def ensure_bool_value(ctx: Any, value: ir.Value, *, name_hint: str) -> ir.Value:
    """Cast ``value`` to BOOL when needed while preserving its shape."""

    dtype = getattr(getattr(value, "type", None), "dtype", None)
    if dtype == ir.DataType.BOOL:
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
