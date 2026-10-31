# jax2onnx/plugins/jax/lax/_source_complex_layout.py

"""The closed static SOURCE complex copy/layout family."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import _Site, _tensor, _validate_complex_layout
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._complex_utils import coerce_dim_values
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


def lower_source_complex_layout(ctx: LoweringContextProtocol, eqn: Any) -> bool:
    """Lower exactly six complex layouts after checking every physical input."""
    if ctx.precision_policy is not PrecisionPolicy.SOURCE:
        return False
    if not any(
        isinstance(var.aval.dtype, np.dtype)
        and var.aval.dtype in (np.dtype(np.complex64), np.dtype(np.complex128))
        for var in (*eqn.invars, *eqn.outvars)
    ):
        # Extended typed-key dtypes are not NumPy dtypes. In particular, no
        # np.dtype(key_dtype) coercion or physical trailing-two inference occurs.
        return False

    name = eqn.primitive.name
    site = _Site("static complex layout lowering", ctx.builder.opset, name)
    _validate_complex_layout(
        eqn,
        [_tensor(var, site) for var in eqn.invars],
        [_tensor(var, site) for var in eqn.outvars],
        site,
    )
    values = [ctx.require_value_for_var(var) for var in eqn.invars]
    for var, value in zip(eqn.invars, values, strict=True):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise TypeError(
                "SOURCE complex layout physical input differs from its declared pair"
            )

    out_var = eqn.outvars[0]
    representation = representation_for_aval(out_var.aval)
    output = ctx.allocate_value_for_var(
        out_var, name_hint=ctx.fresh_name("source_complex_layout")
    )
    if not representation.matches_ir_value(output):
        raise TypeError(
            "SOURCE complex layout physical output differs from its declared pair"
        )
    dtype = representation.ir_dtype
    shape = tuple(int(dim) for dim in eqn.invars[0].aval.shape)
    rank = len(shape)
    source = values[0]

    def stamp(value: ir.Value, kind: ir.DataType, dims: tuple[int, ...]) -> ir.Value:
        value.type = ir.TensorType(kind)
        _stamp_type_and_shape(value, coerce_dim_values(dims))
        _ensure_value_metadata(ctx, value)
        return value

    result: ir.Value
    if name == "copy":
        result = ctx.builder.op("Identity", [source], output=output)
    elif name == "reshape":
        permutation = eqn.params["dimensions"]
        identity_mapping = permutation is None or tuple(permutation) == tuple(
            range(rank)
        )
        target = tuple(int(dim) for dim in eqn.params["new_sizes"])
        if identity_mapping and target == shape:
            result = ctx.builder.op("Identity", [source], output=output)
        else:
            if not identity_mapping:
                source = stamp(
                    ctx.builder.op(
                        "Transpose",
                        [source],
                        {"perm": ir.AttrInt64s("perm", (*permutation, rank))},
                    ),
                    dtype,
                    tuple(shape[axis] for axis in permutation) + (2,),
                )
            sizes = _const_i64(
                ctx,
                np.asarray((*target, 2), dtype=np.int64),
                "source_complex_reshape_sizes",
            )
            result = ctx.builder.op(
                "Reshape", [source, sizes], {"allowzero": 1}, output=output
            )
    elif name == "transpose":
        permutation = tuple(int(axis) for axis in eqn.params["permutation"])
        if permutation == tuple(range(rank)):
            result = ctx.builder.op("Identity", [source], output=output)
        else:
            result = ctx.builder.op(
                "Transpose",
                [source],
                {"perm": ir.AttrInt64s("perm", (*permutation, rank))},
                output=output,
            )
    elif name == "squeeze":
        axes = tuple(int(axis) for axis in eqn.params["dimensions"])
        if not axes:
            result = ctx.builder.op("Identity", [source], output=output)
        else:
            axes_value = _const_i64(
                ctx, np.asarray(axes, dtype=np.int64), "source_complex_squeeze_axes"
            )
            result = ctx.builder.op("Squeeze", [source, axes_value], output=output)
    elif name == "concatenate":
        result = ctx.builder.op(
            "Concat", values, {"axis": int(eqn.params["dimension"])}, output=output
        )
    else:
        # The admission owner proves this final member is canonical native rev.
        axes = tuple(int(axis) for axis in eqn.params["dimensions"])
        if not axes:
            result = ctx.builder.op("Identity", [source], output=output)
        else:
            zero = _const_i64(ctx, np.asarray(0, dtype=np.int64), "complex_rev_zero")
            one = _const_i64(ctx, np.asarray(1, dtype=np.int64), "complex_rev_one")
            current = source
            for index, axis in enumerate(axes):
                extent = shape[axis]
                limit = _const_i64(
                    ctx, np.asarray(extent, dtype=np.int64), "complex_rev_extent"
                )
                ordered = stamp(
                    ctx.builder.op("Range", [zero, limit, one]),
                    ir.DataType.INT64,
                    (extent,),
                )
                last = _const_i64(
                    ctx, np.asarray(extent - 1, dtype=np.int64), "complex_rev_last"
                )
                reversed_indices = stamp(
                    ctx.builder.op("Sub", [last, ordered]),
                    ir.DataType.INT64,
                    (extent,),
                )
                if index + 1 == len(axes):
                    current = ctx.builder.op(
                        "Gather",
                        [current, reversed_indices],
                        {"axis": axis},
                        output=output,
                    )
                else:
                    current = stamp(
                        ctx.builder.op(
                            "Gather", [current, reversed_indices], {"axis": axis}
                        ),
                        dtype,
                        shape + (2,),
                    )
            result = current

    if not representation.matches_ir_value(result):
        raise TypeError("SOURCE complex layout result changed its declared pair")
    ctx.bind_value_for_var(out_var, result)
    return True
