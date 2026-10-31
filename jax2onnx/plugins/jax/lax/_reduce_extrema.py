# jax2onnx/plugins/jax/lax/_reduce_extrema.py

"""SOURCE floating reduction extrema with native NaN and zero-sign semantics."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.jax.lax._opset_utils import builder_reduce_with_axes


def lower_source_float_extrema(
    ctx: LoweringContextProtocol, eqn: JaxprEqn, *, op_type: str
) -> bool:
    """Repair target extrema without rounding or replacing nonzero values.

    ReduceMin/Max kernels can discard NaNs or choose either word of a zero
    tie. Independently reduce the required predicates and select exact typed
    zero/NaN words. GatherElements avoids CPU Where's float64 negative-zero
    behavior gap. Empty reductions and no-axis identities keep their own paths.
    """
    if ctx.precision_policy is not PrecisionPolicy.SOURCE or op_type not in {
        "ReduceMin",
        "ReduceMax",
    }:
        return False
    source_var, target_var = eqn.invars[0], eqn.outvars[0]
    dtype = np.dtype(source_var.aval.dtype)
    if dtype.name not in {"float16", "bfloat16", "float32", "float64"}:
        return False
    axes = eqn.params["axes"]
    if axes == ():
        return False
    source = ctx.require_value_for_var(source_var)
    target = ctx.allocate_value_for_var(
        target_var, name_hint=ctx.fresh_name("source_float_extrema")
    )
    for var, value in ((source_var, source), (target_var, target)):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise ValueError(
                "SOURCE floating extrema canonical representation "
                "differs from its source"
            )
    if source.shape is None or target.shape is None or target.dtype is None:
        raise ValueError("SOURCE floating extrema requires canonical shapes and dtype")
    source_shape, shape, data_type = source.shape, target.shape, target.dtype
    prefix = op_type.lower()

    def emit(
        operation: str,
        inputs: list[ir.Value],
        result_shape: ir.Shape,
        result_dtype: ir.DataType = data_type,
        **attributes: int,
    ) -> ir.Value:
        return ctx.builder.op(
            operation,
            inputs,
            attributes,
            output=ir.Value(
                name=ctx.fresh_name(f"{prefix}_{operation.lower()}"),
                type=ir.TensorType(result_dtype),
                shape=result_shape,
            ),
        )

    def reduce_predicate(predicate: ir.Value) -> ir.Value:
        integers = emit(
            "Cast",
            [predicate],
            source_shape,
            ir.DataType.INT32,
            to=int(ir.DataType.INT32),
        )
        reduced = builder_reduce_with_axes(
            ctx,
            integers,
            op_type="ReduceMax",
            axes=axes,
            keepdims=0,
            name_hint=f"{prefix}_any",
        )
        reduced.type, reduced.shape = ir.TensorType(ir.DataType.INT32), shape
        return emit(
            "Cast", [reduced], shape, ir.DataType.BOOL, to=int(ir.DataType.BOOL)
        )

    base = builder_reduce_with_axes(
        ctx,
        source,
        op_type=op_type,
        axes=axes,
        keepdims=0,
        name_hint=f"{prefix}_candidate",
    )
    base.type, base.shape = ir.TensorType(data_type), shape
    zero = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{prefix}_positive_zero"), array=np.asarray(0, dtype=dtype)
    )
    zero_nan_choices = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{prefix}_zero_nan_choices"),
        array=np.asarray([0.0, -0.0, np.nan], dtype=dtype),
    )
    nan_index = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{prefix}_nan_index"), array=np.asarray(2, dtype=np.int64)
    )
    selection_axis = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{prefix}_selection_axis"),
        array=np.asarray([0], dtype=np.int64),
    )
    any_nan = reduce_predicate(emit("IsNaN", [source], source_shape, ir.DataType.BOOL))
    input_zero = emit("Equal", [source, zero], source_shape, ir.DataType.BOOL)
    reciprocal = emit("Reciprocal", [source], source_shape)
    negative_zero_sign = emit(
        "Less", [reciprocal, zero], source_shape, ir.DataType.BOOL
    )
    # For a zero minimum, any -0 wins. For a zero maximum, any +0 wins.
    # Restrict to zero inputs: e.g. max(-inf, -0) must still return -0.
    tie_sign = (
        negative_zero_sign
        if op_type == "ReduceMin"
        else emit("Not", [negative_zero_sign], source_shape, ir.DataType.BOOL)
    )
    any_zero_sign = reduce_predicate(
        emit("And", [input_zero, tie_sign], source_shape, ir.DataType.BOOL)
    )
    negative = (
        any_zero_sign
        if op_type == "ReduceMin"
        else emit("Not", [any_zero_sign], shape, ir.DataType.BOOL)
    )
    zero_index = emit(
        "Cast", [negative], shape, ir.DataType.INT64, to=int(ir.DataType.INT64)
    )
    correction_index = emit(
        "Where", [any_nan, nan_index, zero_index], shape, ir.DataType.INT64
    )
    correction = emit("Gather", [zero_nan_choices, correction_index], shape, axis=0)
    base_zero = emit("Equal", [base, zero], shape, ir.DataType.BOOL)
    replace = emit("Or", [base_zero, any_nan], shape, ir.DataType.BOOL)
    indices = emit(
        "Cast", [replace], shape, ir.DataType.INT64, to=int(ir.DataType.INT64)
    )
    selected_shape = ir.Shape((1, *shape.dims))
    choices_shape = ir.Shape((2, *shape.dims))
    choices = emit(
        "Concat",
        [
            emit("Unsqueeze", [base, selection_axis], selected_shape),
            emit("Unsqueeze", [correction, selection_axis], selected_shape),
        ],
        choices_shape,
        axis=0,
    )
    indices = emit(
        "Unsqueeze", [indices, selection_axis], selected_shape, ir.DataType.INT64
    )
    selected = emit("GatherElements", [choices, indices], selected_shape, axis=0)
    result = ctx.builder.op("Squeeze", [selected, selection_axis], output=target)
    ctx.bind_value_for_var(target_var, result)
    return True
