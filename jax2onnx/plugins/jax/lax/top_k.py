# jax2onnx/plugins/jax/lax/top_k.py

from __future__ import annotations

from typing import Any

import jax
import numpy as np

import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.source_top_k import (
    native_top_k_result_is_empty,
    require_native_top_k_emission,
)
from jax2onnx.plugins._complex_utils import coerce_dim_values
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.top_k_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.top_k.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="0.10.2",
    context="primitives.lax",
    component="top_k",
    testcases=[
        {
            "testcase": "top_k_last_axis",
            "callable": lambda x: jax.lax.top_k(x, 3),
            "input_shapes": [(5,)],
            "expected_output_dtypes": [np.float32, np.int32],
        },
        {
            "testcase": "top_k_matrix",
            "callable": lambda x: jax.lax.top_k(x, 2),
            "input_shapes": [(4, 6)],
            "expected_output_dtypes": [np.float32, np.int32],
        },
    ],
)
class TopKPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            self._lower_source_native(ctx, eqn)
            return
        (arr_var,) = eqn.invars
        values_var, indices_var = eqn.outvars

        arr_val = ctx.require_value_for_var(arr_var)
        values_spec = ctx.allocate_value_for_var(
            values_var, name_hint=ctx.fresh_name("topk_values")
        )
        indices_spec = ctx.allocate_value_for_var(
            indices_var, name_hint=ctx.fresh_name("topk_indices")
        )

        arr_shape = tuple(arr_var.aval.shape)
        arr_dtype = arr_val.dtype

        k = int(eqn.params.get("k", 1))
        axis = int(eqn.params.get("axis", eqn.params.get("dimension", -1)))
        if axis < 0 and arr_shape:
            axis += len(arr_shape)

        k_val = _const_i64(ctx, np.asarray([k], dtype=np.int64), "topk_k")
        values_spec.name = ctx.fresh_name("TopK_Values")
        values, indices = ctx.builder.op_multi_out(
            "TopK",
            [arr_val, k_val],
            {"axis": axis, "largest": 1, "sorted": 1},
            outputs=[values_spec, ir.Value(name=ctx.fresh_name("TopK_Indices"))],
        )

        if arr_dtype is not None:
            values.type = ir.TensorType(arr_dtype)
        indices.type = ir.TensorType(ir.DataType.INT64)

        result_shape_list = list(arr_shape)
        if arr_shape:
            result_shape_list[axis] = k
        result_shape = tuple(result_shape_list)

        _stamp_type_and_shape(values, result_shape)
        _stamp_type_and_shape(indices, result_shape)
        _ensure_value_metadata(ctx, values)
        _ensure_value_metadata(ctx, indices)

        target_idx_dtype: np.dtype[Any] = np.dtype(indices_var.aval.dtype)
        idx_dtype_enum = numpy_dtype_to_ir(target_idx_dtype)
        result_indices = indices
        if idx_dtype_enum != ir.DataType.INT64:
            result_indices = ctx.builder.op(
                "Cast",
                [indices],
                {"to": int(idx_dtype_enum.value)},
                output=indices_spec,
            )
            result_indices.type = ir.TensorType(idx_dtype_enum)
            _stamp_type_and_shape(result_indices, result_shape)
            _ensure_value_metadata(ctx, result_indices)

        ctx.bind_value_for_var(values_var, values)
        ctx.bind_value_for_var(indices_var, result_indices)

    def _lower_source_native(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        require_native_top_k_emission(eqn, ctx)
        source_var = eqn.invars[0]
        source = representation_for_aval(source_var.aval)
        data = ctx.require_value_for_var(source_var)
        if not source.matches_ir_value(data):
            raise TypeError(
                "SOURCE TopK physical input differs from its native declaration"
            )
        values_var, indices_var = eqn.outvars
        values_output = ctx.allocate_value_for_var(
            values_var, name_hint=ctx.fresh_name("topk_values")
        )
        indices_output = ctx.allocate_value_for_var(
            indices_var, name_hint=ctx.fresh_name("topk_indices")
        )
        axis, k = eqn.params["axis"], eqn.params["k"]

        def emit(
            name: str,
            operands: list[ir.Value],
            dtype: ir.DataType,
            shape: tuple[object, ...],
            attributes: dict[str, Any] | None = None,
            output: ir.Value | None = None,
        ) -> ir.Value:
            result = ctx.builder.op(name, operands, attributes, output=output)
            result.type = ir.TensorType(dtype)
            _stamp_type_and_shape(result, coerce_dim_values(shape))
            _ensure_value_metadata(ctx, result)
            return result

        if native_top_k_result_is_empty(source_var.aval.shape, k):
            # Slice preserves the exact selected dimension even when another
            # extent proves the result empty. No sorting or word keys are needed.
            zero = _const_i64(ctx, [0], "topk_empty_zero")
            end = zero if k == 0 else _const_i64(ctx, [k], "topk_empty_end")
            axis_value = _const_i64(ctx, [axis], "topk_empty_axis")
            values = emit(
                "Slice",
                [data, zero, end, axis_value],
                source.ir_dtype,
                tuple(values_var.aval.shape),
                output=values_output,
            )
            indices = emit(
                "Cast",
                [values],
                ir.DataType.INT32,
                tuple(indices_var.aval.shape),
                {"to": int(ir.DataType.INT32.value)},
                indices_output,
            )
        else:
            shape = tuple(source_var.aval.shape)
            if source.ir_dtype in (
                ir.DataType.FLOAT16,
                ir.DataType.BFLOAT16,
                ir.DataType.FLOAT,
                ir.DataType.DOUBLE,
            ):
                width = {
                    ir.DataType.FLOAT16: 16,
                    ir.DataType.BFLOAT16: 16,
                    ir.DataType.FLOAT: 32,
                    ir.DataType.DOUBLE: 64,
                }[source.ir_dtype]
                word_type = {
                    16: ir.DataType.INT16,
                    32: ir.DataType.INT32,
                    64: ir.DataType.INT64,
                }[width]
                word_dtype = {16: np.int16, 32: np.int32, 64: np.int64}[width]
                words = emit(
                    "BitCast",
                    [data],
                    word_type,
                    shape,
                    {"to": int(word_type.value)},
                )
                if width == 16:
                    # Signed-word widening is exact; selected FP16/BF16 values stay
                    # untouched. INT64 comparison/Xor avoids narrow kernels.
                    words = emit(
                        "Cast",
                        [words],
                        ir.DataType.INT64,
                        shape,
                        {"to": int(ir.DataType.INT64.value)},
                    )
                    word_type, word_dtype = ir.DataType.INT64, np.int64
                zero = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("topk_word_zero"),
                    array=np.asarray(0, dtype=word_dtype),
                )
                signless = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("topk_word_signless_max"),
                    array=np.asarray((1 << (width - 1)) - 1, dtype=word_dtype),
                )
                negative = emit("Less", [words, zero], ir.DataType.BOOL, shape)
                reversed_word = emit("BitwiseXor", [words, signless], word_type, shape)
                keys = emit("Where", [negative, reversed_word, words], word_type, shape)
            elif source.ir_dtype is ir.DataType.UINT64:
                words = emit(
                    "BitCast",
                    [data],
                    ir.DataType.INT64,
                    shape,
                    {"to": int(ir.DataType.INT64.value)},
                )
                sign = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("topk_unsigned_sign"),
                    array=np.asarray(-(1 << 63), dtype=np.int64),
                )
                keys = emit("BitwiseXor", [words, sign], ir.DataType.INT64, shape)
            else:
                keys = data
            if keys.dtype != ir.DataType.INT64:
                keys = emit(
                    "Cast",
                    [keys],
                    ir.DataType.INT64,
                    shape,
                    {"to": int(ir.DataType.INT64.value)},
                )
            k_value = _const_i64(ctx, [k], "topk_k")
            ranked = ctx.builder.op_multi_out(
                "TopK",
                [keys, k_value],
                {"axis": axis, "largest": 1, "sorted": 1},
                outputs=[
                    ir.Value(name=ctx.fresh_name("topk_order_keys")),
                    ir.Value(name=ctx.fresh_name("topk_order_indices")),
                ],
            )
            for result in ranked:
                result.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(
                    result, coerce_dim_values(tuple(values_var.aval.shape))
                )
                _ensure_value_metadata(ctx, result)
            values = emit(
                "GatherElements",
                [data, ranked[1]],
                source.ir_dtype,
                tuple(values_var.aval.shape),
                {"axis": axis},
                values_output,
            )
            indices = emit(
                "Cast",
                [ranked[1]],
                ir.DataType.INT32,
                tuple(indices_var.aval.shape),
                {"to": int(ir.DataType.INT32.value)},
                indices_output,
            )
        ctx.bind_value_for_var(values_var, values)
        ctx.bind_value_for_var(indices_var, indices)
