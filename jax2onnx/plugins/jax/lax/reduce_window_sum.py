# jax2onnx/plugins/jax/lax/reduce_window_sum.py

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax
from jax import lax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _is_static_int,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def reduce_window_sum(
    operand: jax.Array,
    *,
    window_dimensions: Sequence[int],
    window_strides: Sequence[int] | None = None,
    padding: Sequence[tuple[int, int]] | None = None,
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> jax.Array:
    """Bind the reduce_window_sum primitive without touching private JAX APIs."""
    dims = tuple(int(d) for d in window_dimensions)
    if window_strides is None:
        window_strides = (1,) * len(dims)
    if padding is None:
        padding = tuple((0, 0) for _ in dims)
    if base_dilation is None:
        base_dilation = (1,) * len(dims)
    if window_dilation is None:
        window_dilation = (1,) * len(dims)
    pad_pairs = tuple(tuple(int(p) for p in pair) for pair in padding)
    return lax.reduce_window_sum_p.bind(
        operand,
        window_dimensions=dims,
        window_strides=tuple(int(s) for s in window_strides),
        padding=pad_pairs,
        base_dilation=tuple(int(d) for d in base_dilation),
        window_dilation=tuple(int(d) for d in window_dilation),
    )


def _normalize_int_tuple(
    values: Sequence[int | np.integer], name: str
) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in values)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"{name} must be a sequence of integers, received {values!r}"
        ) from exc


def _flatten_padding(pads: Sequence[Sequence[int]]) -> list[int]:
    befores: list[int] = []
    afters: list[int] = []
    for pair in pads:
        if len(pair) != 2:
            raise ValueError(f"Padding entry must have length 2, received {pair!r}")
        lo, hi = int(pair[0]), int(pair[1])
        befores.append(lo)
        afters.append(hi)
    return befores + afters


@register_primitive(
    jaxpr_primitive=lax.reduce_window_sum_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window_sum.html",
    onnx=[
        {
            "component": "Conv",
            "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
        }
    ],
    since="v0.10.1",
    context="primitives.lax",
    component="reduce_window_sum",
    testcases=[
        {
            "testcase": "reduce_window_sum_valid",
            "callable": lambda x: reduce_window_sum(
                x,
                window_dimensions=(3,),
                window_strides=(1,),
                padding=((0, 0),),
            ),
            "input_shapes": [(8,)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Conv -> Squeeze"], no_unused_inputs=True
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "reduce_window_sum_same_padding",
            "callable": lambda x: reduce_window_sum(
                x,
                window_dimensions=(2, 2),
                window_strides=(1, 1),
                padding=((0, 1), (0, 1)),
            ),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Conv -> Squeeze"], no_unused_inputs=True
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "reduce_window_sum_stride_dilate",
            "callable": lambda x: reduce_window_sum(
                x,
                window_dimensions=(1, 3),
                window_strides=(2, 1),
                padding=((0, 0), (1, 1)),
                window_dilation=(1, 2),
            ),
            "input_shapes": [(3, 5)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Conv -> Squeeze"], no_unused_inputs=True
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "reduce_window_sum_int32",
            "callable": lambda x: reduce_window_sum(
                x,
                window_dimensions=(3,),
                window_strides=(1,),
                padding=((0, 0),),
            ),
            "input_values": [np.arange(8, dtype=np.int32)],
            "post_check_onnx_graph": EG(
                ["Cast -> Unsqueeze -> Unsqueeze -> Conv -> Squeeze -> Cast"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "reduce_window_sum_base_dilation",
            "callable": lambda x: reduce_window_sum(
                x,
                window_dimensions=(2,),
                window_strides=(1,),
                padding=((0, 0),),
                base_dilation=(2,),
            ),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                [
                    "Reshape -> Pad -> Reshape -> Slice -> Unsqueeze -> Unsqueeze -> Conv -> Squeeze"
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class ReduceWindowSumPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_window_sum`` via a Conv-based sliding sum."""

    _FLOAT_LIKE_DTYPES = {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }
    if hasattr(np, "bfloat16"):
        _FLOAT_LIKE_DTYPES.add(np.dtype(np.bfloat16))

    def lower(self, ctx: "IRContext", eqn):
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        params = getattr(eqn, "params", {})

        window_dims = _normalize_int_tuple(
            params.get("window_dimensions", ()), "window_dimensions"
        )
        if not window_dims:
            raise ValueError("reduce_window_sum requires non-empty window_dimensions")

        window_strides = _normalize_int_tuple(
            params.get("window_strides", (1,) * len(window_dims)), "window_strides"
        )
        padding_raw = params.get("padding", tuple((0, 0) for _ in window_dims))
        if isinstance(padding_raw, str):
            raise ValueError(
                "reduce_window_sum lowering expects concrete padding pairs inside JAX eqn"
            )
        padding_pairs = tuple(tuple(int(v) for v in pair) for pair in padding_raw)
        if len(padding_pairs) != len(window_dims):
            raise ValueError(
                f"Padding rank mismatch: expected {len(window_dims)}, received {padding_pairs}"
            )

        base_dilation = _normalize_int_tuple(
            params.get("base_dilation", (1,) * len(window_dims)), "base_dilation"
        )
        window_dilation = _normalize_int_tuple(
            params.get("window_dilation", (1,) * len(window_dims)), "window_dilation"
        )

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("reduce_window_sum_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reduce_window_sum_out")
        )

        aval_dtype = getattr(getattr(operand_var, "aval", None), "dtype", None)
        if aval_dtype is None:
            raise TypeError("reduce_window_sum operand dtype is unknown")
        np_dtype = np.dtype(aval_dtype)
        dtype_enum = _dtype_to_ir(np_dtype, ctx.builder.enable_double_precision)
        if dtype_enum is None:
            raise TypeError(f"Unsupported dtype for reduce_window_sum: {np_dtype}")

        enable_x64 = bool(getattr(ctx.builder, "enable_double_precision", False))
        conv_np_dtype = np_dtype
        conv_dtype_enum = dtype_enum
        needs_cast_back = False

        if np_dtype not in self._FLOAT_LIKE_DTYPES:
            conv_np_dtype = np.dtype(np.float64 if enable_x64 else np.float32)
            conv_dtype_enum = _dtype_to_ir(conv_np_dtype, enable_x64)
            needs_cast_back = True

        operand_shape = tuple(
            getattr(getattr(operand_var, "aval", None), "shape", ()) or ()
        )

        working_val = operand_val
        if needs_cast_back:
            cast_in = ctx.builder.Cast(
                operand_val,
                _outputs=[ctx.fresh_name("reduce_window_sum_cast_in")],
                to=int(conv_dtype_enum.value),
            )
            cast_in.type = ir.TensorType(conv_dtype_enum)
            _stamp_type_and_shape(cast_in, operand_shape)
            _ensure_value_metadata(ctx, cast_in)
            working_val = cast_in

        dilated_val, dilated_shape = self._apply_base_dilation(
            ctx,
            working_val,
            operand_shape,
            base_dilation,
            conv_np_dtype,
            conv_dtype_enum,
        )
        operand_shape = dilated_shape
        working_val = dilated_val

        unsq0_axes = _const_i64(
            ctx, np.asarray([0], dtype=np.int64), "rw_sum_unsq0_axes"
        )
        unsq0 = ctx.builder.Unsqueeze(
            working_val,
            unsq0_axes,
            _outputs=[ctx.fresh_name("reduce_window_sum_unsq0")],
        )
        unsq0.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(unsq0, (1,) + operand_shape)
        _ensure_value_metadata(ctx, unsq0)

        unsq1_axes = _const_i64(
            ctx, np.asarray([1], dtype=np.int64), "rw_sum_unsq1_axes"
        )
        unsq1 = ctx.builder.Unsqueeze(
            unsq0,
            unsq1_axes,
            _outputs=[ctx.fresh_name("reduce_window_sum_unsq1")],
        )
        unsq1.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(unsq1, (1, 1) + operand_shape)
        _ensure_value_metadata(ctx, unsq1)

        kernel_shape = (1, 1) + tuple(window_dims)
        kernel_array = np.ones(kernel_shape, dtype=conv_np_dtype)
        kernel_val = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("reduce_window_sum_kernel"),
            array=kernel_array,
        )
        kernel_val.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(kernel_val, kernel_shape)
        _ensure_value_metadata(ctx, kernel_val)

        conv_kwargs: dict[str, object] = {
            "strides": window_strides,
            "group": 1,
        }
        pads_flat = _flatten_padding(padding_pairs)
        if any(pads_flat):
            conv_kwargs["pads"] = pads_flat
        if any(d != 1 for d in window_dilation):
            conv_kwargs["dilations"] = window_dilation

        conv_out = ctx.builder.Conv(
            unsq1,
            kernel_val,
            _outputs=[ctx.fresh_name("reduce_window_sum_conv")],
            **conv_kwargs,
        )
        conv_out.type = ir.TensorType(conv_dtype_enum)
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()) or ())
        _stamp_type_and_shape(conv_out, (1, 1) + out_shape)
        _ensure_value_metadata(ctx, conv_out)

        squeeze_axes = _const_i64(
            ctx, np.asarray([0, 1], dtype=np.int64), "rw_sum_squeeze_axes"
        )
        squeezed = ctx.builder.Squeeze(
            conv_out,
            squeeze_axes,
            _outputs=[
                getattr(out_val, "name", None) or ctx.fresh_name("reduce_window_sum")
            ],
        )
        squeezed.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(squeezed, out_shape)
        _ensure_value_metadata(ctx, squeezed)

        final_val = squeezed
        if needs_cast_back:
            out_dtype_enum = _dtype_to_ir(np_dtype, enable_x64)
            cast_out = ctx.builder.Cast(
                squeezed,
                _outputs=[ctx.fresh_name("reduce_window_sum_cast_out")],
                to=int(out_dtype_enum.value),
            )
            cast_out.type = ir.TensorType(out_dtype_enum)
            _stamp_type_and_shape(cast_out, out_shape)
            _ensure_value_metadata(ctx, cast_out)
            final_val = cast_out

        ctx.bind_value_for_var(out_var, final_val)

    def _apply_base_dilation(
        self,
        ctx: "IRContext",
        tensor: ir.Value,
        operand_shape: Sequence[int | np.integer],
        base_dilation: tuple[int, ...],
        conv_np_dtype: np.dtype,
        conv_dtype_enum: ir.DataType,
    ) -> tuple[ir.Value, tuple[int, ...]]:
        if all(int(b) == 1 for b in base_dilation):
            return tensor, tuple(
                int(dim) if isinstance(dim, (int, np.integer)) else dim
                for dim in operand_shape
            )

        if any(not _is_static_int(dim) for dim in operand_shape):
            raise NotImplementedError(
                "reduce_window_sum base_dilation currently requires static operand shape"
            )

        shape_list = [int(dim) for dim in operand_shape]
        original_dims = [int(dim) for dim in operand_shape]
        builder = ctx.builder
        work_val = tensor

        zero_scalar = builder.add_initializer_from_array(
            name=ctx.fresh_name("reduce_window_sum_zero"),
            array=np.asarray(0, dtype=conv_np_dtype),
        )
        zero_scalar.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(zero_scalar, ())
        _ensure_value_metadata(ctx, zero_scalar)

        for axis, dilation in enumerate(base_dilation):
            dilation = int(dilation)
            if dilation == 1:
                continue

            dim_value = original_dims[axis]
            if dim_value == 0:
                continue

            reshape_dims = shape_list[:axis] + [dim_value, 1] + shape_list[axis + 1 :]
            reshape_shape_val = builder.add_initializer_from_array(
                name=ctx.fresh_name("reduce_window_sum_base_shape"),
                array=np.asarray(reshape_dims, dtype=np.int64),
            )
            reshape_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(reshape_shape_val, (len(reshape_dims),))
            _ensure_value_metadata(ctx, reshape_shape_val)

            reshaped = builder.Reshape(
                work_val,
                reshape_shape_val,
                _outputs=[ctx.fresh_name("reduce_window_sum_base_reshape1")],
            )
            reshaped.type = ir.TensorType(conv_dtype_enum)
            _stamp_type_and_shape(reshaped, tuple(reshape_dims))
            _ensure_value_metadata(ctx, reshaped)
            work_val = reshaped
            shape_list = list(reshape_dims)

            pads = [0] * (2 * len(shape_list))
            pads[axis + 1 + len(shape_list)] = dilation - 1
            pads_val = builder.add_initializer_from_array(
                name=ctx.fresh_name("reduce_window_sum_base_pads"),
                array=np.asarray(pads, dtype=np.int64),
            )
            pads_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(pads_val, (len(pads),))
            _ensure_value_metadata(ctx, pads_val)

            padded = builder.Pad(
                work_val,
                pads_val,
                zero_scalar,
                mode="constant",
                _outputs=[ctx.fresh_name("reduce_window_sum_base_pad")],
            )
            padded.type = ir.TensorType(conv_dtype_enum)
            padded_dims = list(shape_list)
            padded_dims[axis + 1] = 1 + (dilation - 1)
            _stamp_type_and_shape(padded, tuple(padded_dims))
            _ensure_value_metadata(ctx, padded)
            work_val = padded
            shape_list = padded_dims

            collapsed_dims = (
                shape_list[:axis] + [dim_value * dilation] + shape_list[axis + 2 :]
            )
            collapse_shape_val = builder.add_initializer_from_array(
                name=ctx.fresh_name("reduce_window_sum_base_shape2"),
                array=np.asarray(collapsed_dims, dtype=np.int64),
            )
            collapse_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(collapse_shape_val, (len(collapsed_dims),))
            _ensure_value_metadata(ctx, collapse_shape_val)

            collapsed = builder.Reshape(
                work_val,
                collapse_shape_val,
                _outputs=[ctx.fresh_name("reduce_window_sum_base_reshape2")],
            )
            collapsed.type = ir.TensorType(conv_dtype_enum)
            _stamp_type_and_shape(collapsed, tuple(collapsed_dims))
            _ensure_value_metadata(ctx, collapsed)
            work_val = collapsed
            shape_list = list(collapsed_dims)

            desired = (dim_value - 1) * dilation + 1
            total = dim_value * dilation
            if desired < total:
                starts_val = builder.add_initializer_from_array(
                    name=ctx.fresh_name("reduce_window_sum_base_starts"),
                    array=np.zeros((1,), dtype=np.int64),
                )
                ends_val = builder.add_initializer_from_array(
                    name=ctx.fresh_name("reduce_window_sum_base_ends"),
                    array=np.asarray([desired], dtype=np.int64),
                )
                axes_val = builder.add_initializer_from_array(
                    name=ctx.fresh_name("reduce_window_sum_base_axes"),
                    array=np.asarray([axis], dtype=np.int64),
                )
                steps_val = builder.add_initializer_from_array(
                    name=ctx.fresh_name("reduce_window_sum_base_steps"),
                    array=np.asarray([1], dtype=np.int64),
                )
                for val in (starts_val, ends_val, axes_val, steps_val):
                    val.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(val, (1,))
                    _ensure_value_metadata(ctx, val)

                sliced = builder.Slice(
                    work_val,
                    starts_val,
                    ends_val,
                    axes_val,
                    steps_val,
                    _outputs=[ctx.fresh_name("reduce_window_sum_base_slice")],
                )
                sliced.type = ir.TensorType(conv_dtype_enum)
                new_dims = list(shape_list)
                new_dims[axis] = desired
                _stamp_type_and_shape(sliced, tuple(new_dims))
                _ensure_value_metadata(ctx, sliced)
                work_val = sliced
                shape_list = new_dims
            else:
                shape_list[axis] = total

        final_shape = tuple(int(d) for d in shape_list)
        return work_val, final_shape
