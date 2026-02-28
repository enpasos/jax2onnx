# jax2onnx/plugins/jax/lax/reduce_window_max.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax.reduce_window_sum import (
    _flatten_padding,
    _normalize_int_tuple,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def reduce_window_max(
    operand: jax.Array,
    *,
    window_dimensions: tuple[int, ...],
    window_strides: tuple[int, ...] | None = None,
    padding: tuple[tuple[int, int], ...] | None = None,
    base_dilation: tuple[int, ...] | None = None,
    window_dilation: tuple[int, ...] | None = None,
) -> jax.Array:
    """Bind ``lax.reduce_window_max_p`` without relying on removed lax wrappers."""

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
    return jax.lax.reduce_window_max_p.bind(
        operand,
        window_dimensions=dims,
        window_strides=tuple(int(s) for s in window_strides),
        padding=pad_pairs,
        base_dilation=tuple(int(d) for d in base_dilation),
        window_dilation=tuple(int(d) for d in window_dilation),
    )


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_window_max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        }
    ],
    since="0.12.2",
    context="primitives.lax",
    component="reduce_window_max",
    testcases=[
        {
            "testcase": "reduce_window_max_primitive",
            "callable": lambda x: reduce_window_max(
                x,
                window_dimensions=(2, 2),
                window_strides=(1, 1),
                padding=((0, 1), (0, 1)),
            ),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> MaxPool -> Squeeze"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class ReduceWindowMaxPrimitivePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_window_max_p`` to ONNX MaxPool."""

    _MAXPOOL_DTYPES = {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }
    if hasattr(np, "bfloat16"):
        _MAXPOOL_DTYPES.add(np.dtype(np.bfloat16))

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        params = getattr(eqn, "params", {})
        window_dims = _normalize_int_tuple(
            params.get("window_dimensions", ()), "window_dimensions"
        )
        if not window_dims:
            raise ValueError("reduce_window_max requires non-empty window_dimensions")
        window_strides = _normalize_int_tuple(
            params.get("window_strides", (1,) * len(window_dims)), "window_strides"
        )
        padding_raw = params.get("padding", tuple((0, 0) for _ in window_dims))
        if isinstance(padding_raw, str):
            raise ValueError(
                "reduce_window_max lowering expects concrete padding pairs inside JAX eqn"
            )
        padding_pairs = tuple(tuple(int(v) for v in pair) for pair in padding_raw)
        if len(padding_pairs) != len(window_dims):
            raise ValueError(
                f"Padding rank mismatch: expected {len(window_dims)}, received {padding_pairs}"
            )
        base_dilation = _normalize_int_tuple(
            params.get("base_dilation", (1,) * len(window_dims)), "base_dilation"
        )
        if any(d != 1 for d in base_dilation):
            raise NotImplementedError(
                "reduce_window_max with base_dilation != 1 is not yet supported"
            )
        window_dilation = _normalize_int_tuple(
            params.get("window_dilation", (1,) * len(window_dims)), "window_dilation"
        )

        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        np_dtype = np.dtype(getattr(getattr(operand_var, "aval", None), "dtype"))
        if np_dtype not in self._MAXPOOL_DTYPES:
            raise TypeError(
                f"reduce_window_max currently supports floating operand dtypes only, got {np_dtype}"
            )

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("reduce_window_max_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reduce_window_max_out")
        )

        operand_shape = tuple(getattr(getattr(operand_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        dtype_enum = _dtype_to_ir(np_dtype, ctx.builder.enable_double_precision)
        if dtype_enum is None:
            raise TypeError(f"Unsupported dtype for reduce_window_max: {np_dtype}")

        axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "rwm_axes0")
        axes1 = _const_i64(ctx, np.asarray([1], dtype=np.int64), "rwm_axes1")
        unsq0 = ctx.builder.Unsqueeze(
            operand_val,
            axes0,
            _outputs=[ctx.fresh_name("reduce_window_max_unsq0")],
        )
        unsq0.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(unsq0, (1,) + operand_shape)
        _ensure_value_metadata(ctx, unsq0)

        unsq1 = ctx.builder.Unsqueeze(
            unsq0,
            axes1,
            _outputs=[ctx.fresh_name("reduce_window_max_unsq1")],
        )
        unsq1.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(unsq1, (1, 1) + operand_shape)
        _ensure_value_metadata(ctx, unsq1)

        maxpool_kwargs: dict[str, object] = {
            "kernel_shape": tuple(window_dims),
            "strides": tuple(window_strides),
        }
        pads_flat = _flatten_padding(padding_pairs)
        if any(pads_flat):
            maxpool_kwargs["pads"] = pads_flat
        if any(d != 1 for d in window_dilation):
            maxpool_kwargs["dilations"] = tuple(window_dilation)

        pooled = ctx.builder.MaxPool(
            unsq1,
            _outputs=[ctx.fresh_name("reduce_window_max_pool")],
            **maxpool_kwargs,
        )
        pooled.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(pooled, (1, 1) + out_shape)
        _ensure_value_metadata(ctx, pooled)

        squeeze_axes = _const_i64(
            ctx, np.asarray([0, 1], dtype=np.int64), "rwm_squeeze_axes"
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "reduce_window_max"
        )
        result = ctx.builder.Squeeze(
            pooled,
            squeeze_axes,
            _outputs=[desired_name],
        )
        result.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
