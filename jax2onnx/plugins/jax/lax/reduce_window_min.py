# jax2onnx/plugins/jax/lax/reduce_window_min.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import DimInput, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax.reduce_window_sum import (
    _flatten_padding,
    _normalize_int_tuple,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.reduce_window_min_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        },
        {
            "component": "Neg",
            "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html",
        },
    ],
    since="0.19.1",
    context="primitives.lax",
    component="reduce_window_min",
    testcases=[
        {
            # lax.min with its identity reaches the converter as
            # reduce_window_min rather than the generic reduce_window.
            "testcase": "reduce_window_min_primitive",
            "callable": lambda x: jax.lax.reduce_window(
                x, jnp.inf, jax.lax.min, (2, 2), (1, 1), ((0, 1), (0, 1))
            ),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Neg -> MaxPool -> Neg -> Squeeze"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_window_min_stride_dilation",
            "callable": lambda x: jax.lax.reduce_window(
                x,
                jnp.inf,
                jax.lax.min,
                (2, 2),
                (2, 1),
                "VALID",
                window_dilation=(1, 2),
            ),
            "input_shapes": [(6, 7)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Neg -> MaxPool -> Neg -> Squeeze:3x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceWindowMinPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_window_min_p`` as ``-MaxPool(-x)``."""

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        (operand_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = eqn.params

        window = _normalize_int_tuple(
            params.get("window_dimensions", ()), "window_dimensions"
        )
        if not window:
            raise ValueError("reduce_window_min requires non-empty window_dimensions")
        strides = _normalize_int_tuple(
            params.get("window_strides", (1,) * len(window)), "window_strides"
        )
        padding = tuple(
            (int(low), int(high))
            for low, high in params.get("padding", ((0, 0),) * len(window))
        )
        if len(padding) != len(window):
            raise ValueError(
                f"Padding rank mismatch: expected {len(window)}, received {padding}"
            )
        base_dilation = _normalize_int_tuple(
            params.get("base_dilation", (1,) * len(window)), "base_dilation"
        )
        if any(dilation != 1 for dilation in base_dilation):
            raise NotImplementedError(
                "reduce_window_min with base_dilation != 1 is not yet supported"
            )
        window_dilation = _normalize_int_tuple(
            params.get("window_dilation", (1,) * len(window)), "window_dilation"
        )

        dtype = np.dtype(operand_var.aval.dtype)
        # jnp.issubdtype also ranks the bfloat16 extension type as floating.
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(
                "reduce_window_min currently supports floating operand dtypes "
                f"only, got {dtype}"
            )
        ir_dtype = numpy_dtype_to_ir(dtype)

        operand = ctx.require_value_for_var(operand_var)
        out = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("reduce_window_min_out")
        )
        operand_shape: tuple[DimInput, ...] = tuple(operand_var.aval.shape)
        out_shape: tuple[DimInput, ...] = tuple(out_var.aval.shape)

        def work(name: str, shape: tuple[DimInput, ...]) -> ir.Value:
            value = ir.Value(
                name=ctx.fresh_name(f"reduce_window_min_{name}"),
                type=ir.TensorType(ir_dtype),
            )
            _stamp_type_and_shape(value, shape)
            return value

        # MaxPool reads its input as (batch, channel, *spatial).
        batched = ctx.builder.op(
            "Unsqueeze",
            [operand, _const_i64(ctx, np.asarray([0], dtype=np.int64), "rwmin_axes0")],
            output=work("unsq0", (1, *operand_shape)),
        )
        lifted = ctx.builder.op(
            "Unsqueeze",
            [batched, _const_i64(ctx, np.asarray([1], dtype=np.int64), "rwmin_axes1")],
            output=work("unsq1", (1, 1, *operand_shape)),
        )

        # min(x) = -max(-x). MaxPool leaves padded positions out of the maximum,
        # which matches JAX padding the window with the identity +inf.
        negated = ctx.builder.op(
            "Neg", [lifted], output=work("neg_in", (1, 1, *operand_shape))
        )
        pooled_shape: tuple[DimInput, ...] = (1, 1, *out_shape)
        attributes: dict[str, ir.Attr] = {
            "kernel_shape": ir.AttrInt64s("kernel_shape", window),
            "strides": ir.AttrInt64s("strides", strides),
        }
        pads = _flatten_padding(padding)
        if any(pads):
            attributes["pads"] = ir.AttrInt64s("pads", pads)
        if any(dilation != 1 for dilation in window_dilation):
            attributes["dilations"] = ir.AttrInt64s("dilations", window_dilation)
        pooled = ctx.builder.op(
            "MaxPool", [negated], attributes, output=work("pool", pooled_shape)
        )
        restored = ctx.builder.op("Neg", [pooled], output=work("neg_out", pooled_shape))

        result = ctx.builder.op(
            "Squeeze",
            [
                restored,
                _const_i64(ctx, np.asarray([0, 1], dtype=np.int64), "rwmin_squeeze"),
            ],
            output=out,
        )
        ctx.bind_value_for_var(out_var, result)
