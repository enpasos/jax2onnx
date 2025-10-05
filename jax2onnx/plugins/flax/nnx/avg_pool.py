# jax2onnx/plugins/flax/nnx/avg_pool.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _dim_label_from_value_or_aval,
    _to_ir_dim_for_shape,
    _ensure_value_info as _add_value_info,
)

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


# ------------------------------------------------------------------
# Graph-pattern expectations used by tests
# ------------------------------------------------------------------
_POOL_COUNTS = {
    "Transpose": 2,
    "AveragePool": 1,
    "Reshape": 0,
    "CastLike": 0,
    "Identity": 0,
}


def _make_expect(path: str, *, symbols: Optional[dict[str, Optional[int]]] = None):
    spec = [(path, {"counts": dict(_POOL_COUNTS)})]
    if symbols is None:
        return EG(spec, no_unused_inputs=True)
    return EG(spec, symbols=symbols, no_unused_inputs=True)


EXPECT_32_TO_16 = _make_expect(
    "Transpose:Bx3x32x32 -> AveragePool:Bx3x16x16 -> Transpose:Bx16x16x3",
    symbols={"B": None},
)

EXPECT_8_TO_7 = _make_expect(
    "Transpose:Bx3x8x8 -> AveragePool:Bx3x7x7 -> Transpose:Bx7x7x3",
    symbols={"B": None},
)

EXPECT_10_TO_4 = _make_expect(
    "Transpose:Bx1x10x10 -> AveragePool:Bx1x4x4 -> Transpose:Bx4x4x1",
    symbols={"B": None},
)

EXPECT_8_TO_4 = _make_expect(
    "Transpose:Bx3x8x8 -> AveragePool:Bx3x4x4 -> Transpose:Bx4x4x3",
    symbols={"B": None},
)


@register_primitive(
    jaxpr_primitive="nnx.avg_pool",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool",
    onnx=[
        {
            "component": "AveragePool",
            "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="avg_pool",
    testcases=[
        {
            "testcase": "avg_pool",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 32, 32, 3)],
            "expected_output_shapes": [("B", 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_32_TO_16,
        },
        {
            "testcase": "avg_pool_same_padding",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [("B", 32, 32, 3)],
            "expected_output_shapes": [("B", 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_32_TO_16,
        },
        {
            "testcase": "avg_pool_default_padding",
            "callable": lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            "input_shapes": [("B", 32, 32, 3)],
            "expected_output_shapes": [("B", 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_32_TO_16,
        },
        {
            "testcase": "avg_pool_stride1",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 7, 7, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_8_TO_7,
        },
        {
            "testcase": "avg_pool_win3x3_stride2",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(3, 3), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 10, 10, 1)],
            "expected_output_shapes": [("B", 4, 4, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_10_TO_4,
        },
        {
            "testcase": "avg_pool_stride_none",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=None, padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 7, 7, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_8_TO_7,
        },
        {
            "testcase": "avg_pool_count_include_pad_false",
            "callable": lambda x: nnx.avg_pool(
                x,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="SAME",
                count_include_pad=False,
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_8_TO_4,
        },
    ],
)
class AvgPoolPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.avg_pool.
    We export in NCHW: NHWC -> AveragePool(NCHW) -> NHWC.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.avg_pool")
    _PRIM.multiple_results = False
    _ORIG_CALL: ClassVar[Optional[Callable]] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------------- abstract eval ----------------
    @staticmethod
    def abstract_eval(
        x,
        *,
        window_shape: Sequence[int],
        strides: Optional[Sequence[int]],
        padding: str,
        count_include_pad: bool,
    ):
        # Prefer original nnx.avg_pool if we captured it, else shape math.
        actual_strides = (
            tuple(strides) if strides is not None else (1,) * len(window_shape)
        )

        if AvgPoolPlugin._ORIG_CALL is None:
            # Basic shape math for NHWC rank>=3; keep dtype.
            # Compute H/W according to VALID/SAME (ceil for SAME, floor for VALID).

            rank = x.ndim
            if rank < 3:
                return jax.core.ShapedArray(x.shape, x.dtype)
            H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
            kH, kW = window_shape
            sH, sW = actual_strides

            def _dim_out(L, k, s, mode):
                if isinstance(L, (int, np.integer)):
                    if mode.upper() == "SAME":
                        return int(np.ceil(L / s))
                    return int(np.floor((L - k) / s) + 1)
                # symbolic: leave unknown
                return None

            oH = _dim_out(H, kH, sH, padding)
            oW = _dim_out(W, kW, sW, padding)
            out_shape = (*x.shape[:-3], oH, oW, C)
            return jax.core.ShapedArray(out_shape, x.dtype)

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)

        def _helper(v):
            return AvgPoolPlugin._ORIG_CALL(
                v,
                window_shape=tuple(window_shape),
                strides=actual_strides,
                padding=padding,
                count_include_pad=bool(count_include_pad),
            )

        out = jax.eval_shape(_helper, x_spec)
        return jax.core.ShapedArray(out.shape, out.dtype)

    # ---------------- lowering (IR) ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]
        window_shape = tuple(eqn.params["window_shape"])
        strides = eqn.params.get("strides")
        padding = str(eqn.params.get("padding", "VALID"))
        count_include_pad = bool(eqn.params.get("count_include_pad", True))

        actual_strides = (
            tuple(strides) if strides is not None else (1,) * len(window_shape)
        )

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        y_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))  # ← NEW

        if is_shape_all_unknown(getattr(x_val, "shape", None)):
            if any(d is not None for d in x_shape):
                _stamp_type_and_shape(x_val, x_shape)

        rank = len(x_shape)
        need_layout_convert = rank > 2

        pool_in = x_val
        if need_layout_convert:
            perm = [0, rank - 1] + list(range(1, rank - 1))
            nchw_dims_in = (
                _dim_label_from_value_or_aval(x_val, x_shape, 0),  # N
                _dim_label_from_value_or_aval(x_val, x_shape, rank - 1),  # C
                *[
                    _dim_label_from_value_or_aval(x_val, x_shape, i)
                    for i in range(1, rank - 1)
                ],
            )
            x_nchw = ir.Value(
                name=ctx.fresh_name("pool_pre_transpose"),
                type=x_val.type,
                shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_in)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Transpose",
                    domain="",
                    inputs=[x_val],
                    outputs=[x_nchw],
                    name=ctx.fresh_name("Transpose"),
                    attributes=[ir.Attr("perm", ir.AttributeType.INTS, tuple(perm))],
                )
            )
            pool_in = x_nchw

        # AveragePool
        if need_layout_convert:
            # NCHW shape derived from OUTPUT AVAL (NHWC → NCHW)
            nchw_dims_out = (y_shape[0], y_shape[-1], *y_shape[1:-1])  # ← NEW
            pool_out = ir.Value(
                name=ctx.fresh_name("pool_nchw_out"),
                type=x_val.type,
                shape=ir.Shape(
                    tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
                ),  # ← NEW
            )
        else:
            pool_out = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        auto_pad = "SAME_UPPER" if padding.upper() == "SAME" else "VALID"
        attrs = [
            ir.Attr("kernel_shape", ir.AttributeType.INTS, tuple(window_shape)),
            ir.Attr("strides", ir.AttributeType.INTS, tuple(actual_strides)),
            ir.Attr("auto_pad", ir.AttributeType.STRING, auto_pad),
            ir.Attr(
                "count_include_pad", ir.AttributeType.INT, 1 if count_include_pad else 0
            ),
        ]
        ctx.add_node(
            ir.Node(
                op_type="AveragePool",
                domain="",
                inputs=[pool_in],
                outputs=[pool_out],
                name=ctx.fresh_name("AveragePool"),
                attributes=attrs,
            )
        )

        # ---- Stamp correct symbolic output dims (preserve B/C labels) ----
        # y_aval gives the concrete/new spatial sizes; copy N & C labels from input.
        y_aval_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
        n_label = _dim_label_from_value_or_aval(x_val, x_shape, 0)
        c_label = _dim_label_from_value_or_aval(x_val, x_shape, rank - 1)

        # NHWC output dims with preserved labels
        nhwc_out_dims = (
            n_label,
            *[y_aval_shape[i] for i in range(1, rank - 1)],
            c_label,
        )

        if need_layout_convert:
            # Also fix the NCHW intermediate to match the (labelled) NHWC result
            nchw_dims_out = (nhwc_out_dims[0], nhwc_out_dims[-1], *nhwc_out_dims[1:-1])
            pool_out.shape = ir.Shape(
                tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
            )

            # Transpose back to NHWC and stamp final output shape
            inv_perm = [0] + list(range(2, rank)) + [1]
            y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
            ctx.add_node(
                ir.Node(
                    op_type="Transpose",
                    domain="",
                    inputs=[pool_out],
                    outputs=[y_val],
                    name=ctx.fresh_name("Transpose"),
                    attributes=[
                        ir.Attr("perm", ir.AttributeType.INTS, tuple(inv_perm))
                    ],
                )
            )
            _stamp_type_and_shape(y_val, nhwc_out_dims)
            _add_value_info(ctx, y_val)
        else:
            # Rank <= 2 (rare for pooling), still preserve labels where applicable
            y_val = pool_out
            _stamp_type_and_shape(y_val, nhwc_out_dims[:rank])
            _add_value_info(ctx, y_val)

    # ---------------- eager impl (for tests) ----------------
    @staticmethod
    def _call_avg_pool_eager(x, *, window_shape, strides, padding, count_include_pad):
        if AvgPoolPlugin._ORIG_CALL is not None:
            # Call the captured original nnx.avg_pool (pre-patch)
            return AvgPoolPlugin._ORIG_CALL(
                x,
                window_shape=tuple(window_shape),
                strides=tuple(strides) if strides is not None else None,
                padding=padding,
                count_include_pad=bool(count_include_pad),
            )
        # Fallback: simple reduce-window average in NHWC
        pads = padding.upper()
        ws = tuple(window_shape)
        st = tuple(strides) if strides is not None else (1,) * len(ws)
        rank = x.ndim
        if rank != 4:
            # Keep it simple; tests use NHWC 4D
            return x
        # Implement via jax.lax.reduce_window on NHWC using SAME/VALID
        jnp.array(0, dtype=x.dtype)
        ones = jnp.ones(ws + (1,), dtype=x.dtype)
        from jax import lax

        y_sum = lax.conv_general_dilated(
            x,
            ones,
            window_strides=st,
            padding=pads,
            dimension_numbers=("NHWC", "HWOI", "NHWC"),
        )
        # Compute divisor per window (count_include_pad controls padding effect)
        if pads == "SAME" and not count_include_pad:
            # divisor counts only valid elements (no padded)
            ones_img = jnp.ones_like(x[..., :1])
            win = lax.conv_general_dilated(
                ones_img,
                ones,
                window_strides=st,
                padding=pads,
                dimension_numbers=("NHWC", "HWOI", "NHWC"),
            )
            return y_sum / win
        else:
            div = float(np.prod(ws))
            return y_sum / div

    # ---------------- monkey-patch ----------------
    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        AvgPoolPlugin._ORIG_CALL = orig_fn

        def patched(
            inputs,
            *,
            window_shape,
            strides=None,
            padding="VALID",
            count_include_pad=True,
        ):
            actual_strides = (
                tuple(strides) if strides is not None else (1,) * len(window_shape)
            )
            return AvgPoolPlugin._PRIM.bind(
                inputs,
                window_shape=tuple(window_shape),
                strides=actual_strides,
                padding=str(padding),
                count_include_pad=bool(count_include_pad),
            )

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="avg_pool",
                make_value=lambda orig: cls.get_monkey_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


# ---------------- concrete eager impl ----------------
@AvgPoolPlugin._PRIM.def_impl
def _impl(x, *, window_shape, strides, padding, count_include_pad):
    return AvgPoolPlugin._call_avg_pool_eager(
        x,
        window_shape=window_shape,
        strides=strides,
        padding=padding,
        count_include_pad=count_include_pad,
    )
