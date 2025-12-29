# jax2onnx/plugins/flax/linen/avg_pool.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, Final, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import avg_pool as nnx_avg_pool
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _dim_label_from_value_or_aval,
    _to_ir_dim_for_shape,
    _ensure_value_metadata,
)

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


EXPECT_32_TO_16: Final = nnx_avg_pool.EXPECT_32_TO_16
EXPECT_8_TO_7: Final = nnx_avg_pool.EXPECT_8_TO_7
EXPECT_10_TO_4: Final = nnx_avg_pool.EXPECT_10_TO_4
EXPECT_8_TO_4: Final = nnx_avg_pool.EXPECT_8_TO_4


@register_primitive(
    jaxpr_primitive="linen.avg_pool",
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
    since="v0.11.0",
    context="primitives.linen",
    component="avg_pool",
    testcases=[
        {
            "testcase": "avg_pool",
            "callable": lambda x: nn.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 32, 32, 3)],
            "expected_output_shapes": [("B", 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_32_TO_16,
        },
        {
            "testcase": "avg_pool_same_padding",
            "callable": lambda x: nn.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [("B", 32, 32, 3)],
            "expected_output_shapes": [("B", 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_32_TO_16,
        },
        {
            "testcase": "avg_pool_default_padding",
            "callable": lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            "input_shapes": [("B", 32, 32, 3)],
            "expected_output_shapes": [("B", 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_32_TO_16,
        },
        {
            "testcase": "avg_pool_stride1",
            "callable": lambda x: nn.avg_pool(
                x, window_shape=(2, 2), strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 7, 7, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_8_TO_7,
        },
        {
            "testcase": "avg_pool_win3x3_stride2",
            "callable": lambda x: nn.avg_pool(
                x, window_shape=(3, 3), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 10, 10, 1)],
            "expected_output_shapes": [("B", 4, 4, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_10_TO_4,
        },
        {
            "testcase": "avg_pool_stride_none",
            "callable": lambda x: nn.avg_pool(
                x, window_shape=(2, 2), strides=None, padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 7, 7, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_8_TO_7,
        },
        {
            "testcase": "avg_pool_count_include_pad_false",
            "callable": lambda x: nn.avg_pool(
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
    IR-only plugin for flax.linen.avg_pool.
    We export in NCHW: NHWC -> AveragePool(NCHW) -> NHWC.
    """

    _PRIM: ClassVar[Primitive] = Primitive("linen.avg_pool")
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
        actual_strides = (
            tuple(strides) if strides is not None else (1,) * len(window_shape)
        )

        if AvgPoolPlugin._ORIG_CALL is None:
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
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        y_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))

        if is_shape_all_unknown(getattr(x_val, "shape", None)) and any(
            d is not None for d in x_shape
        ):
            _stamp_type_and_shape(x_val, x_shape)

        rank = len(x_shape)
        need_layout_convert = rank > 2
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for AvgPool lowering"
            )

        def _label(idx: int):
            return _dim_label_from_value_or_aval(x_val, x_shape, idx)

        pool_in = x_val
        perm = list(range(rank))
        inv_perm = perm
        if need_layout_convert:
            perm = [0, rank - 1] + list(range(1, rank - 1))
            inv_perm = [perm.index(i) for i in range(rank)]
            nchw_dims_in = (
                _label(0),
                _label(rank - 1),
                *[_label(i) for i in range(1, rank - 1)],
            )
            pool_in = builder.Transpose(
                x_val,
                _outputs=[ctx.fresh_name("avgpool_nchw_in")],
                perm=tuple(perm),
            )
            pool_in.type = x_val.type
            _stamp_type_and_shape(
                pool_in, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_in)
            )
            _ensure_value_metadata(ctx, pool_in)

        pool_result = builder.AveragePool(
            pool_in,
            _outputs=[ctx.fresh_name("AveragePool")],
            kernel_shape=tuple(int(v) for v in window_shape),
            strides=tuple(int(v) for v in actual_strides),
            auto_pad="SAME_UPPER" if padding.upper() == "SAME" else "VALID",
            count_include_pad=1 if count_include_pad else 0,
        )

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            pool_result.type = ir.TensorType(dtype)

        n_label = _label(0) if rank else None
        c_label = _label(rank - 1) if rank else None
        if rank <= 2:
            nhwc_dims = tuple(y_shape)
        else:
            nhwc_dims = (n_label, *y_shape[1:-1], c_label)

        if need_layout_convert:
            nchw_dims_out = (nhwc_dims[0], nhwc_dims[-1], *nhwc_dims[1:-1])
            _stamp_type_and_shape(
                pool_result, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
            )
            _ensure_value_metadata(ctx, pool_result)

            final = builder.Transpose(
                pool_result,
                _outputs=[
                    getattr(y_val, "name", None) or ctx.fresh_name("avgpool_transpose")
                ],
                perm=tuple(inv_perm),
            )
            if dtype is not None:
                final.type = ir.TensorType(dtype)
            _stamp_type_and_shape(final, nhwc_dims)
            _ensure_value_metadata(ctx, final)
        else:
            final = pool_result
            _stamp_type_and_shape(final, nhwc_dims[:rank])
            _ensure_value_metadata(ctx, final)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if callable(bind_value):
            bind_value(y_var, final)
        else:
            raise AttributeError("IR build context missing bind_value_for_var")

    # ---------------- eager impl (for tests) ----------------
    @staticmethod
    def _call_avg_pool_eager(x, *, window_shape, strides, padding, count_include_pad):
        if AvgPoolPlugin._ORIG_CALL is not None:
            return AvgPoolPlugin._ORIG_CALL(
                x,
                window_shape=tuple(window_shape),
                strides=tuple(strides) if strides is not None else None,
                padding=padding,
                count_include_pad=bool(count_include_pad),
            )
        pads = padding.upper()
        ws = tuple(window_shape)
        st = tuple(strides) if strides is not None else (1,) * len(ws)
        rank = x.ndim
        if rank != 4:
            return x
        ones = jnp.ones(ws + (1,), dtype=x.dtype)
        from jax import lax

        y_sum = lax.conv_general_dilated(
            x,
            ones,
            window_strides=st,
            padding=pads,
            dimension_numbers=("NHWC", "HWOI", "NHWC"),
        )
        if pads == "SAME" and not count_include_pad:
            ones_img = jnp.ones_like(x[..., :1])
            win = lax.conv_general_dilated(
                ones_img,
                ones,
                window_strides=st,
                padding=pads,
                dimension_numbers=("NHWC", "HWOI", "NHWC"),
            )
            return y_sum / win
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
                target="flax.linen.pooling",
                attr="avg_pool",
                make_value=lambda orig: cls.get_monkey_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
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


@AvgPoolPlugin._PRIM.def_impl
def _impl(x, *, window_shape, strides, padding, count_include_pad):
    return AvgPoolPlugin._call_avg_pool_eager(
        x,
        window_shape=window_shape,
        strides=strides,
        padding=padding,
        count_include_pad=count_include_pad,
    )
