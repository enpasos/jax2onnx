# jax2onnx/plugins/flax/linen/max_pool.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, Final, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.extend.core import Primitive

import onnx_ir as ir
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import max_pool as nnx_max_pool
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _to_ir_dim_for_shape,
    _ensure_value_metadata,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


EXPECT_T_MP_T: Final = nnx_max_pool.EXPECT_T_MP_T


MAX_POOL_PRIM: Final[Primitive] = Primitive("linen.max_pool")
MAX_POOL_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=MAX_POOL_PRIM.name,
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        }
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="max_pool",
    testcases=[
        {
            "testcase": "max_pool",
            "callable": lambda x: nn.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [(1, 32, 32, 3)],
            "expected_output_shapes": [(1, 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
        {
            "testcase": "max_pool_same_padding",
            "callable": lambda x: nn.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [(1, 32, 32, 3)],
            "expected_output_shapes": [(1, 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
        {
            "testcase": "max_pool_basic",
            "callable": lambda x: nn.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [(1, 8, 8, 3)],
            "expected_output_shapes": [(1, 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
        {
            "testcase": "max_pool_same",
            "callable": lambda x: nn.max_pool(
                x, window_shape=(3, 3), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [("B", 10, 10, 3)],
            "expected_output_shapes": [("B", 5, 5, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
    ],
)
class MaxPoolPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.linen.max_pool."""

    _PRIM: ClassVar[Primitive] = MAX_POOL_PRIM
    _ORIG_CALL: ClassVar[Optional[Callable]] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------------- helpers ----------------
    @staticmethod
    def _normalize_stride(
        strides: Optional[Sequence[int]], window_shape: Sequence[int]
    ) -> Tuple[int, ...]:
        if strides is None:
            return tuple(1 for _ in window_shape)
        return tuple(int(s) for s in strides)

    @staticmethod
    def _compute_output_dim(length, window, stride, padding: str):
        if isinstance(length, (int, np.integer)):
            if padding.upper() == "SAME":
                return int(np.ceil(length / stride))
            return int(np.floor((length - window) / stride) + 1)
        return length

    @staticmethod
    def abstract_eval(x, *, window_shape, strides, padding):
        strides = MaxPoolPlugin._normalize_stride(strides, window_shape)
        padding = str(padding)
        shape = list(x.shape)
        if len(shape) <= 2:
            return jax.core.ShapedArray(tuple(shape), x.dtype)
        spatial = shape[1:-1]
        out_spatial = [
            MaxPoolPlugin._compute_output_dim(dim, w, s, padding)
            for dim, w, s in zip(spatial, window_shape, strides, strict=False)
        ]
        out_shape = (shape[0], *out_spatial, shape[-1])
        return jax.core.ShapedArray(tuple(out_shape), x.dtype)

    # ---------------- lowering ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars[:1]
        (y_var,) = eqn.outvars[:1]

        params = dict(getattr(eqn, "params", {}) or {})
        window_shape = tuple(int(v) for v in params.get("window_shape", ()))
        if not window_shape:
            raise ValueError("max_pool requires a non-empty window_shape")
        strides = MaxPoolPlugin._normalize_stride(params.get("strides"), window_shape)
        padding = str(params.get("padding", "VALID"))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        y_aval_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
        rank = len(x_shape)

        need_layout_convert = rank > 2
        if need_layout_convert:
            perm = [0, rank - 1] + list(range(1, rank - 1))
            inv_perm = [perm.index(i) for i in range(rank)]
        else:
            perm = list(range(rank))
            inv_perm = perm

        def _label(idx: int):
            return _dim_label_from_value_or_aval(x_val, x_shape, idx)

        nchw_dims_in = tuple(_label(i) for i in perm)

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for MaxPool lowering"
            )

        pool_in = x_val
        if need_layout_convert:
            pool_in = builder.Transpose(
                x_val,
                _outputs=[ctx.fresh_name("mp_nchw_in")],
                perm=tuple(perm),
            )
            pool_in.type = x_val.type
            _stamp_type_and_shape(
                pool_in, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_in)
            )
            _ensure_value_metadata(ctx, pool_in)

        pool_result = builder.MaxPool(
            pool_in,
            _outputs=[ctx.fresh_name("MaxPool")],
            kernel_shape=tuple(int(v) for v in window_shape),
            strides=tuple(int(v) for v in strides),
            auto_pad="SAME_UPPER" if padding.upper() == "SAME" else "VALID",
        )

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            pool_result.type = ir.TensorType(dtype)

        n_label = _label(0) if rank else None
        c_label = _label(rank - 1) if rank else None
        if rank <= 2:
            nhwc_dims = tuple(y_aval_shape)
        else:
            nhwc_dims = (n_label, *y_aval_shape[1:-1], c_label)

        if need_layout_convert:
            nchw_dims_out = (nhwc_dims[0], nhwc_dims[-1], *nhwc_dims[1:-1])
            _stamp_type_and_shape(
                pool_result, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
            )
            _ensure_value_metadata(ctx, pool_result)

            final = builder.Transpose(
                pool_result,
                _outputs=[
                    getattr(y_val, "name", None) or ctx.fresh_name("maxpool_transpose")
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

    # ---------------- monkey patch & binding ----------------
    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.pooling",
                attr="max_pool",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="max_pool",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def _make_patch(cls, orig):
        cls._ORIG_CALL = orig

        def patched_max_pool(x, *, window_shape, strides=None, padding="VALID"):
            strides_tuple = cls._normalize_stride(strides, window_shape)
            return cls._PRIM.bind(
                x,
                window_shape=tuple(int(v) for v in window_shape),
                strides=strides_tuple,
                padding=str(padding),
            )

        return patched_max_pool

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    # ---------------- eager impl ----------------
    @staticmethod
    def _call_max_pool_eager(x, *, window_shape, strides, padding):
        if MaxPoolPlugin._ORIG_CALL is not None:
            return MaxPoolPlugin._ORIG_CALL(
                x,
                window_shape=window_shape,
                strides=strides,
                padding=padding,
            )
        if x.ndim == 4:
            pads = padding.upper()
            ws = tuple(window_shape)
            st = tuple(strides)
            x_nchw = jnp.transpose(x, (0, 3, 1, 2))
            y = jax.lax.reduce_window(
                x_nchw,
                -jnp.inf,
                jax.lax.max,
                (1, 1, *ws),
                (1, 1, *st),
                pads,
            )
            return jnp.transpose(y, (0, 2, 3, 1))
        return x


@MaxPoolPlugin._PRIM.def_impl
def _impl_max_pool(x, *, window_shape, strides, padding):
    return MaxPoolPlugin._call_max_pool_eager(
        x,
        window_shape=tuple(int(v) for v in window_shape),
        strides=tuple(int(v) for v in strides),
        padding=str(padding),
    )
