# file: jax2onnx/plugins2/flax/nnx/max_pool.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.extend.core import Primitive

import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _to_ir_dim_for_shape,
    _ensure_value_info as _add_value_info,
)
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


EXPECT_T_MP_T = EG(
    [
        (
            "Transpose -> MaxPool -> Transpose",
            {
                "counts": {
                    "Transpose": 2,
                    "MaxPool": 1,
                    "Reshape": 0,
                }
            },
        )
    ]
)


MAX_POOL_PRIM = Primitive("nnx.max_pool")
MAX_POOL_PRIM.multiple_results = False


def _set_attrs(ctx: Any, node: ir.Node, attrs: dict[str, object]) -> None:
    setter = getattr(ctx, "set_node_attrs", None)
    if callable(setter):
        setter(node, attrs)


@register_primitive(
    jaxpr_primitive=MAX_POOL_PRIM.name,
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.nnx",
    component="max_pool",
    testcases=[
        {
            "testcase": "max_pool",
            "callable": lambda x: nnx.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [(1, 32, 32, 3)],
            "expected_output_shapes": [(1, 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
        {
            "testcase": "max_pool_same_padding",
            "callable": lambda x: nnx.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [(1, 32, 32, 3)],
            "expected_output_shapes": [(1, 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
        {
            "testcase": "max_pool_basic",
            "callable": lambda x: nnx.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [(1, 8, 8, 3)],
            "expected_output_shapes": [(1, 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_MP_T,
        },
        {
            "testcase": "max_pool_same",
            "callable": lambda x: nnx.max_pool(
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
    """IR-only plugin for flax.nnx.max_pool."""

    _PRIM: ClassVar[Primitive] = MAX_POOL_PRIM
    _ORIG_CALL: ClassVar[Optional[Callable]] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------------- helpers ----------------
    @staticmethod
    def _normalize_stride(
        strides: Optional[Sequence[int]], window_shape: Sequence[int]
    ) -> Tuple[int, ...]:
        if strides is None:
            return tuple(int(s) for s in window_shape)
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

        pool_in = x_val
        if need_layout_convert:
            pool_in = ir.Value(
                name=ctx.fresh_name("mp_nchw_in"),
                type=x_val.type,
                shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_in)),
            )
            transpose_in = ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[x_val],
                outputs=[pool_in],
                name=ctx.fresh_name("Transpose"),
            )
            ctx.add_node(transpose_in)
            _set_attrs(ctx, transpose_in, {"perm": tuple(perm)})
            _stamp_type_and_shape(pool_in, nchw_dims_in)

        if need_layout_convert:
            nchw_out_dims = (
                _label(0),
                _label(rank - 1),
                *y_aval_shape[1:-1],
            )
            pool_out = ir.Value(
                name=ctx.fresh_name("mp_nchw_out"),
                type=pool_in.type,
                shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in nchw_out_dims)),
            )
        else:
            pool_out = y_val

        node = ir.Node(
            op_type="MaxPool",
            domain="",
            inputs=[pool_in],
            outputs=[pool_out],
            name=ctx.fresh_name("MaxPool"),
        )
        ctx.add_node(node)
        _set_attrs(
            ctx,
            node,
            {
                "kernel_shape": tuple(int(v) for v in window_shape),
                "strides": tuple(int(v) for v in strides),
                "auto_pad": "SAME_UPPER" if padding.upper() == "SAME" else "VALID",
            },
        )

        if need_layout_convert:
            _stamp_type_and_shape(pool_out, nchw_out_dims)

        if need_layout_convert:
            inv_perm_tuple = tuple(inv_perm)
            transpose_out = ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[pool_out],
                outputs=[y_val],
                name=ctx.fresh_name("Transpose"),
            )
            ctx.add_node(transpose_out)
            _set_attrs(ctx, transpose_out, {"perm": inv_perm_tuple})

        n_label = _label(0) if rank else None
        c_label = _label(rank - 1) if rank else None
        if rank <= 2:
            nhwc_dims = tuple(y_aval_shape)
        else:
            middle_dims = y_aval_shape[1:-1]
            nhwc_dims = (n_label, *middle_dims, c_label)
        _stamp_type_and_shape(y_val, nhwc_dims)
        _add_value_info(ctx, y_val)

    # ---------------- monkey patch & binding ----------------
    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="max_pool",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            )
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
        # Basic fallback for NHWC rank-4 tensors
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
