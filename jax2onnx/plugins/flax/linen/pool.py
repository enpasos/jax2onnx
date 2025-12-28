# jax2onnx/plugins/flax/linen/pool.py

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax.linen import pooling as linen_pooling
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _to_ir_dim_for_shape,
    _ensure_value_metadata,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


POOL_PRIM: Primitive = Primitive("linen.pool")
POOL_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=POOL_PRIM.name,
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.pool",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        },
        {
            "component": "AveragePool",
            "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="pool",
    testcases=[
        {
            "testcase": "pool_max_basic",
            "callable": lambda x: linen_pooling.pool(
                x,
                -jnp.inf,
                jax.lax.max,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "pool_min_basic",
            "callable": lambda x: linen_pooling.pool(
                x,
                jnp.inf,
                jax.lax.min,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "pool_sum_basic",
            "callable": lambda x: linen_pooling.pool(
                x,
                0.0,
                jax.lax.add,
                window_shape=(2, 2),
                strides=(1, 1),
                padding="VALID",
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 7, 7, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class PoolPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.linen.pool (supports add/max/min reductions)."""

    _PRIM: ClassVar[Primitive] = POOL_PRIM
    _ORIG_CALL: ClassVar[Optional[Callable]] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _REDUCE_FN_MAP: ClassVar[dict[str, Callable]] = {
        "add": jax.lax.add,
        "max": jax.lax.max,
        "min": jax.lax.min,
    }

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

    @classmethod
    def _reduce_fn_key(cls, reduce_fn: object) -> str:
        if isinstance(reduce_fn, str):
            key = reduce_fn.lower()
            if key in cls._REDUCE_FN_MAP:
                return key
        if reduce_fn is jax.lax.add:
            return "add"
        if reduce_fn is jax.lax.max:
            return "max"
        if reduce_fn is jax.lax.min:
            return "min"
        raise NotImplementedError(
            "linen.pool only supports reduce_fn in {lax.add, lax.max, lax.min}."
        )

    @classmethod
    def _reduce_fn_from_key(cls, reduce_fn: object) -> Callable:
        key = cls._reduce_fn_key(reduce_fn)
        return cls._REDUCE_FN_MAP[key]

    @staticmethod
    def _init_scalar(init: object) -> object:
        try:
            arr = np.asarray(init)
        except Exception:
            return init
        if arr.size == 1:
            return arr.reshape(()).item()
        return init

    @staticmethod
    def _is_zero(value: object) -> bool:
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        if arr.size == 0:
            return False
        return bool(np.all(arr == 0))

    @staticmethod
    def _is_pos_inf(value: object) -> bool:
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        if arr.size == 0:
            return False
        return bool(np.all(np.isposinf(arr)))

    @staticmethod
    def _is_neg_inf(value: object) -> bool:
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        if arr.size == 0:
            return False
        return bool(np.all(np.isneginf(arr)))

    @classmethod
    def _validate_init(cls, reduce_key: str, init: object) -> None:
        if reduce_key == "add" and not cls._is_zero(init):
            raise ValueError("linen.pool(add) requires init == 0.")
        if reduce_key == "max" and not cls._is_neg_inf(init):
            raise ValueError("linen.pool(max) requires init == -inf.")
        if reduce_key == "min" and not cls._is_pos_inf(init):
            raise ValueError("linen.pool(min) requires init == +inf.")

    # ---------------- abstract eval ----------------
    @staticmethod
    def abstract_eval(x, *, init, reduce_fn, window_shape, strides, padding):
        strides = PoolPlugin._normalize_stride(strides, window_shape)
        if not isinstance(padding, str):
            raise NotImplementedError(
                "linen.pool only supports padding='SAME' or 'VALID'."
            )
        shape = list(x.shape)
        if len(shape) <= 2:
            return jax.core.ShapedArray(tuple(shape), x.dtype)
        spatial = shape[1:-1]
        out_spatial = [
            PoolPlugin._compute_output_dim(dim, w, s, padding)
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
            raise ValueError("linen.pool requires a non-empty window_shape")
        strides = PoolPlugin._normalize_stride(params.get("strides"), window_shape)
        padding = params.get("padding", "VALID")
        if not isinstance(padding, str):
            raise NotImplementedError(
                "linen.pool only supports padding='SAME' or 'VALID'."
            )
        reduce_key = PoolPlugin._reduce_fn_key(params.get("reduce_fn"))
        init = PoolPlugin._init_scalar(params.get("init"))
        PoolPlugin._validate_init(reduce_key, init)

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
            raise AttributeError("IR build context missing builder for Pool lowering")

        pool_in = x_val
        if need_layout_convert:
            pool_in = builder.Transpose(
                x_val,
                _outputs=[ctx.fresh_name("pool_nchw_in")],
                perm=tuple(perm),
            )
            pool_in.type = x_val.type
            _stamp_type_and_shape(
                pool_in, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_in)
            )
            _ensure_value_metadata(ctx, pool_in)

        pool_source = pool_in
        if reduce_key == "min":
            pool_source = builder.Neg(
                pool_in,
                _outputs=[ctx.fresh_name("pool_neg_in")],
            )
            if getattr(pool_in, "type", None) is not None:
                pool_source.type = pool_in.type
            _stamp_type_and_shape(
                pool_source, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_in)
            )
            _ensure_value_metadata(ctx, pool_source)

        if reduce_key in {"max", "min"}:
            pool_result = builder.MaxPool(
                pool_source,
                _outputs=[ctx.fresh_name("MaxPool")],
                kernel_shape=tuple(int(v) for v in window_shape),
                strides=tuple(int(v) for v in strides),
                auto_pad="SAME_UPPER" if padding.upper() == "SAME" else "VALID",
            )
        else:
            pool_result = builder.AveragePool(
                pool_source,
                _outputs=[ctx.fresh_name("AveragePool")],
                kernel_shape=tuple(int(v) for v in window_shape),
                strides=tuple(int(v) for v in strides),
                auto_pad="SAME_UPPER" if padding.upper() == "SAME" else "VALID",
                count_include_pad=1,
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

        nchw_dims_out = None
        if need_layout_convert:
            nchw_dims_out = (nhwc_dims[0], nhwc_dims[-1], *nhwc_dims[1:-1])
            _stamp_type_and_shape(
                pool_result, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
            )
            _ensure_value_metadata(ctx, pool_result)

        if reduce_key == "min":
            neg_out = builder.Neg(
                pool_result,
                _outputs=[ctx.fresh_name("pool_neg_out")],
            )
            if dtype is not None:
                neg_out.type = ir.TensorType(dtype)
            if nchw_dims_out is not None:
                _stamp_type_and_shape(
                    neg_out, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
                )
            _ensure_value_metadata(ctx, neg_out)
            pool_result = neg_out

        if reduce_key == "add":
            aval_dtype = getattr(getattr(x_var, "aval", None), "dtype", None)
            if aval_dtype is None:
                raise TypeError("linen.pool(add) requires a known input dtype.")
            np_dtype = np.dtype(aval_dtype)
            if not np.issubdtype(np_dtype, np.floating):
                raise TypeError(
                    "linen.pool(add) currently supports floating-point inputs only."
                )
            window_area = int(np.prod(window_shape))
            if window_area != 1:
                scale_val = builder.add_initializer_from_scalar(
                    name=ctx.fresh_name("pool_sum_scale"),
                    value=np.asarray(window_area, dtype=np_dtype),
                )
                _stamp_type_and_shape(scale_val, ())
                _ensure_value_metadata(ctx, scale_val)
                scaled = builder.Mul(
                    pool_result,
                    scale_val,
                    _outputs=[ctx.fresh_name("pool_sum")],
                )
                if dtype is not None:
                    scaled.type = ir.TensorType(dtype)
                if nchw_dims_out is not None:
                    _stamp_type_and_shape(
                        scaled, tuple(_to_ir_dim_for_shape(d) for d in nchw_dims_out)
                    )
                _ensure_value_metadata(ctx, scaled)
                pool_result = scaled

        if need_layout_convert:
            final = builder.Transpose(
                pool_result,
                _outputs=[
                    getattr(y_val, "name", None) or ctx.fresh_name("pool_transpose")
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
    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        PoolPlugin._ORIG_CALL = orig_fn

        def patched_pool(inputs, init, reduce_fn, window_shape, strides, padding):
            if not isinstance(padding, str):
                raise NotImplementedError(
                    "linen.pool only supports padding='SAME' or 'VALID'."
                )
            reduce_key = PoolPlugin._reduce_fn_key(reduce_fn)
            init_val = PoolPlugin._init_scalar(init)
            PoolPlugin._validate_init(reduce_key, init_val)
            actual_strides = PoolPlugin._normalize_stride(strides, window_shape)
            return PoolPlugin._PRIM.bind(
                inputs,
                init=init_val,
                reduce_fn=reduce_key,
                window_shape=tuple(int(v) for v in window_shape),
                strides=tuple(int(s) for s in actual_strides),
                padding=str(padding),
            )

        return patched_pool

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.pooling",
                attr="pool",
                make_value=lambda orig: cls.get_monkey_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="pool",
                make_value=lambda orig: cls.get_monkey_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    # ---------------- eager impl ----------------
    @staticmethod
    def _call_pool_eager(x, *, init, reduce_fn, window_shape, strides, padding):
        reduce_callable = PoolPlugin._reduce_fn_from_key(reduce_fn)
        if PoolPlugin._ORIG_CALL is not None:
            return PoolPlugin._ORIG_CALL(
                x,
                init,
                reduce_callable,
                window_shape,
                strides,
                padding,
            )

        num_batch_dims = x.ndim - (len(window_shape) + 1)
        strides = strides or (1,) * len(window_shape)
        strides = (1,) * num_batch_dims + tuple(strides) + (1,)
        dims = (1,) * num_batch_dims + tuple(window_shape) + (1,)

        is_single_input = False
        if num_batch_dims == 0:
            x = x[None]
            strides = (1,) + strides
            dims = (1,) + dims
            is_single_input = True

        if not isinstance(padding, str):
            padding = tuple(map(tuple, padding))
            padding = ((0, 0),) + padding + ((0, 0),)

        y = jax.lax.reduce_window(x, init, reduce_callable, dims, strides, padding)
        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        return y


@PoolPlugin._PRIM.def_impl
def _impl_pool(x, *, init, reduce_fn, window_shape, strides, padding):
    return PoolPlugin._call_pool_eager(
        x,
        init=init,
        reduce_fn=reduce_fn,
        window_shape=tuple(int(v) for v in window_shape),
        strides=tuple(int(v) for v in strides) if strides is not None else None,
        padding=padding,
    )
