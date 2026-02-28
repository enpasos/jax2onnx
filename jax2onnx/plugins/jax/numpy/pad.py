# jax2onnx/plugins/jax/numpy/pad.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Final, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


PadPair = tuple[int, int]
PadWidth = tuple[PadPair, ...]


_PAD_PRIM: Final = make_jnp_primitive("jax.numpy.pad")


def _is_int_scalar(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _normalize_axis_pad(item: Any) -> PadPair:
    if _is_int_scalar(item):
        v = int(item)
        if v < 0:
            raise ValueError("negative pad widths are not supported")
        return (v, v)

    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        vals = tuple(item)
        if len(vals) != 2 or not all(_is_int_scalar(v) for v in vals):
            raise ValueError("each axis pad must be int or (int, int)")
        lo, hi = int(vals[0]), int(vals[1])
        if lo < 0 or hi < 0:
            raise ValueError("negative pad widths are not supported")
        return (lo, hi)

    raise ValueError("invalid axis pad specification")


def _normalize_pad_width(pad_width: Any, rank: int) -> PadWidth:
    if _is_int_scalar(pad_width):
        v = int(pad_width)
        if v < 0:
            raise ValueError("negative pad widths are not supported")
        return tuple((v, v) for _ in range(rank))

    if not isinstance(pad_width, Sequence) or isinstance(pad_width, (str, bytes)):
        raise ValueError("pad_width must be int or sequence")

    items = tuple(pad_width)
    if rank == 0:
        if len(items) == 0:
            return ()
        raise ValueError("pad_width for rank-0 must be empty")

    if len(items) == 2 and all(_is_int_scalar(v) for v in items):
        lo, hi = int(items[0]), int(items[1])
        if lo < 0 or hi < 0:
            raise ValueError("negative pad widths are not supported")
        return tuple((lo, hi) for _ in range(rank))

    if len(items) != rank:
        raise ValueError("pad_width length must match input rank")

    return tuple(_normalize_axis_pad(item) for item in items)


def _normalize_constant_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return np.asarray(arr).reshape(()).item()
    raise ValueError("only scalar constant_values are supported")


def _abstract_eval_via_orig_pad(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    *,
    pad_width: PadWidth,
    constant_value: Any,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(
        lambda a: orig(a, pad_width, mode="constant", constant_values=constant_value),
        x_spec,
    )
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


@register_primitive(
    jaxpr_primitive=_PAD_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html",
    onnx=[
        {
            "component": "Pad",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pad.html",
        }
    ],
    since="0.12.7",
    context="primitives.jnp",
    component="pad",
    testcases=[
        {
            "testcase": "jnp_pad_constant_1d",
            "callable": lambda x: jnp.pad(x, 1),
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(["Pad:7"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_pad_constant_2d_tuple",
            "callable": lambda x: jnp.pad(x, ((1, 0), (0, 2)), constant_values=1.0),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(["Pad:3x5"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_pad_constant_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.pad(y, (1, 1)))(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpPadPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _PAD_PRIM
    _FUNC_NAME: ClassVar[str] = "pad"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        pad_width: PadWidth,
        constant_value: Any,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig_pad(
            JnpPadPlugin._PRIM,
            JnpPadPlugin._FUNC_NAME,
            x,
            pad_width=pad_width,
            constant_value=constant_value,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = dict(getattr(eqn, "params", {}) or {})
        pad_width = cast(PadWidth, params.get("pad_width"))
        constant_value = params.get("constant_value", 0)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_pad_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_pad_out")
        )

        pad_width = _normalize_pad_width(
            pad_width,
            rank=len(tuple(getattr(getattr(x_var, "aval", None), "shape", ()))),
        )

        begins = [int(lo) for lo, _ in pad_width]
        ends = [int(hi) for _, hi in pad_width]
        pads_arr = np.asarray(begins + ends, dtype=np.int64)

        pads_val = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("jnp_pad_pads"),
            array=pads_arr,
        )
        _stamp_type_and_shape(pads_val, (pads_arr.size,))
        _ensure_value_metadata(ctx, pads_val)

        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        cval_np = np.asarray(constant_value, dtype=x_dtype)
        cval = ctx.bind_const_for_var(object(), cval_np)
        _stamp_type_and_shape(cval, tuple(getattr(cval_np, "shape", ())))
        _ensure_value_metadata(ctx, cval)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("jnp_pad_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_pad_out")

        result = ctx.builder.Pad(
            x_val,
            pads_val,
            cval,
            mode="constant",
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.pad not found for patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                array: ArrayLike,
                pad_width: Any,
                mode: str = "constant",
                **kwargs: Any,
            ) -> jax.Array:
                if mode != "constant":
                    return orig(array, pad_width, mode=mode, **kwargs)

                unsupported = {
                    key
                    for key in kwargs
                    if key
                    not in {
                        "constant_values",
                    }
                }
                if unsupported:
                    return orig(array, pad_width, mode=mode, **kwargs)

                arr = jnp.asarray(array)
                rank = len(tuple(arr.shape))
                if rank == 0:
                    return orig(array, pad_width, mode=mode, **kwargs)

                try:
                    pw = _normalize_pad_width(pad_width, rank)
                    cval = _normalize_constant_value(kwargs.get("constant_values", 0))
                except Exception:
                    return orig(array, pad_width, mode=mode, **kwargs)

                return cls._PRIM.bind(arr, pad_width=pw, constant_value=cval)

            return _patched

        return [
            AssignSpec("jax.numpy", "pad_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr="pad",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpPadPlugin._PRIM.def_impl
def _pad_impl(
    array: ArrayLike,
    *,
    pad_width: PadWidth,
    constant_value: Any,
) -> jax.Array:
    orig = get_orig_impl(JnpPadPlugin._PRIM, JnpPadPlugin._FUNC_NAME)
    return orig(array, pad_width, mode="constant", constant_values=constant_value)


register_jvp_via_jax_jvp(JnpPadPlugin._PRIM, _pad_impl)


def _pad_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[Any, ...],
    *,
    pad_width: PadWidth,
    constant_value: Any,
) -> tuple[jax.Array, Any]:
    (x,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpPadPlugin._PRIM.bind(
            x,
            pad_width=pad_width,
            constant_value=constant_value,
        )
        return out, batching.not_mapped

    batch_size = x.shape[int(bdim)]
    x_front = batching.bdim_at_front(x, int(bdim), batch_size)
    orig = get_orig_impl(JnpPadPlugin._PRIM, JnpPadPlugin._FUNC_NAME)
    out = jax.vmap(
        lambda a: orig(
            a,
            pad_width,
            mode="constant",
            constant_values=constant_value,
        )
    )(x_front)
    return out, 0


batching.primitive_batchers[JnpPadPlugin._PRIM] = _pad_batch_rule
