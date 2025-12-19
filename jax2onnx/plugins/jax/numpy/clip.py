# jax2onnx/plugins/jax/numpy/clip.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Final

import jax
import jax.extend.core as jax_core_ext
from jax import core
import jax.numpy as jnp
from jax.interpreters import batching as jax_batching
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

ScalarBound = bool | int | float | np.generic
JaxValue = jax_core_ext.Var | jax_core_ext.Literal


def _np_dtype(x: DTypeLike) -> np.dtype[Any]:
    return x if isinstance(x, np.dtype) else np.dtype(x)


def _dtype_min_max(dtype: np.dtype[Any]) -> tuple[ScalarBound, ScalarBound]:
    if np.issubdtype(dtype, np.floating):
        return -jnp.inf, jnp.inf
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    if dtype == np.bool_:
        return False, True
    return -jnp.inf, jnp.inf


def _cast_value(
    ctx: LoweringContextProtocol,
    src_val: ir.Value,
    src_var: JaxValue,
    target_dtype: np.dtype[Any],
    tag: str,
) -> ir.Value:
    var_dtype = _np_dtype(
        getattr(getattr(src_var, "aval", None), "dtype", target_dtype)
    )
    if var_dtype == target_dtype:
        return src_val

    dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
    cast_val = ctx.builder.Cast(
        src_val,
        _outputs=[ctx.fresh_name(f"clip_{tag}_cast")],
        to=int(dtype_enum.value),
    )
    cast_val.type = ir.TensorType(dtype_enum)
    cast_shape = tuple(getattr(getattr(src_var, "aval", None), "shape", ()))
    _stamp_type_and_shape(cast_val, cast_shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


_CLIP_PRIM: Final = make_jnp_primitive("jax.numpy.clip")


@register_primitive(
    jaxpr_primitive=_CLIP_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.clip.html",
    onnx=[
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="clip",
    testcases=[
        {
            "testcase": "clip_i32_scalar_bounds",
            "callable": lambda x: jnp.clip(x, 0, 4),
            "input_values": [np.array([-3, 1, 9, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 4.0}},
                        "path": "Max:4 -> Min:4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "clip_f32_scalar_bounds_no_upcast_f64_mode",
            "callable": lambda x: jnp.clip(x, -1.5, 2.5),
            "input_values": [np.array([-2.0, 0.5, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2.5}},
                        "path": "Max:3 -> Min:3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "clip_only_upper",
            "callable": lambda x: jnp.clip(x, None, 1.0),
            "input_values": [np.array([-2.0, 0.5, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Max:3 -> Min:3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "clip_only_lower",
            "callable": lambda x: jnp.clip(x, -1, None),
            "input_values": [np.array([-5, -1, 0, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2147483647.0}},
                        "path": "Max:4 -> Min:4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "clip_broadcast_bounds",
            "callable": lambda x, lo, hi: jnp.clip(x, lo, hi),
            "input_values": [
                np.array([[-2.0, -0.5, 3.0], [1.0, 2.0, 5.0]], dtype=np.float64),
                np.array([[-1.0, 0.0, 0.0]], dtype=np.float64),
                np.array([[1.5]], dtype=np.float64),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Max:2x3 -> Min:2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpClipPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for :func:`jax.numpy.clip`."""

    _PRIM: ClassVar = _CLIP_PRIM
    _FUNC_NAME: ClassVar[str] = "clip"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        a_min: core.AbstractValue,
        a_max: core.AbstractValue,
        **_: object,
    ) -> core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, lo_var, hi_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("clip_x"))
        lo_val = ctx.get_value_for_var(lo_var, name_hint=ctx.fresh_name("clip_lo"))
        hi_val = ctx.get_value_for_var(hi_var, name_hint=ctx.fresh_name("clip_hi"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("clip_out"))

        x_dtype = _np_dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))
        dtype_enum = _dtype_to_ir(x_dtype, ctx.builder.enable_double_precision)

        lo_input = _cast_value(ctx, lo_val, lo_var, x_dtype, "lo")
        hi_input = _cast_value(ctx, hi_val, hi_var, x_dtype, "hi")

        max_val = ctx.builder.Max(
            x_val,
            lo_input,
            _outputs=[ctx.fresh_name("clip_max_tmp")],
        )
        max_val.type = ir.TensorType(dtype_enum)
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(max_val, x_shape)
        _ensure_value_metadata(ctx, max_val)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("clip_out")
        result = ctx.builder.Min(
            max_val,
            hi_input,
            _outputs=[desired_name],
        )
        result.type = ir.TensorType(dtype_enum)
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.clip not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                a_min: ArrayLike | None = None,
                a_max: ArrayLike | None = None,
            ) -> jax.Array:
                x = jnp.asarray(a)
                dtype = x.dtype
                lo_default, hi_default = _dtype_min_max(_np_dtype(dtype))

                lo = jnp.asarray(lo_default if a_min is None else a_min, dtype=dtype)
                hi = jnp.asarray(hi_default if a_max is None else a_max, dtype=dtype)

                return cls._PRIM.bind(x, lo, hi)

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpClipPlugin._PRIM.def_impl
def _clip_impl(x: ArrayLike, a_min: ArrayLike, a_max: ArrayLike) -> jax.Array:
    orig = get_orig_impl(JnpClipPlugin._PRIM, JnpClipPlugin._FUNC_NAME)
    return orig(x, a_min, a_max)


JnpClipPlugin._PRIM.def_abstract_eval(JnpClipPlugin.abstract_eval)


def _clip_batching_rule(args, dims):
    return broadcast_batcher_compat(JnpClipPlugin._PRIM, args, dims)


jax_batching.primitive_batchers[JnpClipPlugin._PRIM] = _clip_batching_rule
