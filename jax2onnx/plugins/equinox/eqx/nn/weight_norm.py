# jax2onnx/plugins/equinox/eqx/nn/weight_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    with_requested_dtype,
    with_prng_key,
    register_native_primitive,
)


def _make_eqx_weight_norm_linear(dtype: Any, key: jax.Array) -> Callable[..., Any]:
    layer = eqx.nn.Linear(8, 4, dtype=dtype, key=key)
    return cast(Callable[..., Any], jax.vmap(eqx.nn.WeightNorm(layer)))


def _make_eqx_weight_norm_conv2d(dtype: Any, key: jax.Array) -> eqx.nn.WeightNorm:
    layer = eqx.nn.Conv2d(3, 5, kernel_size=3, dtype=dtype, key=key)
    return eqx.nn.WeightNorm(layer)


_EQX_WEIGHT_NORM_LINEAR: Final = construct_and_call(
    _make_eqx_weight_norm_linear,
    dtype=with_requested_dtype(),
    key=with_prng_key(0),
)
_EQX_WEIGHT_NORM_CONV2D: Final = construct_and_call(
    _make_eqx_weight_norm_conv2d,
    dtype=with_requested_dtype(),
    key=with_prng_key(1),
)


@register_native_primitive(
    jaxpr_primitive="eqx.nn.weight_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.WeightNorm",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.2",
    context="primitives.eqx",
    component="weight_norm",
    testcases=[
        {
            "testcase": "eqx_weight_norm_linear",
            "callable": _EQX_WEIGHT_NORM_LINEAR,
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 4)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "eqx_weight_norm_conv2d",
            "callable": _EQX_WEIGHT_NORM_CONV2D,
            "input_shapes": [(3, 8, 8)],
            "expected_output_shapes": [(5, 6, 6)],
            "run_only_f32_variant": True,
        },
    ],
)
class WeightNormPlugin(PrimitiveLeafPlugin):
    """Support ``equinox.nn.WeightNorm`` by inlining normalized-weight calls."""

    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.weight_norm")
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable[..., Any] | None] = None

    @staticmethod
    def abstract_eval(x: ShapedArray, *args: Any, **kwargs: Any) -> ShapedArray:
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: Any, eqn: Any) -> None:
        raise NotImplementedError(
            "WeightNorm primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            MonkeyPatchSpec(
                target="equinox.nn.WeightNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        WeightNormPlugin._ORIGINAL_CALL = orig_fn

        def patched(
            self: eqx.nn.WeightNorm,
            x: jax.Array,
            *,
            key: jax.Array | None = None,
        ) -> jax.Array:
            if key is not None:
                return orig_fn(self, x, key=key)

            layer = getattr(self, "layer", None)
            g = getattr(self, "g", None)
            if layer is None or g is None:
                return orig_fn(self, x, key=key)

            weight_name = str(getattr(self, "weight_name", "weight"))
            try:
                v = getattr(layer, weight_name)
            except Exception:
                return orig_fn(self, x, key=key)

            axis = getattr(self, "axis", 0)
            try:
                if axis is None:
                    denom = jnp.linalg.norm(v, keepdims=True)
                else:
                    axis = int(axis)
                    denom = jax.vmap(
                        lambda w: jnp.linalg.norm(w, keepdims=True),
                        in_axes=axis,
                        out_axes=axis,
                    )(v)

                weight = v * jnp.asarray(g, dtype=v.dtype) / denom
                normalized_layer = eqx.tree_at(
                    lambda layer_ref: getattr(layer_ref, weight_name),
                    layer,
                    weight,
                )
            except Exception:
                return orig_fn(self, x, key=key)

            return normalized_layer(x)

        return patched
