# jax2onnx/plugins/equinox/eqx/nn/spectral_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import equinox as eqx
import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    with_requested_dtype,
    with_prng_key,
    register_native_primitive,
)


def _make_eqx_spectral_norm(
    dtype: Any,
    layer_key: jax.Array,
    state_key: jax.Array,
) -> Callable[..., Any]:
    module = eqx.nn.SpectralNorm(
        layer=eqx.nn.Linear(8, 4, dtype=dtype, key=layer_key),
        weight_name="weight",
        inference=True,
        key=state_key,
    )
    state = eqx.nn.State(module)

    def call(x: jax.Array) -> jax.Array:
        return module(x, state)[0]

    return call


_EQX_SPECTRAL_NORM: Final = construct_and_call(
    _make_eqx_spectral_norm,
    dtype=with_requested_dtype(),
    layer_key=with_prng_key(0),
    state_key=with_prng_key(1),
)


@register_native_primitive(
    jaxpr_primitive="eqx.nn.spectral_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.SpectralNorm",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.2",
    context="primitives.eqx",
    component="spectral_norm",
    testcases=[
        {
            "testcase": "eqx_spectral_norm_linear_inference",
            "callable": _EQX_SPECTRAL_NORM,
            "input_shapes": [(8,)],
            "expected_output_shapes": [(4,)],
            "run_only_f32_variant": True,
        },
    ],
)
class SpectralNormPlugin(PrimitiveLeafPlugin):
    """IR-only support marker for ``equinox.nn.SpectralNorm`` via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.spectral_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: ShapedArray, *args: object, **kwargs: object) -> ShapedArray:
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: object, eqn: object) -> None:
        raise NotImplementedError(
            "SpectralNorm primitive should not reach lowering; it is inlined."
        )
