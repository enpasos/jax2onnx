# jax2onnx/plugins/equinox/eqx/nn/spectral_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar

import equinox as eqx
import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_native_primitive,
    with_prng_key,
    with_requested_dtype,
)


def _spectral_norm_example(
    dtype: Any, *, layer_key: jax.Array, key: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    model = eqx.nn.SpectralNorm(
        layer=eqx.nn.Linear(8, 4, dtype=dtype, key=layer_key),
        weight_name="weight",
        inference=True,
        key=key,
    )
    state = eqx.nn.State(model)

    def call(x: jax.Array) -> jax.Array:
        return model(x, state)[0]

    return call


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
            "callable": construct_and_call(
                _spectral_norm_example,
                dtype=with_requested_dtype(),
                layer_key=with_prng_key(0),
                key=with_prng_key(1),
            ),
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
