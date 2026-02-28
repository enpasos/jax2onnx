# jax2onnx/plugins/equinox/eqx/nn/spectral_norm.py

from __future__ import annotations

from typing import ClassVar, Final

import equinox as eqx
import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_EQX_SPECTRAL_NORM: Final[eqx.nn.SpectralNorm] = eqx.nn.SpectralNorm(
    layer=eqx.nn.Linear(8, 4, key=jax.random.PRNGKey(0)),
    weight_name="weight",
    inference=True,
    key=jax.random.PRNGKey(1),
)
_EQX_SPECTRAL_NORM_STATE: Final[eqx.nn.State] = eqx.nn.State(_EQX_SPECTRAL_NORM)


@register_primitive(
    jaxpr_primitive="eqx.nn.spectral_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.SpectralNorm",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.5",
    context="primitives.eqx",
    component="spectral_norm",
    testcases=[
        {
            "testcase": "eqx_spectral_norm_linear_inference",
            "callable": lambda x, _mod=_EQX_SPECTRAL_NORM, _state=_EQX_SPECTRAL_NORM_STATE: _mod(
                x, _state
            )[
                0
            ],
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
