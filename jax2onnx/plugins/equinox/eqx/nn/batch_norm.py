# jax2onnx/plugins/equinox/eqx/nn/batch_norm.py

from __future__ import annotations

from typing import ClassVar, Final

import equinox as eqx
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_EQX_BATCH_NORM: Final[eqx.nn.BatchNorm] = eqx.nn.BatchNorm(
    input_size=3,
    axis_name="batch",
    inference=True,
)
_EQX_BATCH_NORM_STATE: Final[eqx.nn.State] = eqx.nn.State(_EQX_BATCH_NORM)


@register_primitive(
    jaxpr_primitive="eqx.nn.batch_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.BatchNorm",
    onnx=[
        {
            "component": "BatchNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
        }
    ],
    since="0.12.2",
    context="primitives.eqx",
    component="batch_norm",
    testcases=[
        {
            "testcase": "eqx_batch_norm_inference",
            "callable": lambda x, _mod=_EQX_BATCH_NORM, _state=_EQX_BATCH_NORM_STATE: _mod(
                x, _state
            )[
                0
            ],
            "input_shapes": [(3, 4, 4)],
            "expected_output_shapes": [(3, 4, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class BatchNormPlugin(PrimitiveLeafPlugin):
    """IR-only support marker for ``equinox.nn.BatchNorm`` via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.batch_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: ShapedArray, *args: object, **kwargs: object) -> ShapedArray:
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: object, eqn: object) -> None:
        raise NotImplementedError(
            "BatchNorm primitive should not reach lowering; it is inlined."
        )
