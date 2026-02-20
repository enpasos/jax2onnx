# jax2onnx/plugins/flax/linen/prelu.py

from __future__ import annotations

from typing import Any, Callable, ClassVar

import jax.numpy as jnp
from flax import linen as nn
from jax.extend.core import Primitive

from jax2onnx.plugins.flax.nnx import prelu as nnx_prelu
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


@register_primitive(
    jaxpr_primitive="linen.prelu",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.PReLU",
    onnx=[
        {"component": "PRelu", "doc": "https://onnx.ai/onnx/operators/onnx__PRelu.html"}
    ],
    since="0.12.1",
    context="primitives.linen",
    component="prelu",
    testcases=[
        {
            "testcase": "linen_prelu_default",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.PReLU,
                input_shape=(1, 6),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["PRelu:Bx6"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linen_prelu_custom_slope",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.PReLU,
                input_shape=(1, 3, 4),
                negative_slope_init=0.2,
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["PRelu:2x3x4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class PReluPlugin(nnx_prelu.PReluPlugin):
    """IR-only plugin for ``flax.linen.PReLU`` → ONNX ``PRelu``."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.prelu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable[..., Any] | None] = None

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        return [
            MonkeyPatchSpec(
                target="flax.linen.PReLU",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        PReluPlugin._ORIGINAL_CALL = orig_fn
        prim = PReluPlugin._PRIM

        def patched(self: nn.PReLU, inputs: Any) -> Any:
            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(self, inputs)
            variables = scope.variables()
            params = variables.get("params", {})
            slope = params.get("negative_slope")
            if slope is None:
                return orig_fn(self, inputs)
            return prim.bind(inputs, jnp.asarray(slope, dtype=inputs.dtype))

        return patched


@PReluPlugin._PRIM.def_impl
def _linen_prelu_impl(inputs: Any, negative_slope: Any) -> Any:
    return nnx_prelu._prelu_impl(inputs, negative_slope)
