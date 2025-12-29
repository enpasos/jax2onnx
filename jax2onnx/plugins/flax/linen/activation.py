# jax2onnx/plugins/flax/linen/activation.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax.nn as jnn
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import linen as nn

from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ACTIVATION_TARGETS: Final[dict[str, tuple[tuple[object, str], ...]]] = {
    "glu": ((jnn, "glu"),),
    "hard_sigmoid": ((jnn, "hard_sigmoid"),),
    "hard_swish": ((jnn, "hard_swish"),),
    "hard_tanh": ((jnn, "hard_tanh"),),
    "log_sigmoid": ((jnn, "log_sigmoid"),),
    "log_softmax": ((jnn, "log_softmax"),),
    "relu6": ((jnn, "relu6"),),
    "silu": ((jnn, "silu"), (jnn, "swish")),
    "swish": ((jnn, "swish"), (jnn, "silu")),
    "tanh": ((jnn, "tanh"), (jnp, "tanh")),
    "normalize": ((jnn, "normalize"), (jnn, "standardize")),
    "one_hot": ((jnn, "one_hot"),),
}

_PATCH_MODULES: Final[tuple[str, ...]] = ("flax.linen.activation", "flax.linen")


def _resolve_target(name: str) -> Callable | None:
    for module, attr in _ACTIVATION_TARGETS[name]:
        fn = getattr(module, attr, None)
        if fn is not None:
            return fn
    return None


def _make_forwarder(name: str) -> Callable[[Callable | None], Callable | None]:
    def _patch(orig: Callable | None) -> Callable | None:
        def _forward(*args, **kwargs):
            target = _resolve_target(name)
            if target is not None:
                return target(*args, **kwargs)
            if callable(orig):
                return orig(*args, **kwargs)
            raise RuntimeError(f"Activation '{name}' is not available in jax.nn/jnp.")

        return _forward

    return _patch


@register_primitive(
    jaxpr_primitive="linen.activation",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/activation.html",
    since="v0.11.0",
    context="primitives.linen",
    component="activation",
    testcases=[
        {
            "testcase": "activation_glu_basic",
            "callable": lambda x: nn.activation.glu(x),
            "input_shapes": [(2, 4)],
            "expected_output_shapes": [(2, 2)],
        },
        {
            "testcase": "activation_hard_sigmoid_basic",
            "callable": lambda x: nn.activation.hard_sigmoid(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_hard_silu_basic",
            "callable": lambda x: nn.activation.hard_silu(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_hard_swish_basic",
            "callable": lambda x: nn.activation.hard_swish(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_hard_tanh_basic",
            "callable": lambda x: nn.activation.hard_tanh(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_log_sigmoid_basic",
            "callable": lambda x: nn.activation.log_sigmoid(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_log_softmax_basic",
            "callable": lambda x: nn.activation.log_softmax(x, axis=-1),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_relu6_basic",
            "callable": lambda x: nn.activation.relu6(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_silu_basic",
            "callable": lambda x: nn.activation.silu(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_swish_basic",
            "callable": lambda x: nn.activation.swish(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_tanh_basic",
            "callable": lambda x: nn.activation.tanh(x),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "activation_normalize_basic",
            "callable": lambda x: nn.activation.normalize(x, axis=-1),
            "input_shapes": [(2, 4)],
            "expected_output_shapes": [(2, 4)],
        },
        {
            "testcase": "activation_one_hot_basic",
            "callable": lambda x: nn.activation.one_hot(x, num_classes=6),
            "input_shapes": [(4,)],
            "input_dtypes": [jnp.int32],
            "expected_output_shapes": [(4, 6)],
        },
    ],
)
class LinenActivationPlugin(PrimitiveLeafPlugin):
    """Route flax.linen.activation helpers to jax.nn/jnp implementations."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.activation")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "Linen activation patching should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        specs: list[MonkeyPatchSpec] = []
        for module in _PATCH_MODULES:
            for name in _ACTIVATION_TARGETS:
                specs.append(
                    MonkeyPatchSpec(
                        target=module,
                        attr=name,
                        make_value=_make_forwarder(name),
                        delete_if_missing=False,
                    )
                )
        return specs
