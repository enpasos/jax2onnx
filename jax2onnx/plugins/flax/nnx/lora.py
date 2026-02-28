# jax2onnx/plugins/flax/nnx/lora.py

from __future__ import annotations

from typing import Final

from flax import nnx

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


_LORA_ONNX: Final[list[dict[str, str]]] = [
    {"component": "MatMul", "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html"},
    {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
]


@register_primitive(
    jaxpr_primitive="nnx.lora",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/lora.html#flax.nnx.LoRA",
    onnx=_LORA_ONNX,
    since="0.12.2",
    context="primitives.nnx",
    component="lora",
    testcases=[
        {
            "testcase": "lora_basic",
            "callable": construct_and_call(
                nnx.LoRA,
                8,
                2,
                4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 4)],
        },
        {
            "testcase": "lora_static",
            "callable": construct_and_call(
                nnx.LoRA,
                8,
                2,
                4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 8)],
            "expected_output_shapes": [(3, 4)],
        },
    ],
)
class LoRAPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.LoRA`` (module body is inlined)."""

    def lower(self, ctx, eqn):  # type: ignore[override]
        raise NotImplementedError(
            "nnx.lora primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="nnx.lora_linear",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/lora.html#flax.nnx.LoRALinear",
    onnx=_LORA_ONNX,
    since="0.12.2",
    context="primitives.nnx",
    component="lora_linear",
    testcases=[
        {
            "testcase": "lora_linear_basic",
            "callable": construct_and_call(
                nnx.LoRALinear,
                8,
                4,
                lora_rank=2,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                lora_dtype=with_requested_dtype(),
                lora_param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 4)],
        },
        {
            "testcase": "lora_linear_high_rank",
            "callable": construct_and_call(
                nnx.LoRALinear,
                8,
                4,
                lora_rank=2,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                lora_dtype=with_requested_dtype(),
                lora_param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 8)],
            "expected_output_shapes": [("B", 10, 4)],
        },
    ],
)
class LoRALinearPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.LoRALinear`` (module body is inlined)."""

    def lower(self, ctx, eqn):  # type: ignore[override]
        raise NotImplementedError(
            "nnx.lora_linear primitive should not reach lowering; it is inlined."
        )
