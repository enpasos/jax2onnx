# jax2onnx/plugins/flax/linen/multi_head_attention.py

from __future__ import annotations

from typing import Callable, ClassVar

from jax.core import ShapedArray
from jax.extend.core import Primitive
from flax import linen as nn

from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.flax.linen.multi_head_dot_product_attention import _make_mha_patch
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


@register_primitive(
    jaxpr_primitive="linen.multi_head_attention",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.MultiHeadAttention",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="multi_head_attention",
    testcases=[
        {
            "testcase": "multi_head_attention_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.MultiHeadAttention,
                input_shape=(1, 4, 8),
                num_heads=2,
                dropout_rate=0.0,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 8)],
            "expected_output_shapes": [("B", 4, 8)],
        },
        {
            "testcase": "multi_head_attention_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.MultiHeadAttention,
                input_shape=(1, 3, 6),
                num_heads=3,
                use_bias=False,
                dropout_rate=0.0,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 3, 6)],
            "expected_output_shapes": [("B", 3, 6)],
        },
    ],
)
class MultiHeadAttentionPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.MultiHeadAttention via dot_product_attention."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.multi_head_attention")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "MultiHeadAttention primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.MultiHeadAttention",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            )
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        MultiHeadAttentionPlugin._ORIGINAL_CALL = orig_fn
        return _make_mha_patch(orig_fn)
