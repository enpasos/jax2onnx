# jax2onnx/plugins/flax/linen/embed.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import linen as nn

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import embed as nnx_embed
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

EXPECT_EMBED_GATHER: Final = nnx_embed.EXPECT_EMBED_GATHER


@register_primitive(
    jaxpr_primitive="linen.embed",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Embed",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="0.11.0",
    context="primitives.linen",
    component="embed",
    testcases=[
        {
            "testcase": "token_embedding",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Embed,
                input_shape=(1, 64),
                dtype=jnp.int32,
                num_embeddings=3144,
                features=48,
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 64)],
            "input_dtypes": [jnp.int32],
            "expected_output_shapes": [("B", 64, 48)],
            "post_check_onnx_graph": EXPECT_EMBED_GATHER,
        },
        {
            "testcase": "positional_embedding",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Embed,
                input_shape=(1, 64),
                dtype=jnp.int32,
                num_embeddings=64,
                features=32,
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 64)],
            "input_dtypes": [jnp.int32],
            "expected_output_shapes": [("B", 64, 32)],
            "post_check_onnx_graph": EXPECT_EMBED_GATHER,
        },
    ],
)
class EmbedPlugin(nnx_embed.EmbedPlugin):
    """IR-only plugin for flax.linen.Embed → ONNX Gather."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.embed")
    _ORIGINAL_CALL: ClassVar[Callable[..., Any] | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _make_patch(orig_fn: Callable[..., Any] | None) -> Callable[..., Any]:
        EmbedPlugin._ORIGINAL_CALL = orig_fn
        prim = EmbedPlugin._PRIM

        def call_orig(self: Any, inputs: Any) -> Any:
            if orig_fn is None:
                raise RuntimeError("flax.linen.Embed.__call__ is not available.")
            return orig_fn(self, inputs)

        def patched(self: Any, inputs: Any) -> Any:
            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return call_orig(self, inputs)

            if int(getattr(self, "num_embeddings", 0)) == 1:
                return call_orig(self, inputs)

            variables = scope.variables()
            params = variables.get("params", {})
            embedding = params.get("embedding")
            if embedding is None:
                return call_orig(self, inputs)

            if not jnp.issubdtype(inputs.dtype, jnp.integer):
                raise ValueError("Input type must be an integer or unsigned integer.")

            (embedding,) = self.promote_dtype(
                embedding, dtype=getattr(self, "dtype", None), inexact=False
            )

            return prim.bind(inputs, embedding)

        return patched

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.linen", "embed_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.linen.Embed",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]


@EmbedPlugin._PRIM.def_impl
def _embed_impl(indices: Any, embedding: Any) -> Any:
    return jnp.take(embedding, indices, axis=0)
