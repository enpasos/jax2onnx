# file: jax2onnx/plugins/flax/nnx/embed.py

from typing import TYPE_CHECKING, Callable, Any

import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# 1. Define a new JAX primitive for nnx.Embed's call behavior.
nnx.embed_p = Primitive("nnx.embed")
nnx.embed_p.multiple_results = False


# 2. Register the plugin with its metadata and test cases.
@register_primitive(
    jaxpr_primitive=nnx.embed_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Embed",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nnx",
    component="embed",
    testcases=[
        {
            "testcase": "token_embedding",
            "callable": nnx.Embed(num_embeddings=3144, features=48, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 64)],
            "input_dtypes": [jnp.int32],
        },
        {
            "testcase": "positional_embedding",
            "callable": nnx.Embed(num_embeddings=64, features=48, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 64)],
            "input_dtypes": [jnp.int32],
        },
    ],
)
class EmbedPlugin(PrimitiveLeafPlugin):
    """Plugin for converting flax.nnx.Embed to ONNX."""

    _ORIG_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(indices_aval, embedding_aval):
        """
        Computes the output shape for the embedding operation.
        """
        features_dim = embedding_aval.shape[-1]
        output_shape = indices_aval.shape + (features_dim,)
        return core.ShapedArray(output_shape, embedding_aval.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs,
        node_outputs,
        params,
    ):
        """
        Converts the embed primitive to an ONNX Gather node.
        """
        indices_var, embedding_var = node_inputs
        (output_var,) = node_outputs

        # Determine the correct output dtype based on the converter's settings
        original_dtype = embedding_var.aval.dtype
        output_dtype = original_dtype
        if jnp.issubdtype(original_dtype, jnp.floating):
            if (
                hasattr(s.builder, "enable_double_precision")
                and s.builder.enable_double_precision
            ):
                output_dtype = jnp.float64

        # *** THE FIX IS HERE ***
        # Update the output variable's abstract value (aval) with the correct dtype.
        # The main converter loop will use this updated aval to create the
        # final ONNX ValueInfoProto, preventing the type mismatch.
        output_var.aval = core.ShapedArray(output_var.aval.shape, output_dtype)

        # Now, create the ONNX node.
        indices_name = s.get_name(indices_var)
        embedding_name = s.get_name(embedding_var)
        output_name = s.get_name(output_var)

        gather_node = helper.make_node(
            "Gather",
            inputs=[embedding_name, indices_name],
            outputs=[output_name],
            axis=0,
            name=s.get_unique_name("embed_gather"),
        )
        s.add_node(gather_node)
        # We no longer need to call s.add_shape_info here, as the main loop handles it.

    @staticmethod
    def _embed_binding(embedding_table, indices):
        """Binds inputs to the embed primitive."""
        return nnx.embed_p.bind(indices, embedding_table)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns a patched version of Embed's __call__ method."""
        EmbedPlugin._ORIG_CALL = orig_fn

        def patched_embed_call(self, inputs):
            embedding = self.embedding.value
            return EmbedPlugin._embed_binding(embedding, inputs)

        return patched_embed_call

    @staticmethod
    def patch_info():
        """Provides patching information for nnx.Embed."""
        return {
            "patch_targets": [nnx.Embed],
            "patch_function": EmbedPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


# Register the abstract evaluation rule with the primitive.
nnx.embed_p.def_abstract_eval(EmbedPlugin.abstract_eval)
