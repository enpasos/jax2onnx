# file: jax2onnx/plugins/flax/nnx/embed.py

from typing import TYPE_CHECKING

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
            "callable": nnx.Embed(num_embeddings=50304, features=768, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
        },
        {
            "testcase": "positional_embedding",
            "callable": nnx.Embed(num_embeddings=1024, features=768, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 1024)],
            "input_dtypes": [jnp.int32],
        },
    ],
)
class EmbedPlugin(PrimitiveLeafPlugin):
    """Plugin for converting flax.nnx.Embed to ONNX."""

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

        # --- FIX STARTS HERE ---
        # The jaxpr provides the original dtype. If it's a floating type,
        # we must use the converter's working_dtype to ensure precision
        # (e.g., float32 vs float64) is handled correctly.
        original_dtype = embedding_var.aval.dtype
        output_dtype = original_dtype
        if jnp.issubdtype(original_dtype, jnp.floating):
            output_dtype = s.working_dtype
        # --- FIX ENDS HERE ---

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

        # Update the graph's output metadata with the correct shape and dtype
        batch_shape = indices_var.aval.shape
        feat = embedding_var.aval.shape[-1]
        s.add_shape_info(output_name, batch_shape + (feat,), output_dtype)


# Register the abstract evaluation rule with the primitive.
nnx.embed_p.def_abstract_eval(EmbedPlugin.abstract_eval)
