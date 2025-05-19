# File: jax2onnx/plugins/jax/lax/copy.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

from jax import lax, core
import numpy as np

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax.core import Var
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.copy")

# 1. Get the primitive
lax_copy_p = lax.copy_p

# Special Note on this Plugin:
# The JAX public API `jax.lax.copy(x)` has been deprecated and subsequently removed
# in recent JAX versions. However, the underlying JAX primitive `jax.lax.copy_p`
# still exists and can appear in JAXprs resulting from various JAX transformations
# or explicit binding. This plugin handles the `lax.copy_p` primitive, which
# essentially acts as an identity operation (passing the input tensor through).
# Test cases for this plugin must therefore use `lax.copy_p.bind(x)` or another
# JAX operation that reliably lowers to `lax.copy_p`.


@register_primitive(
    jaxpr_primitive=lax_copy_p.name,
    jax_doc="Handles the JAX primitive lax.copy_p. Note: jax.lax.copy API is removed.",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="<your_current_version>",  # Please update with your jax2onnx version
    context="primitives.lax",
    component="copy",
    testcases=[
        {
            "testcase": "copy_float32_array",
            "callable": lambda x: lax.copy_p.bind(x),
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.float32],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float32],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "copy_int64_scalar",
            "callable": lambda x: lax.copy_p.bind(x),
            "input_values": [np.array(10, dtype=np.int64)],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int64],
            "run_only_f64_variant": True,
        },
    ],
)
class CopyPlugin(PrimitiveLeafPlugin):
    """Map JAX lax.copy_p to ONNX Identity."""

    @staticmethod
    def abstract_eval(
        operand_aval: core.ShapedArray, **kwargs: Any
    ) -> core.ShapedArray:
        return core.ShapedArray(
            operand_aval.shape, operand_aval.dtype, weak_type=operand_aval.weak_type
        )

    def to_onnx(
        self,
        s: Jaxpr2OnnxConverter,
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        if len(node_inputs) != 1 or len(node_outputs) != 1:
            logger.error(
                f"CopyPlugin expects 1 input and 1 output for primitive '{lax_copy_p.name}', "
                f"got {len(node_inputs)} inputs and {len(node_outputs)} outputs."
            )
            raise ValueError("CopyPlugin: Invalid number of inputs or outputs.")

        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])

        # Corrected: Use s.builder.create_node to make the NodeProto,
        # then s.add_node to add it to the graph.
        identity_node_proto = s.builder.create_node(  # This calls helper.make_node
            op_type="Identity",
            inputs=[input_name],
            outputs=[output_name],
            name=s.builder.get_unique_name(f"IdentityFor_{lax_copy_p.name}"),
        )
        s.add_node(identity_node_proto)
        # Shape info for the output will be handled by the converter based on out_v.aval
