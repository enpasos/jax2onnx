# file: jax2onnx/plugins/jax/numpy/shape.py

from typing import TYPE_CHECKING

import jax.core as core
import numpy as np
import onnx
from jax import numpy as jnp
from jax.extend.core import Primitive  # If defined in this file
from onnx import helper


from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# --- Primitive Definition (copy from above if defining here) ---
shape_p = Primitive("jnp.shape")
shape_p.multiple_results = False


def shape_impl(operand):
    # Return the dynamic shape as a 1-D JAX array of int64
    # (pull shape directly from the operand to avoid a host→device round-trip)
    return jnp.asarray(operand.shape, dtype=jnp.int64)


def shape_abstract_eval(operand_aval):
    rank = len(operand_aval.shape)
    return core.ShapedArray((rank,), np.dtype(np.int64))


shape_p.def_impl(shape_impl)
shape_p.def_abstract_eval(shape_abstract_eval)
# --- End Primitive Definition ---


@register_primitive(
    # Use the name of the primitive we defined
    jaxpr_primitive=shape_p.name,
    # Link to JAX docs for np.shape (or jnp.shape)
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.shape.html",
    onnx=[
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        }
    ],
    # Assign appropriate version, context, component
    since="0.4.0",
    context="primitives.jnp",
    component="shape",
    testcases=[
        {
            "testcase": "shape_basic",
            "callable": lambda x: jnp.shape(x),
            "input_shapes": [(3, 4, 5)],
            "expected_output_shapes": [(3,)],  # Output is 1D tensor of length = rank
        },
        {
            "testcase": "shape_dynamic",
            "callable": lambda x: jnp.shape(x),
            "input_shapes": [("B", 10)],
            "expected_output_shapes": [
                (2,)
            ],  # Output rank is fixed, value depends on B
        },
    ],
)
class ShapePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jnp.shape to ONNX Shape."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.shape."""

        if len(node_inputs) != 1:
            raise ValueError(f"Shape expects 1 input, got {len(node_inputs)}")
        if len(node_outputs) != 1:
            raise ValueError(f"Shape expects 1 output, got {len(node_outputs)}")

        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])  # Use get_name for output var

        # Create the ONNX Shape node
        shape_node = helper.make_node(
            "Shape",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("shape"),  # Unique node name
        )
        s.add_node(shape_node)

        # Register metadata for the output shape tensor (1D, rank, int64)
        input_aval = node_inputs[0].aval
        output_rank = len(input_aval.shape)
        output_shape_tuple = (output_rank,)
        # ONNX Shape outputs INT64
        output_dtype_enum = onnx.TensorProto.INT64

        # Register output metadata (ONNX Shape outputs a 1-D int64 tensor)
        s.builder.register_value_info_metadata(
            output_name, shape=output_shape_tuple, dtype=output_dtype_enum
        )

    # --- Monkey Patching Setup ---
    # This tells the plugin system to replace jnp.shape with our patched version
    # that uses the shape_p primitive.

    @staticmethod
    def _patched_jnp_shape(a):
        # Bind the primitive to the input operand
        # This ensures jax.make_jaxpr records our custom primitive
        return shape_p.bind(a)

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],  # Target module is jax.numpy
            "patch_function": lambda _: ShapePlugin._patched_jnp_shape,  # The function to patch with
            "target_attribute": "shape",  # The attribute name in the target module
        }


_original_jnp_shape = jnp.shape
jnp.shape = ShapePlugin._patched_jnp_shape
