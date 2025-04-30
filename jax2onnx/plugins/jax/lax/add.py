from typing import TYPE_CHECKING

import jax
from onnx import helper, TensorProto  # <-- Import TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[
        {
            "component": "Add",
            "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="add",
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x1, x2: x1 + x2,
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class AddPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.add to ONNX.
    """

    # --- abstract_eval (if needed, otherwise inherit/omit) ---
    # @staticmethod
    # def abstract_eval(input0, input1):
    #     # JAX handles abstract eval for basic ops
    #     # You might need this if behavior diverges significantly
    #     pass

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX add primitive."""
        # --- Setup ---
        input0_v, input1_v = node_inputs
        output_v = node_outputs[0]
        input0_name = s.get_name(input0_v)
        input1_name = s.get_name(input1_v)
        # Use get_var_name for output JAX variable
        output_name = s.get_var_name(output_v)

        # --- Determine ONNX Input Types (Enums) ---
        # Ensure metadata exists for inputs before lookup
        # This might require checking if they are constants first
        try:
            _, input0_dtype_enum = s.builder.get_shape_dtype(input0_name)
        except ValueError:
            # Handle cases where input might be a new constant not yet fully registered?
            # Fallback or raise error - for now, assume inputs are registered.
            # A safer approach might involve checking input_v.aval.dtype first
            # and converting via s._ensure_onnx_dtype if metadata lookup fails.
            raise ValueError(f"Metadata not found for Add input: {input0_name}")
        try:
            _, input1_dtype_enum = s.builder.get_shape_dtype(input1_name)
        except ValueError:
            raise ValueError(f"Metadata not found for Add input: {input1_name}")

        # --- Determine Expected ONNX Output Type (based on ONNX Add spec) ---
        # ONNX Add promotes types. Simplified rules:
        if (
            input0_dtype_enum == TensorProto.DOUBLE
            or input1_dtype_enum == TensorProto.DOUBLE
        ):
            onnx_output_dtype_enum = TensorProto.DOUBLE
        elif (
            input0_dtype_enum == TensorProto.FLOAT
            or input1_dtype_enum == TensorProto.FLOAT
        ):
            onnx_output_dtype_enum = TensorProto.FLOAT
        elif (
            input0_dtype_enum == TensorProto.UINT64
            or input1_dtype_enum == TensorProto.UINT64
        ):
            onnx_output_dtype_enum = TensorProto.UINT64
        elif (
            input0_dtype_enum == TensorProto.INT64
            or input1_dtype_enum == TensorProto.INT64
        ):
            onnx_output_dtype_enum = TensorProto.INT64  # <<< Ensures INT64 output
        elif (
            input0_dtype_enum == TensorProto.UINT32
            or input1_dtype_enum == TensorProto.UINT32
        ):
            onnx_output_dtype_enum = TensorProto.UINT32
        elif (
            input0_dtype_enum == TensorProto.INT32
            or input1_dtype_enum == TensorProto.INT32
        ):
            # If neither is INT64, output is INT32
            onnx_output_dtype_enum = TensorProto.INT32
        # Add other types as needed (FLOAT16, BOOL not valid for Add)
        else:
            # Fallback if types are unusual (e.g., INT8/16 - check ONNX spec for Add constraints)
            # Defaulting to first input might be risky.
            # It's better to ensure inputs are compatible types beforehand if possible.
            onnx_output_dtype_enum = input0_dtype_enum

        # --- Create the Add node ---
        node = helper.make_node(
            "Add",
            inputs=[input0_name, input1_name],
            outputs=[output_name],
            name=s.get_unique_name("add"),
        )
        s.add_node(node)

        # *** FIX: Add shape info using the determined ONNX output dtype ***
        s.add_shape_info(
            output_name,
            output_v.aval.shape,  # Get shape from JAXPR output variable
            onnx_output_dtype_enum,  # Use the calculated ONNX enum
        )
