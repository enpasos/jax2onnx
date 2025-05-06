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
        output_name = s.get_var_name(output_v)

        # --- Determine ONNX Input Types (Enums) ---
        # Try to get dtype from metadata, fallback to input_v.aval.dtype if missing
        def get_dtype_enum(name, var):
            try:
                _, dtype_enum = s.builder.get_shape_dtype(name)
                return dtype_enum
            except Exception:
                # Fallback: use the JAX dtype and convert to ONNX enum
                return s.builder._numpy_dtype_to_onnx(var.aval.dtype)

        input0_dtype_enum = get_dtype_enum(input0_name, input0_v)
        input1_dtype_enum = get_dtype_enum(input1_name, input1_v)

        # --- Determine Expected ONNX Output Type (based on ONNX Add spec) ---
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
            onnx_output_dtype_enum = TensorProto.INT64
        elif (
            input0_dtype_enum == TensorProto.UINT32
            or input1_dtype_enum == TensorProto.UINT32
        ):
            onnx_output_dtype_enum = TensorProto.UINT32
        elif (
            input0_dtype_enum == TensorProto.INT32
            or input1_dtype_enum == TensorProto.INT32
        ):
            onnx_output_dtype_enum = TensorProto.INT32
        else:
            onnx_output_dtype_enum = input0_dtype_enum

        # --- Create the Add node ---
        node = helper.make_node(
            "Add",
            inputs=[input0_name, input1_name],
            outputs=[output_name],
            name=s.get_unique_name("add"),
        )
        s.add_node(node)

        s.add_shape_info(
            output_name,
            output_v.aval.shape,
            onnx_output_dtype_enum,
        )
