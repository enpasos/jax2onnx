# file: jax2onnx/plugins/flax/nnx/linear_general.py

"""Linear General Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.LinearGeneral layers to ONNX format.
It handles the transformation of JAX's linear_general operations (which are similar
to dot_general but specialized for linear layers) to ONNX's Gemm operator with
appropriate reshaping operations.

The plugin implements:
1. Shape calculation and transformation for linear_general operations
2. Abstract evaluation for JAX's tracing system
3. ONNX conversion using Gemm and Reshape operators
4. Monkey-patching mechanism to intercept LinearGeneral calls
"""

# Standard library imports
from typing import TYPE_CHECKING, Dict, List, Tuple, Any

# Third-party imports
import numpy as np
from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper

# Local imports
from jax2onnx.plugin_system import register_plugin, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the primitive that will represent nnx.linear_general operations
nnx.linear_general_p = Primitive("nnx.linear_general")


@register_plugin(
    jaxpr_primitive=nnx.linear_general_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
    onnx=[
        {
            "component": "Gemm",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.1.0",
    context="plugins.nnx",
    testcases=[
        {
            "testcase": "linear_general",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 4, 8, 32)],
        },
        {
            "testcase": "linear_general_2",
            "callable": nnx.LinearGeneral(
                in_features=(30,),
                out_features=(20,),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 30)],
        },
        {
            "testcase": "linear_general_3",
            "callable": nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 256)],
        },
        {
            "testcase": "linear_general_4",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
        },
    ],
)
class LinearGeneralPlugin(PrimitivePlugin):
    """Plugin for converting flax.nnx.LinearGeneral to ONNX.

    This plugin handles the conversion of LinearGeneral layers from JAX/Flax to ONNX format.
    LinearGeneral is a generalized linear transformation that supports contracting over
    arbitrary input dimensions and producing arbitrary output dimensions, making it more
    flexible than standard linear layers.

    The conversion process involves:
    1. Reshaping the input tensor to a 2D matrix
    2. Reshaping the kernel weights to a 2D matrix
    3. Using ONNX's Gemm operator for the matrix multiplication
    4. Reshaping the result back to the expected output dimensions
    """

    @staticmethod
    def _normalize_contracting_dims(dimension_numbers, x_shape, kernel_shape):
        """Normalize the contracting dimensions to positive indices."""
        ((lhs_contract, rhs_contract), _) = dimension_numbers
        lhs_contract = [d % len(x_shape) for d in lhs_contract]
        rhs_contract = [d % len(kernel_shape) for d in rhs_contract]
        return lhs_contract, rhs_contract

    @staticmethod
    def _compute_batch_and_output_dims(
        x_shape, kernel_shape, lhs_contract, rhs_contract
    ):
        """Compute batch dimensions from input and output dimensions from kernel."""
        x_batch_dims = [i for i in range(len(x_shape)) if i not in lhs_contract]
        x_batch_dims_sizes = [x_shape[i] for i in x_batch_dims]

        kernel_noncontract_dims = [
            i for i in range(len(kernel_shape)) if i not in rhs_contract
        ]
        kernel_out_dims = [kernel_shape[i] for i in kernel_noncontract_dims]

        return x_batch_dims_sizes, kernel_out_dims

    @staticmethod
    def _shape_linear_general(x_shape, kernel_shape, dimension_numbers):
        """Calculate shapes for linear general operation transformation."""
        lhs_contract, rhs_contract = LinearGeneralPlugin._normalize_contracting_dims(
            dimension_numbers, x_shape, kernel_shape
        )

        x_batch_dims_sizes, kernel_out_dims = (
            LinearGeneralPlugin._compute_batch_and_output_dims(
                x_shape, kernel_shape, lhs_contract, rhs_contract
            )
        )

        output_shape = tuple(x_batch_dims_sizes + kernel_out_dims)

        # Calculate reshaped kernel dimensions for Gemm operation
        new_kernel_dims_sizes = (
            np.prod([kernel_shape[i] for i in rhs_contract]).item(),
            np.prod(kernel_out_dims).item(),
        )

        # Calculate input and output shapes for Gemm operation
        input_gemm_shape = (
            np.prod(x_batch_dims_sizes).item(),
            np.prod([x_shape[i] for i in lhs_contract]).item(),
        )
        output_gemm_shape = (input_gemm_shape[0], new_kernel_dims_sizes[1])

        return {
            "input": x_shape,
            "input_gemm": input_gemm_shape,
            "output_gemm": output_gemm_shape,
            "output": output_shape,
            "new_kernel": new_kernel_dims_sizes,
        }

    @staticmethod
    def abstract_eval(x, kernel, bias, dimension_numbers):
        """Abstract evaluation function for linear_general.

        This method is called during JAX's abstract interpretation phase to determine
        the shape and dtype of the output without performing the actual computation.

        Args:
            x: ShapedArray representing the input tensor
            kernel: ShapedArray representing the weight matrix
            bias: ShapedArray representing the optional bias tensor or None
            dimension_numbers: Tuple specifying which dimensions to contract

        Returns:
            ShapedArray representing the output of the linear_general operation
        """
        shapes = LinearGeneralPlugin._shape_linear_general(
            x.shape, kernel.shape, dimension_numbers
        )
        return core.ShapedArray(shapes["output"], x.dtype)

    @staticmethod
    def to_onnx(
        converter: "Jaxpr2OnnxConverter",
        node_inputs,
        node_outputs,
        dimension_params,
    ):
        """Convert linear_general operation to ONNX format.

        The conversion process:
        1. Extract shapes and calculate transformations
        2. Reshape inputs to format compatible with Gemm
        3. Set up bias vector for Gemm operation
        4. Execute Gemm (matrix multiplication)
        5. Reshape output to final dimensions
        """
        # Extract inputs and outputs
        input_var, kernel_var, bias_var = node_inputs[:3]
        output_var = node_outputs[0]

        input_name = converter.get_name(input_var)
        output_name = converter.get_name(output_var)
        kernel_name = converter.get_name(kernel_var)
        bias_name = converter.get_name(bias_var) if bias_var else None

        # Calculate shapes for transformation
        shape_info = LinearGeneralPlugin._shape_linear_general(
            input_var.aval.shape,
            kernel_var.aval.shape,
            dimension_params["dimension_numbers"],
        )
        output_shape = shape_info["output"]
        new_kernel_shape = shape_info["new_kernel"]
        input_gemm_shape = shape_info["input_gemm"]
        output_gemm_shape = shape_info["output_gemm"]

        # Reshape kernel for Gemm operation
        kernel_const = converter.name_to_const[kernel_name]
        weights_name = converter.get_constant_name(
            kernel_const.reshape(new_kernel_shape)
        )

        # Reshape input to 2D format for Gemm operation if needed
        input_reshape_name = LinearGeneralPlugin._prepare_input_for_gemm(
            converter, input_var, input_name, input_gemm_shape
        )

        # Prepare bias for Gemm operation
        bias_name = LinearGeneralPlugin._prepare_bias_for_gemm(
            converter, bias_var, bias_name, output_gemm_shape, input_var.aval.dtype
        )

        gemm_inputs = [input_reshape_name, weights_name, bias_name]

        # Determine if we need an extra reshape after Gemm
        gemm_output_name = (
            output_name
            if (
                len(output_gemm_shape) == len(output_shape)
                and output_gemm_shape[1:] == output_shape[1:]
            )
            else converter.get_unique_name("gemm_output")
        )

        # Add Gemm operation
        converter.add_node(
            helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=[gemm_output_name],
                name=converter.get_unique_name("gemm"),
            )
        )
        converter.add_shape_info(gemm_output_name, output_gemm_shape)

        # Reshape output if necessary
        if gemm_output_name != output_name:
            LinearGeneralPlugin._reshape_gemm_output(
                converter, gemm_output_name, output_name, output_shape
            )

    @staticmethod
    def _prepare_input_for_gemm(converter, input_var, input_name, input_gemm_shape):
        """Reshape input tensor to 2D format required for Gemm operation."""
        target_input_shape = (-1,) + input_gemm_shape[1:]
        if not (
            len(input_var.aval.shape) == len(target_input_shape)
            and input_var.aval.shape[1:] == target_input_shape[1:]
        ):
            input_reshape_name = converter.get_unique_name("input_reshape")
            converter.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        input_name,
                        converter.get_constant_name(
                            np.array(target_input_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[input_reshape_name],
                    name=converter.get_unique_name("reshape_input"),
                )
            )
            converter.add_shape_info(input_reshape_name, input_gemm_shape)
            return input_reshape_name
        return input_name

    @staticmethod
    def _prepare_bias_for_gemm(
        converter, bias_var, bias_name, output_gemm_shape, dtype
    ):
        """Prepare bias tensor for Gemm operation, creating zeros if needed."""
        if bias_name is not None:
            bias_const = converter.name_to_const[bias_name]
            target_bias_shape = (output_gemm_shape[1],)
            if bias_const.shape != target_bias_shape:
                bias_const = bias_const.reshape(target_bias_shape)
                bias_name = converter.get_constant_name(bias_const)
        else:
            bias_shape = (output_gemm_shape[1],)
            zero_bias = np.zeros(bias_shape, dtype=dtype)
            bias_name = converter.get_constant_name(zero_bias)
        return bias_name

    @staticmethod
    def _reshape_gemm_output(converter, gemm_output_name, output_name, output_shape):
        """Reshape Gemm output to final required dimensions."""
        target_output_shape = [-1] + list(output_shape[1:])
        converter.add_node(
            helper.make_node(
                "Reshape",
                inputs=[
                    gemm_output_name,
                    converter.get_constant_name(
                        np.array(target_output_shape, dtype=np.int64)
                    ),
                ],
                outputs=[output_name],
                name=converter.get_unique_name("reshape_output"),
            )
        )

    @staticmethod
    def linear_general(x, kernel, bias, dimension_numbers):
        """Defines the primitive binding for linear_general.

        This function creates the binding between the JAX primitive and its implementation,
        ensuring that calls to LinearGeneral are intercepted and processed correctly.

        Args:
            x: Input tensor
            kernel: Weight matrix
            bias: Optional bias tensor
            dimension_numbers: Specifies which dimensions to contract

        Returns:
            Result of the linear general operation
        """
        nnx.linear_general_p.multiple_results = False
        return nnx.linear_general_p.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of LinearGeneral's call method.

        This function creates a replacement for the LinearGeneral.__call__ method
        that redirects to our primitive implementation, allowing us to capture and
        convert these operations during the ONNX conversion process.

        Returns:
            Function to replace the original LinearGeneral.__call__ method
        """

        def patched_linear_general_call(self, x):
            """Patched implementation of LinearGeneral.__call__."""
            # Convert axis to the format expected by dimension_numbers
            contracting_dims = (
                (self.axis,) if isinstance(self.axis, int) else self.axis,
                tuple(range(len(self.in_features))),
            )
            dimension_numbers = (contracting_dims, ((), ()))
            return LinearGeneralPlugin.linear_general(
                x,
                self.kernel.value,
                self.bias.value if self.bias else None,
                dimension_numbers,
            )

        return patched_linear_general_call

    @staticmethod
    def patch_info():
        """Provides information about what needs to be monkey patched.

        Returns:
            Dictionary with targets to patch and the function to apply
        """
        return {
            "patch_targets": [nnx.LinearGeneral],
            "patch_function": lambda _: LinearGeneralPlugin.get_monkey_patch(),
        }


# Register abstract evaluation function
nnx.linear_general_p.def_abstract_eval(LinearGeneralPlugin.abstract_eval)
