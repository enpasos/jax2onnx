# file: jax2onnx/plugins/flax/nnx/linear_general.py
"""
Linear General Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.LinearGeneral layers to ONNX format.
It transforms JAX’s linear_general operations (a specialized dot_general for linear layers)
into an ONNX Gemm operator with necessary Reshape operations.

The conversion process involves:
  1. Calculating the output shape and the reshaping parameters.
  2. Providing an abstract evaluation for JAX's tracing system.
  3. Converting the operation to ONNX using Gemm and Reshape nodes.
  4. Monkey-patching LinearGeneral.__call__ to redirect calls to our primitive.
"""

from typing import TYPE_CHECKING

import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the primitive for linear_general operations.
nnx.linear_general_p = Primitive("nnx.linear_general")


@register_primitive(
    jaxpr_primitive=nnx.linear_general_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.LinearGeneral",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="linear_general",
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
class LinearGeneralPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.LinearGeneral to ONNX.

    Converts a LinearGeneral operation into a Gemm (matrix multiplication)
    followed by a Reshape to recover the desired output shape.
    """

    @staticmethod
    def _normalize_contracting_dims(dimension_numbers, x_shape, kernel_shape):
        # Unpack and normalize contracting dimensions to positive indices.
        ((lhs_contract, rhs_contract), _) = dimension_numbers
        lhs_contract = [d % len(x_shape) for d in lhs_contract]
        rhs_contract = [d % len(kernel_shape) for d in rhs_contract]
        return lhs_contract, rhs_contract

    @staticmethod
    def _compute_batch_and_kernel_output_dims(
        x_shape, kernel_shape, lhs_contract, rhs_contract
    ):
        # Compute sizes for batch dimensions from input and non-contracted (output) dimensions from kernel.
        x_batch_dims = [i for i in range(len(x_shape)) if i not in lhs_contract]
        x_batch_dims_sizes = [x_shape[i] for i in x_batch_dims]
        kernel_noncontract_dims = [
            i for i in range(len(kernel_shape)) if i not in rhs_contract
        ]
        kernel_out_dims = [kernel_shape[i] for i in kernel_noncontract_dims]
        return x_batch_dims_sizes, kernel_out_dims

    @staticmethod
    def _shape_linear_general(x_shape, kernel_shape, dimension_numbers):
        """Calculate all reshaping parameters for the Gemm transformation."""
        lhs_contract, rhs_contract = LinearGeneralPlugin._normalize_contracting_dims(
            dimension_numbers, x_shape, kernel_shape
        )
        x_batch_dims_sizes, kernel_out_dims = (
            LinearGeneralPlugin._compute_batch_and_kernel_output_dims(
                x_shape, kernel_shape, lhs_contract, rhs_contract
            )
        )

        output_shape = tuple(x_batch_dims_sizes + kernel_out_dims)
        new_kernel_dims_sizes = (
            np.prod([kernel_shape[i] for i in rhs_contract]).item(),
            np.prod(kernel_out_dims).item(),
        )
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
    def _is_noop_reshape(original_shape, target_shape):
        # A reshape is a no-op if the shapes are identical except possibly the batch dimension.
        return (
            len(original_shape) == len(target_shape)
            and original_shape[1:] == target_shape[1:]
        )

    @staticmethod
    def abstract_eval(x, kernel, bias, dimension_numbers):
        """Abstract evaluation: computes output shape and dtype."""
        shapes = LinearGeneralPlugin._shape_linear_general(
            x.shape, kernel.shape, dimension_numbers
        )
        return core.ShapedArray(shapes["output"], x.dtype)

    def to_onnx(
        self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, dimension_params
    ):
        """Convert linear_general operation to ONNX format."""
        input_var, kernel_var, bias_var = node_inputs[:3]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var) if bias_var else None

        shape_info = LinearGeneralPlugin._shape_linear_general(
            input_var.aval.shape,
            kernel_var.aval.shape,
            dimension_params["dimension_numbers"],
        )
        output_shape = shape_info["output"]
        new_kernel_shape = shape_info["new_kernel"]
        input_gemm_shape = shape_info["input_gemm"]
        output_gemm_shape = shape_info["output_gemm"]

        kernel_const = s.name_to_const[kernel_name]
        weights_name = s.get_constant_name(kernel_const.reshape(new_kernel_shape))

        target_input_shape = (-1,) + input_gemm_shape[1:]
        if LinearGeneralPlugin._is_noop_reshape(
            input_var.aval.shape, target_input_shape
        ):
            input_reshape_name = input_name
        else:
            input_reshape_name = s.get_unique_name("input_reshape")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        input_name,
                        s.get_constant_name(
                            np.array(target_input_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[input_reshape_name],
                    name=s.get_unique_name("reshape_input"),
                )
            )
            s.add_shape_info(input_reshape_name, input_gemm_shape)

        # Prepare bias: reshape if necessary or create zero bias.
        if bias_name is not None:
            bias_const = s.name_to_const[bias_name]
            target_bias_shape = (output_gemm_shape[1],)
            if bias_const.shape != target_bias_shape:
                bias_const = bias_const.reshape(target_bias_shape)
                bias_name = s.get_constant_name(bias_const)
            gemm_inputs = [input_reshape_name, weights_name, bias_name]
        else:
            bias_shape = (output_gemm_shape[1],)
            zero_bias = np.zeros(bias_shape, dtype=input_var.aval.dtype)
            bias_name = s.get_constant_name(zero_bias)
            gemm_inputs = [input_reshape_name, weights_name, bias_name]

        gemm_output_name = (
            output_name
            if LinearGeneralPlugin._is_noop_reshape(output_gemm_shape, output_shape)
            else s.get_unique_name("gemm_output")
        )
        s.add_node(
            helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=[gemm_output_name],
                name=s.get_unique_name("gemm"),
            )
        )
        s.add_shape_info(gemm_output_name, output_gemm_shape)

        if gemm_output_name != output_name:
            target_output_shape = [-1] + list(output_shape[1:])
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        gemm_output_name,
                        s.get_constant_name(
                            np.array(target_output_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[output_name],
                    name=s.get_unique_name("reshape_output"),
                )
            )

    @staticmethod
    def _linear_general(x, kernel, bias, dimension_numbers):
        nnx.linear_general_p.multiple_results = False
        return nnx.linear_general_p.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    @staticmethod
    def linear_general(x, kernel, bias, dimension_numbers):
        """Binding function for linear_general."""
        return LinearGeneralPlugin._linear_general(x, kernel, bias, dimension_numbers)

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of LinearGeneral.__call__."""

        def patched_linear_general_call(self, x):
            contracting_dims = (
                (self.axis,) if isinstance(self.axis, int) else self.axis,
                tuple(range(len(self.in_features))),
            )
            dimension_numbers = (contracting_dims, ((), ()))
            return LinearGeneralPlugin._linear_general(
                x,
                self.kernel.value,
                self.bias.value if self.bias else None,
                dimension_numbers,
            )

        return patched_linear_general_call

    @staticmethod
    def patch_info():
        """Provides patching information."""
        return {
            "patch_targets": [nnx.LinearGeneral],
            "patch_function": lambda _: LinearGeneralPlugin.get_monkey_patch(),
        }


# Register abstract evaluation function.
nnx.linear_general_p.def_abstract_eval(LinearGeneralPlugin.abstract_eval)
