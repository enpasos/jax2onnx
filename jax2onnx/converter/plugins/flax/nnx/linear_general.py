# file: jax2onnx/converter/plugins/flax/nnx/linear_general.py
import numpy as np
from jax import core
from jax.extend.core import Jaxpr, JaxprEqn, Var, Literal, ClosedJaxpr, Primitive
from onnx import helper
import contextlib
from flax import nnx
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def _shape_linear_general(x_shape, kernel_shape, dimension_numbers):
    # For now we assume x_shape has x_batch dims on the left
    # and x_feature_dims dimensions on the right
    ((lhs_contract, rhs_contract), _) = dimension_numbers

    # Remove negative indices.
    lhs_contract = [d % len(x_shape) for d in lhs_contract]
    rhs_contract = [d % len(kernel_shape) for d in rhs_contract]

    x_dims = len(x_shape)
    x_feature_dims = len(lhs_contract)
    x_batch_dims = x_dims - x_feature_dims

    x_feature_dims_sizes = [x_shape[d] for d in lhs_contract]
    x_feature_dims_size = np.prod(x_feature_dims_sizes).item()
    x_batch_dims_sizes = [
        x_shape[d] for d in range(len(x_shape)) if d not in lhs_contract
    ]
    x_batch_dims_size = np.prod(x_batch_dims_sizes).item()

    kernel_dims_size = np.prod(kernel_shape).item()
    kernel_left_dims_size = np.prod([kernel_shape[d] for d in rhs_contract]).item()
    kernel_right_dims_size = kernel_dims_size // kernel_left_dims_size

    new_kernel_dims_sizes = (kernel_left_dims_size, kernel_right_dims_size)
    input_gemm_shape = (x_batch_dims_size, x_feature_dims_size)
    output_gemm_shape = (x_batch_dims_size, kernel_right_dims_size)
    output_shape = tuple(x_batch_dims_sizes + [kernel_right_dims_size])

    return {
        "input": x_shape,
        "input_gemm": input_gemm_shape,
        "output_gemm": output_gemm_shape,
        "output": output_shape,
        "new_kernel": new_kernel_dims_sizes,
    }


nnx.linear_general_p = Primitive("nnx.linear_general")


def get_primitive():
    return nnx.linear_general_p


def _get_monkey_patch():
    def linear_general(x, kernel, bias, dimension_numbers):
        def linear_general_abstract_eval(x, kernel, bias, dimension_numbers):
            shapes = _shape_linear_general(x.shape, kernel.shape, dimension_numbers)
            return core.ShapedArray(shapes["output"], x.dtype)

        nnx.linear_general_p.multiple_results = False  # Our operation has one output
        nnx.linear_general_p.def_abstract_eval(linear_general_abstract_eval)
        return nnx.linear_general_p.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    def patched_linear_general_call(self, x):
        contracting_dims = (self.axis, tuple(range(len(self.in_features))))
        batch_dims = ((), ())
        dimension_numbers = (contracting_dims, batch_dims)
        # Use the custom linear_general function.
        return linear_general(
            x, self.kernel.value, self.bias.value, dimension_numbers=dimension_numbers
        )

    return patched_linear_general_call


@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.LinearGeneral.__call__
    nnx.LinearGeneral.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.LinearGeneral.__call__ = original_call


def _is_noop_reshape(original_shape, target_shape):
    """Return True if target_shape is equivalent to original_shape,
    allowing for a dynamic (-1) in the first dimension."""
    if len(original_shape) != len(target_shape):
        return False
    # Assume target_shape[0] is -1 representing the batch dimension.
    return original_shape[1:] == target_shape[1:]


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_linear_general(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]
        kernel_var = node_inputs[1]
        bias_var = node_inputs[2]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var)

        shapes = _shape_linear_general(
            input_var.aval.shape, kernel_var.aval.shape, params["dimension_numbers"]
        )
        output_shape = shapes["output"]
        new_kernel_shape = shapes["new_kernel"]
        input_gemm_shape = shapes["input_gemm"]
        output_gemm_shape = shapes["output_gemm"]

        # Add reshaped kernel weights to ONNX initializers.
        kernel_const = s.name_to_const[kernel_name]
        weights_name = s.get_constant_name(kernel_const.reshape(new_kernel_shape))

        # First Reshape: flatten input tensor for GEMM operation.
        target_input_shape = tuple([-1] + list(input_gemm_shape[1:]))
        if _is_noop_reshape(input_var.aval.shape, target_input_shape):
            input_reshape_name = input_name
        else:
            input_reshape_name = s.get_unique_name("input_reshape")
            input_reshape_node = helper.make_node(
                "Reshape",
                inputs=[
                    input_name,
                    s.get_constant_name(np.array(target_input_shape, dtype=np.int64)),
                ],
                outputs=[input_reshape_name],
                name=s.get_unique_name("reshape_input"),
            )
            s.add_node(input_reshape_node)
            s.add_shape_info(input_reshape_name, input_gemm_shape)

        # Determine if the GEMM output requires a final reshape.
        target_output_shape = [-1] + list(output_shape[1:])
        if _is_noop_reshape(output_gemm_shape, tuple(target_output_shape)):
            # No reshape needed. Use output_name directly for the GEMM node.
            gemm_output_name = output_name
        else:
            gemm_output_name = s.get_unique_name("gemm_output")

        # GEMM operation for linear transformation.
        gemm_node = helper.make_node(
            "Gemm",
            inputs=[input_reshape_name, weights_name, bias_name],
            outputs=[gemm_output_name],
            name=s.get_unique_name("gemm"),
        )
        s.add_node(gemm_node)
        s.add_shape_info(gemm_output_name, output_gemm_shape)

        # Final Reshape: from GEMM output to the final output shape.
        if not _is_noop_reshape(output_gemm_shape, tuple(target_output_shape)):
            reshape_output_node = helper.make_node(
                "Reshape",
                inputs=[
                    gemm_output_name,
                    s.get_constant_name(np.array(target_output_shape, dtype=np.int64)),
                ],
                outputs=[output_name],
                name=s.get_unique_name("reshape_output"),
            )
            s.add_node(reshape_output_node)

    return handle_linear_general


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.linear_general",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
        "onnx": [
            {
                "component": "Gemm",
                "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
            },
            {
                "component": "Reshape",
                "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
            },
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
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
        ],
    }
