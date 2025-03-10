# jax2onnx/converter/primitives/flax/nnx/linear_general.py

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

    # for now we assume x_shape has x_batch dims on the left
    # and x_feature_dims dimensions on the right

    ((lhs_contract, rhs_contract), _) = dimension_numbers

    # remove negative indices
    lhs_contract = [d % len(x_shape) for d in lhs_contract]
    rhs_contract = [d % len(kernel_shape) for d in rhs_contract]

    x_dims = len(x_shape)
    x_feature_dims = len(lhs_contract)
    x_batch_dims = x_dims - x_feature_dims

    # x_dims_size = np.prod(x_shape).item()
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
    # combine x_feature_dims_sizes and kernel_right_dims_size into the single output_shape tupel
    output_shape = tuple(x_batch_dims_sizes + [kernel_right_dims_size])

    return {
        "input": x_shape,
        "input_gemm": input_gemm_shape,
        "output_gemm": output_gemm_shape,
        "output": output_shape,
        "new_kernel": new_kernel_dims_sizes,
    }


linear_general_p = Primitive("linear_general")


def get_primitive():
    return linear_general_p


def _get_monkey_patch():
    def linear_general(x, kernel, bias, dimension_numbers):

        def linear_general_abstract_eval(x, kernel, bias, dimension_numbers):
            shapes = _shape_linear_general(x.shape, kernel.shape, dimension_numbers)
            return core.ShapedArray(shapes["output"], x.dtype)

        linear_general_p.multiple_results = False  # Our operation has one output
        linear_general_p.def_abstract_eval(linear_general_abstract_eval)
        return linear_general_p.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    def patched_linear_general_call(self, x):
        contracting_dims = (self.axis, tuple(range(len(self.in_features))))
        batch_dims = ((), ())
        dimension_numbers = (contracting_dims, batch_dims)
        # Use the custom linear_general function
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

        # Add reshaped kernel weights to ONNX initializers
        kernel_const = s.name_to_const[kernel_name]
        weights_name = s.get_constant_name(kernel_const.reshape(new_kernel_shape))

        # First Reshape: flatten input tensor for GEMM operation
        input_reshape_name = s.get_unique_name("input_reshape")
        reshape_shape_input = tuple([-1] + list(input_gemm_shape[1:]))
        input_reshape_node = helper.make_node(
            "Reshape",
            inputs=[
                input_name,
                s.get_constant_name(np.array(reshape_shape_input, dtype=np.int64)),
            ],
            outputs=[input_reshape_name],
            name=s.get_unique_name("reshape_input"),
        )
        s.add_node(input_reshape_node)

        s.add_shape_info(input_reshape_name, input_gemm_shape)

        # GEMM operation for linear transformation
        gemm_output_name = s.get_unique_name("gemm_output")
        gemm_node = helper.make_node(
            "Gemm",
            inputs=[input_reshape_name, weights_name, bias_name],
            outputs=[gemm_output_name],
            name=s.get_unique_name("gemm"),
        )
        s.add_node(gemm_node)

        s.add_shape_info(gemm_output_name, output_gemm_shape)

        dynamic_output_shape = [-1] + list(output_shape[1:])

        reshape_output_node = helper.make_node(
            "Reshape",
            inputs=[
                gemm_output_name,
                s.get_constant_name(np.array(dynamic_output_shape, dtype=np.int64)),
            ],
            outputs=[output_name],
            name=s.get_unique_name("reshape_output"),
        )
        s.add_node(reshape_output_node)

    return handle_linear_general


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "linear_general",
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
            }
        ],
    }
