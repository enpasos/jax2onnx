# file: jax2onnx/converter/plugins/flax/nnx/linear.py
import numpy as np
from jax import core
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from flax import nnx
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for linear.
nnx.linear_p = Primitive("nnx.linear")


def get_primitive():
    return nnx.linear_p


def _shape_linear(
    x_shape: Tuple[Union[int, str], ...],
    kernel_shape: Tuple[int, ...],
    dimension_numbers=None,
) -> dict:
    """
    Calculates the input/output shapes for a standard linear layer.

    For nnx.Linear we assume:
      - x has shape (batch, in_features) (or higher-rank with batch dims preceding feature dim)
      - kernel has shape (in_features, out_features)

    If dimension_numbers is not provided, we assume the standard contraction:
      - Contract on the last dimension of x and the first dimension of kernel.
    """
    if len(x_shape) < 1:
        raise ValueError("Input must have at least one dimension")

    # Default: contract last dim of x with first dim of kernel.
    if dimension_numbers is None:
        lhs_contract = (len(x_shape) - 1,)  # last dimension of input
        rhs_contract = (0,)  # first dimension of kernel
        dimension_numbers = ((lhs_contract, rhs_contract), ((), ()))

    ((lhs_contract, rhs_contract), _) = dimension_numbers
    # Normalize negative indices.
    lhs_contract = (lhs_contract[0] % len(x_shape),)
    rhs_contract = (rhs_contract[0] % len(kernel_shape),)

    # The feature dimensions come from the contracted dimensions.
    x_feature_dims = lhs_contract
    # All other dimensions are considered batch dimensions.
    x_batch_dims = [d for d in range(len(x_shape)) if d not in lhs_contract]

    # Compute feature size.
    x_feature_sizes = [x_shape[d] for d in x_feature_dims]
    x_feature_size = (
        np.prod(x_feature_sizes).item()
        if all(isinstance(dim, int) for dim in x_feature_sizes)
        else x_shape[-1]
    )

    # Compute batch shape and size.
    x_batch_sizes = [x_shape[d] for d in x_batch_dims]
    dynamic_dim = None
    for dim in x_batch_sizes:
        if isinstance(dim, str):
            dynamic_dim = dim
            break
    x_batch_size = (
        dynamic_dim if dynamic_dim is not None else np.prod(x_batch_sizes).item()
    )

    # For a standard Linear, kernel shape is (in_features, out_features)
    out_features = kernel_shape[1]
    new_kernel_shape = kernel_shape  # No reshaping needed.

    # The GEMM will operate on a flattened input of shape (batch, in_features)
    input_gemm_shape = (x_batch_size, x_feature_size)
    output_gemm_shape = (x_batch_size, out_features)
    output_shape = tuple(x_batch_sizes) + (out_features,)

    return {
        "input": x_shape,
        "input_gemm": input_gemm_shape,
        "output_gemm": output_gemm_shape,
        "output": output_shape,
        "new_kernel": new_kernel_shape,
    }


def _is_noop_reshape(original_shape, target_shape):
    """Return True if target_shape is equivalent to original_shape,
    allowing for a dynamic (-1) in the first dimension.
    """
    if len(original_shape) != len(target_shape):
        return False
    # Compare all dimensions except possibly the first.
    return original_shape[1:] == target_shape[1:]


def _get_monkey_patch():
    def linear(x, kernel, bias, dimension_numbers=None):
        # Set default dimension numbers for a standard linear layer.
        if dimension_numbers is None:
            lhs_contract = (-1,)  # last dim of x
            rhs_contract = (0,)  # first dim of kernel
            dimension_numbers = ((lhs_contract, rhs_contract), ((), ()))

        # Define abstract evaluation using our shape function.
        def linear_abstract_eval(x, kernel, bias, dimension_numbers):
            shapes = _shape_linear(x.shape, kernel.shape, dimension_numbers)
            return core.ShapedArray(shapes["output"], x.dtype)

        nnx.linear_p.multiple_results = False
        nnx.linear_p.def_abstract_eval(linear_abstract_eval)
        return nnx.linear_p.bind(x, kernel, bias, dimension_numbers=dimension_numbers)

    def patched_linear_call(self, x):
        # For nnx.Linear the standard contraction is last dim of input with first dim of kernel.
        lhs_contract = (-1,)
        rhs_contract = (0,)
        dimension_numbers = ((lhs_contract, rhs_contract), ((), ()))
        return linear(
            x, self.kernel.value, self.bias.value, dimension_numbers=dimension_numbers
        )

    return patched_linear_call


@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.Linear.__call__
    nnx.Linear.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.Linear.__call__ = original_call


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_linear(node_inputs, node_outputs, params):
        # node_inputs: [x, kernel, bias]
        x_var = node_inputs[0]
        kernel_var = node_inputs[1]
        bias_var = node_inputs[2]

        dimension_numbers = params.get("dimension_numbers")
        input_name = s.get_name(x_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var)
        output_name = s.get_name(node_outputs[0])

        shapes = _shape_linear(
            x_var.aval.shape, kernel_var.aval.shape, dimension_numbers
        )
        input_gemm_shape = shapes["input_gemm"]
        output_gemm_shape = shapes["output_gemm"]
        output_shape = shapes["output"]
        new_kernel_shape = shapes["new_kernel"]

        # Retrieve the kernel constant and reshape it if necessary.
        kernel_const = s.name_to_const[kernel_name]
        weights_name = s.builder.get_constant_name(
            kernel_const.reshape(new_kernel_shape)
        )

        # Determine target reshape shape for input.
        target_input_shape = tuple([-1] + list(input_gemm_shape[1:]))
        if _is_noop_reshape(x_var.aval.shape, target_input_shape):
            input_reshape_name = input_name
        else:
            input_reshape_name = s.get_unique_name("input_reshape")
            reshape_shape_input = np.array(target_input_shape, dtype=np.int64)
            reshape_shape_input_name = s.builder.get_constant_name(reshape_shape_input)
            reshape_input_node = helper.make_node(
                "Reshape",
                inputs=[input_name, reshape_shape_input_name],
                outputs=[input_reshape_name],
                name=s.get_unique_name("reshape_input"),
                allowzero=0,
            )
            s.add_node(reshape_input_node)
            s.add_shape_info(input_reshape_name, input_gemm_shape)

        # Gemm node.
        gemm_output_name = s.get_unique_name("gemm_output")
        gemm_node = helper.make_node(
            "Gemm",
            inputs=[input_reshape_name, weights_name, bias_name],
            outputs=[gemm_output_name],
            name=s.get_unique_name("gemm"),
        )
        s.add_node(gemm_node)
        s.add_shape_info(gemm_output_name, output_gemm_shape)

        # Final reshape: restore to the original output shape if necessary.
        target_output_shape = [-1] + list(output_shape[1:])
        if _is_noop_reshape(output_gemm_shape, tuple(target_output_shape)):
            # Instead of inserting an extra node, update the variable mapping and builder outputs.
            s.var_to_name[node_outputs[0]] = gemm_output_name
            # Update the corresponding output in the builder.
            for i, out in enumerate(s.builder.outputs):
                if out.name == output_name:
                    s.builder.outputs[i] = helper.make_tensor_value_info(
                        gemm_output_name,
                        out.type.tensor_type.elem_type,
                        [dim.dim_value for dim in out.type.tensor_type.shape.dim],
                    )
                    break
        else:
            reshape_output_shape = np.array(target_output_shape, dtype=np.int64)
            reshape_output_shape_name = s.builder.get_constant_name(
                reshape_output_shape
            )
            reshape_output_node = helper.make_node(
                "Reshape",
                inputs=[gemm_output_name, reshape_output_shape_name],
                outputs=[output_name],
                name=s.get_unique_name("reshape_output"),
                allowzero=0,
            )
            s.add_node(reshape_output_node)

    return handle_linear


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "linear",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html",
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
                "testcase": "linear_2d",
                "callable": nnx.Linear(
                    in_features=128,
                    out_features=64,
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [
                    (32, 10, 128)
                ],  # Higher-rank input with batch dims (32,10)
            },
            {
                "testcase": "linear",
                "callable": nnx.Linear(
                    in_features=128,
                    out_features=64,
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [("B", 128)],
            },
        ],
    }
