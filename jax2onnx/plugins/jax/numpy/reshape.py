# file: jax2onnx/plugins/reshape.py
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.helper as oh

from jax2onnx.to_onnx import Z
from jax2onnx.to_onnx import pre_transpose, post_transpose


def to_onnx_reshape(z: Z, **params) -> Z:
    """
    Converts `jax.numpy.reshape` into an ONNX `Reshape` node.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing the target shape.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    if isinstance(params, list):
        if not params or not isinstance(params[0], dict):
            raise ValueError(
                "Parameters for reshape must be a dictionary containing 'shape'."
            )
        params = params[0]  # Extract dictionary from list

    if "shape" not in params:
        raise ValueError("Parameters for reshape must include 'shape'.")

    target_shape = list(params["shape"])  # Convert to list for modification

    onnx_graph = z.onnx_graph
    input_shape = list(map(int, z.shapes[0]))  # Convert all values to Python int
    total_elements = np.prod(input_shape)

    # ✅ Compute the correct inferred dimension if `-1` is present
    inferred_dim_index = None
    known_size = 1
    for i, dim in enumerate(target_shape):
        if dim == -1:
            if inferred_dim_index is not None:
                raise ValueError(
                    "ONNX Reshape only allows one inferred dimension (-1)."
                )
            inferred_dim_index = i
        else:
            known_size *= int(dim)  # Ensure conversion to Python int

    # ✅ Replace `-1` with computed dimension
    if inferred_dim_index is not None:
        if total_elements % known_size != 0:
            raise ValueError(
                f"Cannot reshape {input_shape} to {target_shape} because sizes do not match."
            )
        target_shape[inferred_dim_index] = int(total_elements // known_size)

    # ✅ Ensure all values are Python `int`
    target_shape = [int(dim) for dim in target_shape]

    # ✅ Apply pre-transpose if necessary
    z = pre_transpose(z, **params)

    node_name = f"node{onnx_graph.next_id()}"
    input_name = z.names[0]
    output_names = [f"{node_name}_output"]

    # ✅ Create a shape tensor
    shape_tensor_name = f"{node_name}_shape"
    onnx_graph.add_initializer(
        oh.make_tensor(
            name=shape_tensor_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(target_shape)],
            vals=target_shape,
        )
    )

    # ✅ Add Reshape node
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[input_name, shape_tensor_name],
            outputs=output_names,
            name=node_name,
        )
    )

    # ✅ Compute output shapes
    output_shapes = [tuple(target_shape)]

    # Register final output in ONNX graph
    onnx_graph.add_local_outputs(output_shapes, output_names)

    # ✅ Apply post-transpose if necessary
    z.shapes = output_shapes
    z.names = output_names
    z = post_transpose(z, **params)

    # ✅ Store JAX function with dynamic shape handling
    z.jax_function = lambda x: jnp.reshape(x, tuple(target_shape))
    return z


# ✅ Attach ONNX conversion function to `jax.numpy.reshape`
jnp.reshape.to_onnx = to_onnx_reshape


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of reshape.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "jax_component": "jax.numpy.reshape",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html",
            "onnx": [
                {
                    "component": "Reshape",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "reshape",
                    "input_shapes": [(30,)],
                    "component": jnp.reshape,
                    "params": {"shape": (10, 3)},
                },
                {
                    "testcase": "reshape_dynamic",
                    "input_shapes": [(3, 7, 7, 64)],
                    "component": jnp.reshape,
                    "params": {
                        "shape": (3, -1),  # Dynamic reshape now correctly inferred
                    },
                },
                {
                    "testcase": "reshape_batch",
                    "input_shapes": [(1, 3, 224, 224)],
                    "component": jnp.reshape,
                    "params": {
                        "shape": (1, -1),  # Dynamic reshape to flatten feature maps
                    },
                },
                # {
                #     "testcase": "reshape_invalid",
                #     "input_shapes": [(3, 7, 7, 64)],
                #     "component": jnp.reshape,
                #     "params": {
                #         "shape": (3, 7),  # ❌ Should trigger an error
                #     },
                # },
            ],
        }
    ]
