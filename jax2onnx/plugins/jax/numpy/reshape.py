# file: jax2onnx/plugins/reshape.py
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.helper as oh

from jax2onnx.convert import Z, OnnxGraph
from jax2onnx.convert import pre_transpose, post_transpose


def to_onnx_reshape(z: Z, **params) -> Z:
    """
    Converts `jax.numpy.reshape` into an ONNX `Reshape` node.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing the target shape or output shape.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    if isinstance(params, list):
        if not params or not isinstance(params[0], dict):
            raise ValueError(
                "Parameters for reshape must be a dictionary containing 'shape' or 'output_shape'."
            )
        params = params[0]  # Extract dictionary from list

    if "shape" not in params and "output_shape" not in params:
        raise ValueError("Parameters for reshape must include 'shape' or 'output_shape'.")

    input_name = z.names[0]
    input_shape = z.shapes[0]
    reshape_shape = params.get("shape", None)
    output_shape = params.get("output_shape", None)
    onnx_graph : OnnxGraph = z.onnx_graph
    jax_function = lambda x: jnp.reshape(x, reshape_shape or output_shape)


    # ✅ Apply pre-transpose if necessary
    z = pre_transpose(z, **params)
    input_name = z.names[0]
    input_shape = z.shapes[0]

    if reshape_shape and not output_shape:
        if not onnx_graph.dynamic_batch_dim:
            input = np.zeros(input_shape)
            output_shape = jax_function(input).shape
        else:
            input_shape_with_batch_1 = [1 if dim == 'B' else dim for dim in input_shape]
            output_shape = jax_function(np.zeros(input_shape_with_batch_1)).shape
            # replace the first dimension with 'B' again
            output_shape_with_batch_1_try = tuple([1] + list(output_shape)[1:])
            if np.prod(input_shape_with_batch_1) == np.prod(output_shape_with_batch_1_try):
                output_shape = tuple(['B'] + list(output_shape)[1:])
            else:
                output_shape = reshape_shape



    node_name = f"node{onnx_graph.next_id()}"
    
    output_names = [f"{node_name}_output"]

    # ✅ Create a shape tensor
    shape_tensor_name = f"{node_name}_shape"
    onnx_graph.add_initializer(
        oh.make_tensor(
            name=shape_tensor_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(reshape_shape or output_shape)],
            vals=reshape_shape or output_shape,
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
    output_shapes = [tuple(reshape_shape or output_shape)]

    # Register final output in ONNX graph
    onnx_graph.add_local_outputs([output_shape], output_names) 

    # ✅ Apply post-transpose if necessary
    z.shapes = output_shapes
    z.names = output_names
    z = post_transpose(z, **params)
    output_shapes = z.shapes

    # ✅ Store JAX function with dynamic shape handling
    z.jax_function = lambda x: jnp.reshape(x, tuple(reshape_shape or output_shape))
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
                    "testcase": "reshapeA",
                    "input_shapes": [(3,10,6)],
                    "component": jnp.reshape,
                    "params": {"shape": (-1, 6)},
                },
                {
                    "testcase": "reshapeB",
                    "input_shapes": [(3, 7, 6, 4)],
                    "component": jnp.reshape,
                    "params": {
                        "shape": (-1, 7, 24),  
                    },
                },
                {
                    "testcase": "reshapeC",
                    "input_shapes": [(1, 3, 224, 224)],
                    "component": jnp.reshape,
                    "params": {
                        "shape": (1, -1),  # Dynamic reshape to flatten feature maps
                    },
                }
            ],
        }
    ]
