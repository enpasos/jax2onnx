# file: jax2onnx/plugins/layernorm.py

# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html

import onnx
import onnx.helper as oh
from flax import nnx


def build_onnx(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for a LayerNorm operation.

    This function converts an `nnx.LayerNorm` layer into an ONNX `LayerNormalization` node,
    adding the scale and bias initializers if applicable.

    Args:
        self: The `nnx.LayerNorm` instance.
        input_shapes (list of tuples): List containing input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters, currently unused.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """

    # Extract input shape
    input_shape = input_shapes[0]

    # Generate a unique node name
    node_name = f"node{onnx_graph.counter_plusplus()}"


    # Extract parameters
    feature_axes = self.feature_axes
    epsilon = self.epsilon
    use_bias = self.bias is not None
    use_scale = self.scale is not None

    # Define ONNX input names
    inputs = [input_names[0]]

    if use_scale:
        scale_name = f"{node_name}_scale"
        inputs.append(scale_name)
        onnx_graph.add_initializer(
            oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, self.scale.shape, self.scale.value.flatten())
        )

    if use_bias:
        bias_name = f"{node_name}_bias"
        inputs.append(bias_name)
        onnx_graph.add_initializer(
            oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, self.bias.shape, self.bias.value.flatten())
        )

    # Define ONNX output names
    onnx_output_names = [f"{node_name}_output"]

    # Compute output shapes
    output_shapes = [input_shape]  # LayerNorm does not alter the shape

    # Determine ONNX axis (feature_axes)
    axis = feature_axes if isinstance(feature_axes, int) else feature_axes[0]

    # Add the LayerNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=onnx_output_names,
            name=node_name,
            epsilon=epsilon,
            axis=axis,
        )
    )

    # Register the output tensor in the ONNX graph
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


# Attach the `build_onnx` method to nnx.LayerNorm
nnx.LayerNorm.build_onnx = build_onnx


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.LayerNorm`.

    The test parameters define:
    - A simple `nnx.LayerNorm` model with input and output dimensions.
    - The corresponding input tensor shape.
    - The ONNX conversion function to be used in unit tests.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "layernorm_default",
            "model": lambda: nnx.LayerNorm(64, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 64)],
            "build_onnx": nnx.LayerNorm.build_onnx,
        },
        {
            "model_name": "layernorm_multiaxis",
            "model": lambda: nnx.LayerNorm(3 * 3 * 64, reduction_axes=(1, 2, 3), feature_axes=(1, 2, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 3, 3, 64)],
            "build_onnx": nnx.LayerNorm.build_onnx,
            "export": {
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
            }
        },
    ]
