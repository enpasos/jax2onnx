# file: jax2onnx/plugins/layernorm.py

import onnx
import onnx.helper as oh
from flax import nnx

from jax2onnx.convert import Z
from jax2onnx.typing_helpers import Supports2Onnx


def build_layernorm_onnx_node(self: Supports2Onnx, z: Z, **params) -> Z:
    """
    Converts an `nnx.LayerNorm` layer into an ONNX `LayerNormalization` node.

    Args:
        self: The LayerNorm instance.
        z (Z): Contains input shapes, names, and the ONNX graph.
        **params: Additional conversion parameters.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    # Generate a unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # Extract parameters
    feature_axes = self.feature_axes
    epsilon = self.epsilon
    use_bias = self.bias is not None
    use_scale = self.scale is not None

    # Define ONNX input names
    inputs = [input_name]

    if use_scale:
        scale_name = f"{node_name}_scale"
        inputs.append(scale_name)
        onnx_graph.add_initializer(
            oh.make_tensor(
                scale_name,
                onnx.TensorProto.FLOAT,
                self.scale.shape,
                self.scale.value.flatten(),
            )
        )

    if use_bias:
        bias_name = f"{node_name}_bias"
        inputs.append(bias_name)
        onnx_graph.add_initializer(
            oh.make_tensor(
                bias_name,
                onnx.TensorProto.FLOAT,
                self.bias.shape,
                self.bias.value.flatten(),
            )
        )

    # Define ONNX output names
    output_names = [f"{node_name}_output"]

    # Compute output shapes (LayerNorm does not alter shape)
    output_shapes = [input_shape]

    # Determine ONNX axis (feature_axes)
    axis = feature_axes if isinstance(feature_axes, int) else feature_axes[0]

    # Add the LayerNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=output_names,
            name=node_name,
            epsilon=epsilon,
            axis=axis,
        )
    )

    # Register the output tensor in the ONNX graph
    onnx_graph.add_local_outputs(output_shapes, output_names)

    # Update and return Z
    z.shapes = output_shapes
    z.names = output_names
    z.jax_function = self
    return z


# ✅ Correctly attach `to_onnx` method to `nnx.LayerNorm`
nnx.LayerNorm.to_onnx = build_layernorm_onnx_node


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.LayerNorm`.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "jax_component": "flax.nnx.LayerNorm",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm",
            "onnx": [
                {
                    "component": "LayerNormalization",
                    "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "layernorm_default",
                    "component": nnx.LayerNorm(64, rngs=nnx.Rngs(0)),
                    "input_shapes": [(1, 10, 64)],
                },
                {
                    "testcase": "layernorm_multiaxis",
                    "component": nnx.LayerNorm(
                        3 * 3 * 64,
                        reduction_axes=(1, 2, 3),
                        feature_axes=(1, 2, 3),
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 3, 3, 64)],
                    "params": {
                        "pre_transpose": [
                            (0, 3, 1, 2)
                        ],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                        "post_transpose": [
                            (0, 2, 3, 1)
                        ],  # Convert ONNX output back to JAX format
                    },
                },
            ],
        }
    ]
