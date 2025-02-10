# file: jax2onnx/plugins/conv.py

import jax.numpy as jnp
import onnx
import onnx.helper as oh
from flax import nnx

from jax2onnx.to_onnx import Z
from jax2onnx.transpose_utils import onnx_shape_to_jax_shape, jax_shape_to_onnx_shape
from jax2onnx.typing_helpers import Supports2Onnx  # Import protocol


def to_onnx(self: Supports2Onnx, z: Z, **params) -> Z:
    """
    Converts an `nnx.Conv` layer into an ONNX `Conv` node.

    Args:
        self: The `nnx.Conv` instance.
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Additional parameters (e.g., pre_transpose, post_transpose).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_names = z.names

    # Convert ONNX shape to JAX format (B, H, W, C)
    input_shape_jax = onnx_shape_to_jax_shape(input_shape)

    # Infer output shape using a dummy JAX input tensor
    jax_example_input = jnp.zeros(input_shape_jax)
    jax_example_output = self(jax_example_input)
    output_shape_jax = jax_example_output.shape

    # Convert back to ONNX format (B, C, H, W)
    output_shapes = [jax_shape_to_onnx_shape(output_shape_jax)]

    # Generate a unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # Handle padding for 'SAME' mode
    if self.padding == "SAME":
        input_height, input_width = input_shape[1], input_shape[2]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = (
            self.strides
            if isinstance(self.strides, tuple)
            else (self.strides, self.strides)
        )
        dilation_height, dilation_width = (
            self.kernel_dilation
            if isinstance(self.kernel_dilation, tuple)
            else (self.kernel_dilation, self.kernel_dilation)
        )

        pad_h = max(
            (input_shape[1] - 1) * stride_height
            + (kernel_height - 1) * dilation_height
            + 1
            - input_height,
            0,
        )
        pad_w = max(
            (input_shape[2] - 1) * stride_width
            + (kernel_width - 1) * dilation_width
            + 1
            - input_width,
            0,
        )
        pads = [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]
    else:
        pads = [0, 0, 0, 0]

    # Define ONNX Conv node
    conv_node = oh.make_node(
        "Conv",
        inputs=[input_names[0], f"{node_name}_weight"]
        + ([f"{node_name}_bias"] if self.use_bias else []),
        outputs=[f"{node_name}_output"],
        name=node_name,
        strides=(
            list(self.strides)
            if isinstance(self.strides, tuple)
            else [self.strides, self.strides]
        ),
        dilations=(
            list(self.kernel_dilation)
            if isinstance(self.kernel_dilation, tuple)
            else [self.kernel_dilation, self.kernel_dilation]
        ),
        pads=pads,
        group=self.feature_group_count,
    )
    onnx_graph.add_node(conv_node)

    # Add kernel weights as an initializer (transposed to ONNX format)
    kernel_onnx = self.kernel.value.transpose(3, 2, 0, 1).flatten().astype(jnp.float32)
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_weight",
            onnx.TensorProto.FLOAT,
            (
                self.out_features,
                self.in_features // self.feature_group_count,
                *self.kernel_size,
            ),
            kernel_onnx,
        )
    )

    # Add bias as an initializer if `use_bias` is True
    if self.use_bias:
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_bias",
                onnx.TensorProto.FLOAT,
                [self.out_features],
                self.bias.value.astype(jnp.float32),
            )
        )

    # Register ONNX output
    output_names = [f"{node_name}_output"]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    # Update and return Z
    z.shapes = output_shapes
    z.names = output_names
    return z


# Attach the `to_onnx` method to `nnx.Conv`
nnx.Conv.to_onnx = to_onnx


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.Conv`.
    """
    return [
        {
            "testcase": "conv",
            "component": nnx.Conv(
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_dilation=(1, 1),
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 64, 64, 3)],  # JAX shape: (B, H, W, C)
            "params": {
                "pre_transpose": [
                    (0, 3, 1, 2)
                ],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [
                    (0, 2, 3, 1)
                ],  # Convert ONNX output back to JAX format
            },
        }
    ]
