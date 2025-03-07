# file: jax2onnx/plugins/conv.py

import jax.numpy as jnp
import onnx
import onnx.helper as oh
from flax import nnx

from obsolete.convert import Z
from obsolete.transpose_utils import onnx_shape_to_jax_shape, jax_shape_to_onnx_shape
from obsolete.typing_helpers import Supports2Onnx  # Import protocol


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
    input_shape = z.shapes[0]  # ONNX shape: (B, C, H, W)
    input_names = z.names

    # Convert ONNX shape to JAX format (B, H, W, C)
    input_shape_jax = list(onnx_shape_to_jax_shape(input_shape))

    # Remember the original batch dimension value
    original_batch_dim = input_shape_jax[0]

    # Use a concrete batch dimension of 1 for dummy test input
    input_shape_jax[0] = 1

    # Compute expected output shape from JAX
    jax_example_input = jnp.zeros(input_shape_jax)
    jax_example_output = self(jax_example_input)
    output_shape_jax = list(jax_example_output.shape)

    # Restore the original batch dimension value
    input_shape_jax[0] = original_batch_dim
    output_shape_jax[0] = original_batch_dim

    # Convert back to ONNX format (B, C, H, W)
    output_shapes = [jax_shape_to_onnx_shape(tuple(output_shape_jax))]

    # Generate unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # Compute ONNX-compatible padding for "SAME"
    if self.padding == "SAME":
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = (
            self.strides
            if isinstance(self.strides, tuple)
            else (self.strides, self.strides)
        )
        input_height, input_width = input_shape[2], input_shape[3]  # ONNX (B, C, H, W)

        output_height = int(jnp.ceil(input_height / stride_height))
        output_width = int(jnp.ceil(input_width / stride_width))

        pad_h = max(
            (output_height - 1) * stride_height + kernel_height - input_height, 0
        )
        pad_w = max((output_width - 1) * stride_width + kernel_width - input_width, 0)

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

    # Add kernel weights as an initializer (transpose for ONNX format)
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

    # Add bias if `use_bias=True`
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

    # Update and return `Z`
    z.shapes = output_shapes
    z.names = output_names
    return z


# Attach `to_onnx` to `nnx.Conv`
nnx.Conv.to_onnx = to_onnx


def get_test_params():
    """Defines test parameters for `nnx.Conv` ONNX conversion, including `ConvEmbedding`."""
    return [
        {
            "jax_component": "flax.nnx.Conv",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Conv",
            "onnx": [
                {
                    "component": "Conv",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
                }
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "conv_3x3_1",
                    "component": nnx.Conv(
                        in_features=1,
                        out_features=32,
                        kernel_size=(3, 3),
                        padding="SAME",
                        use_bias=True,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 28, 28, 1)],  # JAX shape (B, H, W, C)
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
                        "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX → JAX
                    },
                },
                {
                    "testcase": "conv_3x3_2",
                    "component": nnx.Conv(
                        in_features=32,
                        out_features=64,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="SAME",
                        use_bias=True,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 28, 28, 32)],  # JAX shape (B, H, W, C)
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
                        "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX → JAX
                    },
                },
                {
                    "testcase": "conv_3x3_3",
                    "component": nnx.Conv(
                        in_features=64,
                        out_features=128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="SAME",
                        use_bias=True,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 28, 28, 64)],  # JAX shape (B, H, W, C)
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
                        "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX → JAX
                    },
                },
            ],
        }
    ]
