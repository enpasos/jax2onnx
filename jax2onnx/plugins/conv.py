# file: jax2onnx/plugins/conv.py

import jax.numpy as jnp
import onnx
import onnx.helper as oh
from flax import nnx

from transpose_utils import onnx_shape_to_jax_shape, jax_shape_to_onnx_shape


def to_onnx(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for a convolutional layer.

    This function converts an `nnx.Conv` layer into an ONNX `Conv` node,
    adding the kernel and bias initializers to the ONNX graph.

    Args:
        self: The `nnx.Conv` instance.
        input_shapes (list of tuples): List containing input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters, currently unused.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """

    input_shape = input_shapes[0]

    # assume that the input shape is (B, C, H, W) in the ONNX format

    # transform shape to JAX format (B, H, W, C)
    input_shape_jax = onnx_shape_to_jax_shape(input_shape)
    # construct jax_example_input
    jax_example_input = jnp.zeros(input_shape_jax)
    jax_example_output = self(jax_example_input)
    jax_example_output_shape = jax_example_output.shape
    output_shapes = [jax_shape_to_onnx_shape(jax_example_output_shape)]

    # Generate a unique node name
    node_name = f"node{onnx_graph.counter_plusplus()}"


    # Handle padding for 'SAME' mode
    if self.padding == 'SAME':
        input_height, input_width = input_shape[1], input_shape[2]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        dilation_height, dilation_width = self.kernel_dilation if isinstance(self.kernel_dilation, tuple) else (self.kernel_dilation, self.kernel_dilation)

        pad_h = max((input_shape[1] - 1) * stride_height +
                    (kernel_height - 1) * dilation_height + 1 - input_height, 0)
        pad_w = max((input_shape[2] - 1) * stride_width +
                    (kernel_width - 1) * dilation_width + 1 - input_width, 0)
        pads = [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]
    else:
        pads = [0, 0, 0, 0]

    # Define ONNX node for convolution
    conv_node = oh.make_node(
        'Conv',
        inputs=[input_names[0], f'{node_name}_weight'] + ([f'{node_name}_bias'] if self.use_bias else []),
        outputs=[f'{node_name}_output'],
        name=node_name,
        strides=list(self.strides) if isinstance(self.strides, tuple) else [self.strides, self.strides],
        dilations=list(self.kernel_dilation) if isinstance(self.kernel_dilation, tuple) else [self.kernel_dilation, self.kernel_dilation],
        pads=pads,
        group=self.feature_group_count,
    )
    onnx_graph.add_node(conv_node)

    # Add the kernel weights as an initializer (transposed to ONNX format)
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_weight",
            onnx.TensorProto.FLOAT,
            (self.out_features, self.in_features // self.feature_group_count, *self.kernel_size),
            self.kernel.value.transpose(3, 2, 0, 1).flatten().astype(jnp.float32),
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


    onnx_output_names = [f"{node_name}_output"]
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names

# Attach the `to_onnx` method to nnx.Conv
nnx.Conv.to_onnx = to_onnx

def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.Conv`.

    The test parameters define:
    - A simple `nnx.Conv` model with input and output dimensions.
    - The corresponding input tensor shape.
    - The ONNX conversion function to be used in unit tests.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "conv",
            "model": lambda: nnx.Conv(
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_dilation=(1, 1),
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 64, 64, 3)],  # JAX shape: (B, H, W, C)
            "to_onnx": nnx.Conv.to_onnx,
            "export": {
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
            }
        }
    ]
