# file: jax2onnx/plugins/conv.py
import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx
from jax2onnx.onnx_export import OnnxGraph, jax_shape_to_onnx_shape

def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    """
    Build the ONNX node for a Conv operation.

    Args:
        self: The nnx.Conv instance.
        jax_inputs: List of input tensors in JAX format (e.g., `(B, H, W, C)`).
        input_names: List of corresponding input names in ONNX format.
        onnx_graph: The ONNX graph being constructed.
        parameters: Additional parameters (not used here).

    Returns:
        jax_outputs: The output tensors in JAX format.
        output_names: The corresponding output names in ONNX format.
    """
    # Compute the JAX output for reference
    jax_output = self(jax_inputs[0])

    # Convert the JAX output to ONNX format to extract the shape
    onnx_output_shape = jax_shape_to_onnx_shape(jax_output.shape)
    jax_input_shape = jax_inputs[0].shape

    # Generate a unique node name for the ONNX graph
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Handle padding for 'SAME' mode
    if self.padding == 'SAME':
        input_height, input_width = jax_input_shape[1], jax_input_shape[2]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        dilation_height, dilation_width = self.kernel_dilation if isinstance(self.kernel_dilation, tuple) else (self.kernel_dilation, self.kernel_dilation)

        pad_h = max((onnx_output_shape[2] - 1) * stride_height +
                    (kernel_height - 1) * dilation_height + 1 - input_height, 0)
        pad_w = max((onnx_output_shape[3] - 1) * stride_width +
                    (kernel_width - 1) * dilation_width + 1 - input_width, 0)
        pads = [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]
    else:
        pads = [0, 0, 0, 0]

    # Create the ONNX Conv node
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
            self.kernel.value.transpose(3, 2, 0, 1).flatten().astype(np.float32)
        )
    )

    # Add bias as an initializer if `use_bias` is True
    if self.use_bias:
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_bias",
                onnx.TensorProto.FLOAT,
                [self.out_features],
                self.bias.value.astype(np.float32)
            )
        )

    # Define ONNX output names and return both JAX outputs and ONNX output names
    output_names = [f"{node_name}_output"]
    jax_outputs = [jax_output]
    onnx_graph.add_local_outputs(jax_outputs, output_names)

    return jax_outputs, output_names

# Attach the `build_onnx_node` method to nnx.Conv
nnx.Conv.build_onnx_node = build_onnx_node

def get_test_params():
    """
    Define test parameters for Conv.
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
                rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(1, 64, 64, 3)],  # JAX shape: (B, H, W, C)
            "build_onnx_node": nnx.Conv.build_onnx_node
        }
    ]
