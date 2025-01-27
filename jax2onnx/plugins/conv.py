# jax2onnx/plugins/conv.py
import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx
from jax2onnx.onnx_export import OnnxGraph, jax_shape_to_onnx_shape, transpose_to_onnx

def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    # Convert JAX input to ONNX format
    example_onnx_input = transpose_to_onnx(jax_inputs[0])
    example_output = self(jax_inputs[0])  # Keep this as JAX output for consistency

    # Transpose the JAX output shape to ONNX format
    input_shape = example_onnx_input.shape
    output_shape = jax_shape_to_onnx_shape(example_output.shape)

    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Handle padding calculation
    if self.padding == 'SAME':
        pad_h = max((output_shape[2] - 1) * self.strides[0] +
                    (self.kernel_size[0] - 1) * self.kernel_dilation[0] + 1 - input_shape[2], 0)
        pad_w = max((output_shape[3] - 1) * self.strides[1] +
                    (self.kernel_size[1] - 1) * self.kernel_dilation[1] + 1 - input_shape[3], 0)
        pads = [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]
    else:
        pads = [0, 0, 0, 0]

    # Define Conv node with proper parameters
    conv_node = oh.make_node(
        'Conv',
        inputs=[input_names[0], f'{node_name}_weight'] + ([f'{node_name}_bias'] if self.use_bias else []),
        outputs=[f'{node_name}_output'],
        name=node_name,
        dilations=list(self.kernel_dilation),
        strides=list(self.strides),
        pads=pads,
        group=self.feature_group_count,
    )
    onnx_graph.add_node(conv_node)

    # Add kernel tensor (transpose weights to ONNX format)
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_weight",
            onnx.TensorProto.FLOAT,
            (self.out_features, self.in_features // self.feature_group_count, *self.kernel_size),
            np.transpose(self.kernel.value, axes=(3, 2, 0, 1)).reshape(-1).astype(np.float32)
        )
    )

    # Add bias tensor if applicable
    if self.use_bias:
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_bias",
                onnx.TensorProto.FLOAT,
                [self.out_features],
                self.bias.value.astype(np.float32)
            )
        )

    return [f"{node_name}_output"]

# Attach the build_onnx_node method to nnx.Conv
nnx.Conv.build_onnx_node = build_onnx_node

def get_test_params():
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
