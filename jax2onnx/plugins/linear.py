# file: jax2onnx/plugins/linear.py
import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx

def build_onnx_node(self, jax_inputs, input_names, onnx_graph):
    example_output = self(jax_inputs[0])
    output_shape = example_output.shape
    node1_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    node = oh.make_node(
        'Gemm',
        inputs=[input_names[0], f'{node1_name}_weight', f'{node1_name}_bias'],
        outputs=[f'{node1_name}_output'],
        name=node1_name,
    )
    onnx_graph.add_node(node)

    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node1_name}_weight",
            onnx.TensorProto.FLOAT,
            self.kernel.shape,
            self.kernel.value.reshape(-1).astype(np.float32)
        )
    )
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node1_name}_bias",
            onnx.TensorProto.FLOAT,
            [output_shape[-1]],
            self.bias.value.astype(np.float32)
        )
    )
    return node.output

nnx.Linear.build_onnx_node = build_onnx_node

def get_test_params():
    # Return a list of dictionaries
    return [
        {
            "model_name": "linear",
            "model": lambda: nnx.Linear(5, 3, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 5)],
            "build_onnx_node": nnx.Linear.build_onnx_node
        }
    ]

