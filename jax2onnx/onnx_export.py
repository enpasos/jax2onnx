# jax2onnx/onnx_export.py
import onnx
import onnx.helper as oh
import numpy as np
import jax
from flax import nnx

def count(counter):
    counter[0] += 1
    return counter[0]

def monkey_patches_for_onnx_export():
    def build_linear_onnx_node(self, example_input, input_name, nodes, parameters, counter):
        example_output = self(example_input)
        input_shape = example_input.shape
        output_shape = example_output.shape
        node1_name = f"node{count(counter)}"

        node = oh.make_node(
            'Gemm',
            inputs=[input_name, f'{node1_name}_weight', f'{node1_name}_bias'],
            outputs=[f'{node1_name}_output'],
            name=node1_name,
        )
        nodes.append(node)

        bias: nnx.Param = self.bias

        parameters.append(
            oh.make_tensor(
                f"{node1_name}_weight",
                onnx.TensorProto.FLOAT,
                self.kernel.shape,
                self.kernel.value.reshape(-1).astype(np.float32)
            )
        )
        parameters.append(
            oh.make_tensor(
                f"{node1_name}_bias",
                onnx.TensorProto.FLOAT,
                [output_shape[-1]],
                bias.value.astype(np.float32)
            )
        )
        return node.output[0]  # Use node.output instead of node.outputs

    def build_relu_onnx_node(example_input, input_name, nodes, parameters, counter):
        example_output = jax.nn.relu(example_input)
        input_shape = example_input.shape
        output_shape = example_output.shape
        node1_name = f"node{count(counter)}"
        nodes.append(
            oh.make_node(
                'Relu',
                inputs=[input_name],
                outputs=[f'{node1_name}_output'],
                name=node1_name,
            )
        )
        return f'{node1_name}_output'  # Return the output name

    nnx.Module.build_onnx_node = lambda self, example_input, nodes, parameters, counter: None
    nnx.Linear.build_onnx_node = build_linear_onnx_node
    nnx.Module.build_relu_onnx_node = build_relu_onnx_node

def export_to_onnx(model, example_input, output_path="model.onnx"):
    monkey_patches_for_onnx_export()

    example_output = model(example_input)
    input_shape = example_input.shape
    output_shape = example_output.shape

    input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)
    nodes = []
    initializers = []

    counter = [0]
    output_name = model.build_onnx_node(example_input, "input", nodes, initializers, counter)
    output_tensor = oh.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)

    graph_def = oh.make_graph(
        nodes,
        "NNXExportGraph",
        [input_tensor],
        [output_tensor],
        initializers
    )

    model_def = oh.make_model(graph_def, producer_name="nnx2onnx", opset_imports=[oh.make_operatorsetid("", 21)])
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")

