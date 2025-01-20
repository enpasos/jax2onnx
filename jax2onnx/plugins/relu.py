# jax2onnx/plugins/relu.py
import onnx.helper as oh
import jax

def build_onnx_node(example_input, input_name, nodes, parameters, counter):
    node1_name = f"node{counter[0]}"
    counter[0] += 1

    nodes.append(
        oh.make_node(
            'Relu',
            inputs=[input_name],
            outputs=[f'{node1_name}_output'],
            name=node1_name,
        )
    )
    return f'{node1_name}_output'

jax.nn.relu.build_onnx_node = build_onnx_node
