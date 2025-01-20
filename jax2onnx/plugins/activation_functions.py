# jax2onnx/plugins/activation_functions.py
import onnx.helper as oh
import jax

# Generic function to create ONNX nodes for activation functions
def build_generic_onnx_node(op_type, example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    nodes.append(
        oh.make_node(
            op_type,
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
        )
    )
    return f'{node_name}_output'

# Relu
jax.nn.relu.build_onnx_node = lambda example_input, input_name, nodes, parameters, counter: \
    build_generic_onnx_node('Relu', example_input, input_name, nodes, parameters, counter)

# Sigmoid
jax.nn.sigmoid.build_onnx_node = lambda example_input, input_name, nodes, parameters, counter: \
    build_generic_onnx_node('Sigmoid', example_input, input_name, nodes, parameters, counter)

# Tanh
jax.nn.tanh.build_onnx_node = lambda example_input, input_name, nodes, parameters, counter: \
    build_generic_onnx_node('Tanh', example_input, input_name, nodes, parameters, counter)

# Softmax
jax.nn.softmax.build_onnx_node = lambda example_input, input_name, nodes, parameters, counter: \
    build_generic_onnx_node('Softmax', example_input, input_name, nodes, parameters, counter)

# LeakyRelu (requires alpha parameter)
def build_leaky_relu_onnx_node(example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    alpha = parameters.get('alpha', 0.01)  # Default alpha value for LeakyRelu
    nodes.append(
        oh.make_node(
            'LeakyRelu',
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
            alpha=alpha,
        )
    )
    return f'{node_name}_output'

jax.nn.leaky_relu.build_onnx_node = build_leaky_relu_onnx_node

# GELU
def build_gelu_onnx_node(example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    nodes.append(
        oh.make_node(
            'Gelu',
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
        )
    )
    return f'{node_name}_output'

jax.nn.gelu.build_onnx_node = build_gelu_onnx_node

# Define test parameters for each activation function
def get_test_params():
    return [
        {
            "model_name": "relu",
            "model": lambda: lambda x: jax.nn.relu(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.relu.build_onnx_node,
        },
        {
            "model_name": "sigmoid",
            "model": lambda: lambda x: jax.nn.sigmoid(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.sigmoid.build_onnx_node,
        },
        {
            "model_name": "tanh",
            "model": lambda: lambda x: jax.nn.tanh(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.tanh.build_onnx_node,
        },
        {
            "model_name": "softmax",
            "model": lambda: lambda x: jax.nn.softmax(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.softmax.build_onnx_node,
        },
        {
            "model_name": "leaky_relu",
            "model": lambda: lambda x: jax.nn.leaky_relu(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.leaky_relu.build_onnx_node,
        },
        {
            "model_name": "gelu",
            "model": lambda: lambda x: jax.nn.gelu(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.gelu.build_onnx_node,
        },
    ]
