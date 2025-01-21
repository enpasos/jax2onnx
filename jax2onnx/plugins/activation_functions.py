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

# LogSoftmax
def build_log_softmax_onnx_node(example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    # Extract axis from parameters or use default (last axis)
    axis = next((param['axis'] for param in parameters if 'axis' in param), -1)

    nodes.append(
        oh.make_node(
            'LogSoftmax',
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
            axis=axis,
        )
    )
    return f'{node_name}_output'

jax.nn.log_softmax.build_onnx_node = build_log_softmax_onnx_node


# LeakyRelu (requires alpha parameter)
def build_leaky_relu_onnx_node(example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    # Extract alpha from parameters or use default
    alpha = next((param['alpha'] for param in parameters if 'alpha' in param), 0.01)


    nodes.append(
        oh.make_node(
            'LeakyRelu',
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
            alpha=alpha
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

# CELU (requires alpha parameter)
def build_celu_onnx_node(example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    alpha = next((param['alpha'] for param in parameters if 'alpha' in param), 1.0)

    nodes.append(
        oh.make_node(
            'Celu',
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
            alpha=alpha,
        )
    )
    return f'{node_name}_output'

jax.nn.celu.build_onnx_node = build_celu_onnx_node

# ELU
def build_elu_onnx_node(example_input, input_name, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    nodes.append(
        oh.make_node(
            'Elu',
            inputs=[input_name],
            outputs=[f'{node_name}_output'],
            name=node_name,
        )
    )
    return f'{node_name}_output'

jax.nn.elu.build_onnx_node = build_elu_onnx_node

# def build_hard_sigmoid_onnx_node(example_input, input_name, nodes, parameters, counter):
#     node_name = f"node{counter[0]}"
#     counter[0] += 1
#
#     alpha = 0.2  # Slope
#     beta = 0.5   # Intercept
#
#     nodes.append(
#         oh.make_node(
#             'HardSigmoid',
#             inputs=[input_name],
#             outputs=[f'{node_name}_output'],
#             name=node_name,
#             alpha=alpha,
#             beta=beta,
#         )
#     )
#     return f'{node_name}_output'
#
# jax.nn.hard_sigmoid.build_onnx_node = build_hard_sigmoid_onnx_node




# Hard Swish (alias for Hard Sigmoid)
# jax.nn.hard_silu.build_onnx_node = jax.nn.hard_sigmoid.build_onnx_node
# jax.nn.hard_swish.build_onnx_node = jax.nn.hard_sigmoid.build_onnx_node

# Softplus
jax.nn.softplus.build_onnx_node = lambda example_input, input_name, nodes, parameters, counter: \
    build_generic_onnx_node('Softplus', example_input, input_name, nodes, parameters, counter)

# Softsign
jax.nn.soft_sign.build_onnx_node = lambda example_input, input_name, nodes, parameters, counter: \
    build_generic_onnx_node('Softsign', example_input, input_name, nodes, parameters, counter)

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
            "parameters": [{"alpha": 0.1}],
        },
        {
            "model_name": "gelu",
            "model": lambda: lambda x: jax.nn.gelu(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.gelu.build_onnx_node,
        },
        {
            "model_name": "celu",
            "model": lambda: lambda x: jax.nn.celu(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.celu.build_onnx_node,
            "parameters": [{"alpha": 1.0}],
        },
        {
            "model_name": "elu",
            "model": lambda: lambda x: jax.nn.elu(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.elu.build_onnx_node,
        },
        # {
        #     "model_name": "hard_sigmoid",
        #     "model": lambda: lambda x: jax.nn.hard_sigmoid(x),
        #     "input_shape": (1, 10),
        #     "build_onnx_node": jax.nn.hard_sigmoid.build_onnx_node,
        # }
        # ,
        {
            "model_name": "softplus",
            "model": lambda: lambda x: jax.nn.softplus(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.softplus.build_onnx_node,
        },
        {
            "model_name": "soft_sign",
            "model": lambda: lambda x: jax.nn.soft_sign(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.soft_sign.build_onnx_node,
        },
        {
            "model_name": "log_softmax",
            "model": lambda: lambda x: jax.nn.log_softmax(x),
            "input_shape": (1, 10),
            "build_onnx_node": jax.nn.log_softmax.build_onnx_node,
            "parameters": [{"axis": -1}],
        },

    ]
