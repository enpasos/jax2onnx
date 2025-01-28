# file: jax2onnx/plugins/activation_functions.py
import onnx.helper as oh
import jax
import onnx

# Generic function to create ONNX nodes for activation functions
def build_generic_onnx_node(op_type, jax_inputs, input_names, onnx_graph, parameters=None):
    """
    Create a generic ONNX node for activation functions.

    Args:
        op_type: The type of ONNX operation (e.g., 'Relu', 'Sigmoid').
        jax_inputs: The input tensors in JAX format.
        input_names: The corresponding input names in ONNX format.
        onnx_graph: The ONNX graph being constructed.
        parameters: Additional parameters for the ONNX node (if any).
    """
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Compute the JAX output for the operation
    jax_outputs = [getattr(jax.nn, op_type.lower())(jax_inputs[0])]

    # Flatten input_names in case it's nested
    flat_input_names = [name for sublist in input_names for name in (sublist if isinstance(sublist, list) else [sublist])]

    outputs = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            op_type,
            inputs=flat_input_names,  # Use flattened input names
            outputs=outputs,
            name=node_name,
        )
    )


    onnx_graph.add_local_outputs(jax_outputs, outputs)
    return jax_outputs, outputs

# Relu
jax.nn.relu.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Relu', jax_inputs, input_names, onnx_graph, parameters)

# Sigmoid
jax.nn.sigmoid.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Sigmoid', jax_inputs, input_names, onnx_graph, parameters)

# Tanh
jax.nn.tanh.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Tanh', jax_inputs, input_names, onnx_graph, parameters)

# Softmax
jax.nn.softmax.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Softmax', jax_inputs, input_names, onnx_graph, parameters)

# LogSoftmax
def build_log_softmax_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    axis = next((param.get('axis', -1) for param in parameters if isinstance(param, dict)), -1)

    outputs = [f'{node_name}_output']

    jax_outputs = [jax.nn.log_softmax(jax_inputs[0], axis=axis)]

    onnx_graph.add_node(
        oh.make_node(
            'LogSoftmax',
            inputs=[input_names[0]],
            outputs=outputs,
            name=node_name,
            axis=axis,
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, outputs)
    return jax_outputs, outputs

jax.nn.log_softmax.build_onnx_node = build_log_softmax_onnx_node

# Leaky ReLU
def build_leaky_relu_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    alpha = next((param.get('alpha', 0.01) for param in parameters if isinstance(param, dict)), 0.01)

    jax_outputs = [jax.nn.leaky_relu(jax_inputs[0], alpha=alpha)]

    outputs = [f'{node_name}_output']
    onnx_graph.add_node(
        oh.make_node(
            'LeakyRelu',
            inputs=[input_names[0]],
            outputs=outputs,
            name=node_name,
            alpha=alpha,
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, outputs)
    return jax_outputs, outputs

jax.nn.leaky_relu.build_onnx_node = build_leaky_relu_onnx_node

# CELU
def build_celu_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    alpha = next((param.get('alpha', 1.0) for param in parameters if isinstance(param, dict)), 1.0)

    jax_outputs = [jax.nn.celu(jax_inputs[0], alpha=alpha)]

    outputs = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            'Celu',
            inputs=[input_names[0]],
            outputs=outputs,
            name=node_name,
            alpha=alpha,
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, outputs)
    return jax_outputs, outputs

jax.nn.celu.build_onnx_node = build_celu_onnx_node

# ELU
jax.nn.elu.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Elu', jax_inputs, input_names, onnx_graph, parameters)

# Softplus
jax.nn.softplus.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Softplus', jax_inputs, input_names, onnx_graph, parameters)

# Define test parameters for each activation function
def get_test_params():
    return [
        {
            "model_name": "relu",
            "model": lambda: lambda x: jax.nn.relu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.relu.build_onnx_node,
        },
        {
            "model_name": "sigmoid",
            "model": lambda: lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.sigmoid.build_onnx_node,
        },
        {
            "model_name": "tanh",
            "model": lambda: lambda x: jax.nn.tanh(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.tanh.build_onnx_node,
        },
        {
            "model_name": "softmax",
            "model": lambda: lambda x: jax.nn.softmax(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.softmax.build_onnx_node,
        },
        {
            "model_name": "log_softmax",
            "model": lambda: lambda x: jax.nn.log_softmax(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.log_softmax.build_onnx_node,
            "parameters": [{"axis": -1}],
        },
    ]
