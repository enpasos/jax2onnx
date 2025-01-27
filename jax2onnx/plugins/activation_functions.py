import onnx.helper as oh
import jax

import onnx

# Generic function to create ONNX nodes for activation functions
def build_generic_onnx_node(op_type, jax_inputs, input_names, onnx_graph, parameters=None):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    outputs = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            op_type,
            inputs=input_names,
            outputs=outputs,
            name=node_name,
        )
    )
    return outputs

# Relu
jax.nn.relu.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Relu', jax_inputs, input_names, onnx_graph, parameters )

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

    onnx_graph.add_node(
        oh.make_node(
            'LogSoftmax',
            inputs=[input_names[0]],
            outputs=outputs,
            name=node_name,
            axis=axis,
        )
    )
    return outputs

jax.nn.log_softmax.build_onnx_node = build_log_softmax_onnx_node

# Leaky ReLU
def build_leaky_relu_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    alpha = next((param.get('alpha', 0.01) for param in parameters if isinstance(param, dict)), 0.01)

    alpha_node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()
    alpha_output = f"{alpha_node_name}_output"

    onnx_graph.add_node(
        oh.make_node(
            'Constant',
            inputs=[],
            outputs=[alpha_output],
            name=alpha_node_name,
            value=oh.make_tensor(
                name=alpha_node_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[alpha]
            )
        )
    )

    outputs = [f'{node_name}_output']
    onnx_graph.add_node(
        oh.make_node(
            'LeakyRelu',
            inputs=[input_names[0]],
            outputs=outputs,
            name=node_name,
        )
    )
    return outputs

jax.nn.leaky_relu.build_onnx_node = build_leaky_relu_onnx_node

# GELU
jax.nn.gelu.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Gelu', jax_inputs, input_names, onnx_graph, parameters)

# CELU
def build_celu_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    alpha = next((param.get('alpha', 1.0) for param in parameters if isinstance(param, dict)), 1.0)

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
    return outputs

jax.nn.celu.build_onnx_node = build_celu_onnx_node

# ELU
jax.nn.elu.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Elu', jax_inputs, input_names, onnx_graph, parameters)

# LogSigmoid
def build_log_sigmoid_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    negate_node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    softplus_node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    negate_output = f"{negate_node_name}_output"
    onnx_graph.add_node(
        oh.make_node(
            'Neg',
            inputs=[input_names[0]],
            outputs=[negate_output],
            name=negate_node_name,
        )
    )

    softplus_output = f"{softplus_node_name}_output"
    onnx_graph.add_node(
        oh.make_node(
            'Softplus',
            inputs=[negate_output],
            outputs=[softplus_output],
            name=softplus_node_name,
        )
    )

    final_negate_node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    final_output = f"{final_negate_node_name}_output"
    onnx_graph.add_node(
        oh.make_node(
            'Neg',
            inputs=[softplus_output],
            outputs=[final_output],
            name=final_negate_node_name,
        )
    )

    return [final_output]

jax.nn.log_sigmoid.build_onnx_node = build_log_sigmoid_onnx_node

# Softplus
jax.nn.softplus.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Softplus', jax_inputs, input_names, onnx_graph, parameters)

# Softsign
jax.nn.soft_sign.build_onnx_node = lambda jax_inputs, input_names, onnx_graph, parameters: \
    build_generic_onnx_node('Softsign', jax_inputs, input_names, onnx_graph, parameters)



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
            "model_name": "leaky_relu",
            "model": lambda: lambda x: jax.nn.leaky_relu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.leaky_relu.build_onnx_node,
            "parameters": [{"alpha": 0.1}],
        },
        {
            "model_name": "gelu",
            "model": lambda: lambda x: jax.nn.gelu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.gelu.build_onnx_node,
        },
        {
            "model_name": "celu",
            "model": lambda: lambda x: jax.nn.celu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.celu.build_onnx_node,
            "parameters": [{"alpha": 1.0}],
        },
        {
            "model_name": "elu",
            "model": lambda: lambda x: jax.nn.elu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.elu.build_onnx_node,
        },
        # {
        #     "model_name": "hard_sigmoid",
        #     "model": lambda: lambda x: jax.nn.hard_sigmoid(x),
        #     "input_shapes": (1, 10),
        #     "build_onnx_node": jax.nn.hard_sigmoid.build_onnx_node,
        # }
        # ,
        {
            "model_name": "softplus",
            "model": lambda: lambda x: jax.nn.softplus(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.softplus.build_onnx_node,
        },
        {
            "model_name": "soft_sign",
            "model": lambda: lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.soft_sign.build_onnx_node,
        },
        {
            "model_name": "log_softmax",
            "model": lambda: lambda x: jax.nn.log_softmax(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.log_softmax.build_onnx_node,
            "parameters": [{"axis": -1}],
        },
        {
            "model_name": "log_sigmoid",
            "model": lambda: lambda x: jax.nn.log_sigmoid(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.log_sigmoid.build_onnx_node,
        },

    ]
