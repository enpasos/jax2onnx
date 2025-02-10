# file: jax2onnx/plugins/activation_functions.py

from functools import partial

import jax
import onnx.helper as oh

from jax2onnx.to_onnx import Z
from jax2onnx.typing_helpers import Supports2Onnx


def build_generic_onnx_node(
    op_type: str, jax_function: Supports2Onnx, z: Z, **params
) -> Z:
    """Creates an ONNX node for standard activation functions (e.g., ReLU, Sigmoid)."""
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]
    z.jax_function = jax_function

    node_name = f"node{onnx_graph.next_id()}"
    output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            op_type,
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    z.shapes = output_shapes
    z.names = output_names
    return z


# Attach ONNX conversion methods to JAX activation functions
jax.nn.relu.to_onnx = lambda *args: build_generic_onnx_node("Relu", jax.nn.relu, *args)
jax.nn.sigmoid.to_onnx = lambda *args: build_generic_onnx_node(
    "Sigmoid", jax.nn.sigmoid, *args
)
jax.nn.tanh.to_onnx = lambda *args: build_generic_onnx_node("Tanh", jax.nn.tanh, *args)
jax.nn.softmax.to_onnx = lambda *args: build_generic_onnx_node(
    "Softmax", jax.nn.softmax, *args
)
jax.nn.elu.to_onnx = lambda *args: build_generic_onnx_node("Elu", jax.nn.elu, *args)
jax.nn.softplus.to_onnx = lambda *args: build_generic_onnx_node(
    "Softplus", jax.nn.softplus, *args
)


def build_log_softmax_onnx_node(z: Z, **params) -> Z:
    """Creates an ONNX node for LogSoftmax with configurable axis."""
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    axis = params.get("axis", -1)
    output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "LogSoftmax",
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
            axis=axis,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    return Z(output_shapes, output_names, onnx_graph, jax_function=jax.nn.log_softmax)


jax.nn.log_softmax.to_onnx = build_log_softmax_onnx_node


def build_leaky_relu_onnx_node(z: Z, **params) -> Z:
    """Creates an ONNX node for LeakyReLU with configurable alpha."""
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    alpha = params.get("alpha", 0.01)
    output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "LeakyRelu",
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
            alpha=alpha,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    return Z(output_shapes, output_names, onnx_graph, jax_function=jax.nn.leaky_relu)


jax.nn.leaky_relu.to_onnx = build_leaky_relu_onnx_node


def build_gelu_onnx_node(z: Z, **params) -> Z:
    """Creates an ONNX node for GELU activation function."""
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "Gelu",
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)
    return Z(
        output_shapes,
        output_names,
        onnx_graph,
        jax_function=partial(jax.nn.gelu, approximate=False),
    )


jax.nn.gelu.to_onnx = build_gelu_onnx_node


def get_test_params():
    """Defines test parameters for verifying the ONNX conversion of activation functions."""
    return [
        {
            "testcase": "relu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.relu.to_onnx,
        },
        {
            "testcase": "sigmoid",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.sigmoid.to_onnx,
        },
        {
            "testcase": "tanh",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.tanh.to_onnx,
        },
        {
            "testcase": "softmax",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.softmax.to_onnx,
        },
        {
            "testcase": "log_softmax",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.log_softmax.to_onnx,
            "parameters": {"axis": -1},
        },
        {
            "testcase": "leaky_relu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.leaky_relu.to_onnx,
            "parameters": {"alpha": 0.02},
        },
        {
            "testcase": "elu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.elu.to_onnx,
        },
        {
            "testcase": "softplus",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.softplus.to_onnx,
        },
        {
            "testcase": "gelu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.gelu.to_onnx,
        },
        {
            "testcase": "gelu2",
            "input_shapes": [(1, 512)],
            "to_onnx": jax.nn.gelu.to_onnx,
        },
        {
            "testcase": "gelu3",
            "input_shapes": [(1, 10, 512)],
            "to_onnx": jax.nn.gelu.to_onnx,
        },
    ]
