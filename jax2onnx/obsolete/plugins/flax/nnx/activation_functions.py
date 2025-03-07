# file: jax2onnx/plugins/activation_functions.py

from functools import partial
import onnx.helper as oh
import flax.nnx as nnx
from obsolete.convert import Z
from obsolete.typing_helpers import Supports2Onnx


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
nnx.relu.to_onnx = lambda *args: build_generic_onnx_node("Relu", nnx.relu, *args)
nnx.sigmoid.to_onnx = lambda *args: build_generic_onnx_node(
    "Sigmoid", nnx.sigmoid, *args
)
nnx.tanh.to_onnx = lambda *args: build_generic_onnx_node("Tanh", nnx.tanh, *args)
nnx.softmax.to_onnx = lambda *args: build_generic_onnx_node(
    "Softmax", nnx.softmax, *args
)
nnx.elu.to_onnx = lambda *args: build_generic_onnx_node("Elu", nnx.elu, *args)
nnx.softplus.to_onnx = lambda *args: build_generic_onnx_node(
    "Softplus", nnx.softplus, *args
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

    return Z(output_shapes, output_names, onnx_graph, jax_function=nnx.log_softmax)


nnx.log_softmax.to_onnx = build_log_softmax_onnx_node


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

    return Z(output_shapes, output_names, onnx_graph, jax_function=nnx.leaky_relu)


nnx.leaky_relu.to_onnx = build_leaky_relu_onnx_node


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
        jax_function=partial(nnx.gelu, approximate=False),
    )


nnx.gelu.to_onnx = build_gelu_onnx_node


def get_test_params():
    """Defines test parameters for verifying the ONNX conversion of activation functions."""
    return [
        {
            "jax_component": "flax.nnx.relu",
            "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.nn.relu.html#jax.nn.relu",
            "onnx": [
                {
                    "component": "Relu",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {"testcase": "relu", "component": nnx.relu, "input_shapes": [(1, 10)]}
            ],
        },
        {
            "jax_component": "flax.nnx.sigmoid",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.sigmoid",
            "onnx": [
                {
                    "component": "Sigmoid",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "sigmoid",
                    "component": nnx.sigmoid,
                    "input_shapes": [(1, 10)],
                }
            ],
        },
        {
            "jax_component": "flax.nnx.tanh",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh",
            "onnx": [
                {
                    "component": "Tanh",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {"testcase": "tanh", "component": nnx.tanh, "input_shapes": [(1, 10)]}
            ],
        },
        {
            "jax_component": "flax.nnx.softmax",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.softmax",
            "onnx": [
                {
                    "component": "Softmax",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "softmax",
                    "component": nnx.softmax,
                    "input_shapes": [(1, 10)],
                }
            ],
        },
        {
            "jax_component": "flax.nnx.log_softmax",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.log_softmax",
            "onnx": [
                {
                    "component": "LogSoftmax",
                    "doc": "https://onnx.ai/onnx/operators/onnx__LogSoftmax.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "log_softmax",
                    "component": nnx.log_softmax,
                    "input_shapes": [(1, 10)],
                    "parameters": {"axis": -1},
                }
            ],
        },
        {
            "jax_component": "flax.nnx.leaky_relu",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.leaky_relu",
            "onnx": [
                {
                    "component": "LeakyRelu",
                    "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "leaky_relu",
                    "component": nnx.leaky_relu,
                    "input_shapes": [(1, 10)],
                    "parameters": {"alpha": 0.02},
                }
            ],
        },
        {
            "jax_component": "flax.nnx.elu",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.elu",
            "onnx": [
                {
                    "component": "Elu",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Elu.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {"testcase": "elu", "component": nnx.elu, "input_shapes": [(1, 10)]}
            ],
        },
        {
            "jax_component": "flax.nnx.softplus",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.softplus",
            "onnx": [
                {
                    "component": "Softplus",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "softplus",
                    "component": nnx.softplus,
                    "input_shapes": [(1, 10)],
                }
            ],
        },
        {
            "jax_component": "flax.nnx.gelu",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.gelu",
            "onnx": [
                {
                    "component": "Gelu",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "gelu",
                    "component": nnx.gelu,
                    "input_shapes": [(1, 10)],
                },
                {
                    "testcase": "gelu2",
                    "component": nnx.gelu,
                    "input_shapes": [(1, 512)],
                },
                {
                    "testcase": "gelu3",
                    "component": nnx.gelu,
                    "input_shapes": [(1, 10, 512)],
                },
            ],
        },
    ]
