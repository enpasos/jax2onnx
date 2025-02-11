# file: jax2onnx/plugins/concat.py

import jax.numpy as jnp
import onnx.helper as oh

from jax2onnx.to_onnx import Z


def build_add_onnx_node(z: Z, **params) -> Z:
    """Constructs an ONNX node for element-wise addition."""
    onnx_graph = z.onnx_graph
    input_shapes = z.shapes
    input_names = z.names

    node_name = f"node{onnx_graph.next_id()}"
    output_shapes = [input_shapes[0]]
    output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "Add",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
        )
    )

    onnx_graph.add_local_outputs(output_shapes, output_names)

    z.shapes = output_shapes
    z.names = output_names
    z.jax_function = jnp.add
    return z


# ✅ Wrap functions with lambdas to ensure correct argument passing
jnp.add.to_onnx = lambda z, **params: build_add_onnx_node(z, **params)


def get_test_params():
    return [
        {
            "jax_component": "jax.numpy.add",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html",
            "onnx": [
                {
                    "component": "Add",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "add",
                    "input_shapes": [(1, 10), (1, 10)],
                    "component": jnp.add,
                },
            ],
        }
    ]
