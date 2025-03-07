# file: jax2onnx/plugins/concat.py

import jax.numpy as jnp
import onnx.helper as oh

from obsolete.convert import Z


def build_concat_onnx_node(z: Z, **params) -> Z:
    """Constructs an ONNX node for concatenation along a specified axis."""
    if "axis" not in params:
        raise TypeError("Expected 'axis' parameter for concatenation.")

    axis = params["axis"]
    onnx_graph = z.onnx_graph
    input_shapes = z.shapes
    input_names = z.names

    node_name = f"node{onnx_graph.next_id()}"

    # Compute output shape
    output_shape = list(input_shapes[0])
    output_shape[axis] = sum(shape[axis] for shape in input_shapes)

    output_shapes = [tuple(output_shape)]
    output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "Concat",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            axis=axis,
        )
    )

    onnx_graph.add_local_outputs(output_shapes, output_names)

    z.shapes = output_shapes
    z.names = output_names
    z.jax_function = lambda *args: jnp.concatenate(args, axis=axis)
    return z


# ✅ Wrap functions with lambdas
jnp.concatenate.to_onnx = lambda z, **params: build_concat_onnx_node(z, **params)


def get_test_params():
    return [
        {
            "jax_component": "jax.numpy.concat",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.concat.html",
            "onnx": [
                {
                    "component": "Concat",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "concat",
                    "input_shapes": [(1, 10), (1, 10)],
                    "component": jnp.concatenate,
                    "params": {"axis": 1},
                },
            ],
        }
    ]
