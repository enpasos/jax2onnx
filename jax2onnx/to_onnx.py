# file: jax2onnx/to_onnx.py
import importlib
import os
import pkgutil

import jax.numpy as jnp
import onnx
import onnx.helper as oh


class OnnxGraph:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.value_info = []
        self.counter = 0

    def add_node(self, node):
        self.nodes.append(node)

    def add_initializer(self, initializer):
        self.initializers.append(initializer)

    def next_id(self):
        self.counter += 1
        return self.counter

    def add_local_outputs(self, output_shapes, output_names):
        for i in range(len(output_names)):
            self.value_info.append(
                oh.make_tensor_value_info(
                    output_names[i], onnx.TensorProto.FLOAT, output_shapes[i]
                )
            )


class Z:
    def __init__(self, shapes, names, onnx_graph, jax_function=None):
        self.shapes = shapes
        self.names = names
        self.onnx_graph = onnx_graph
        self.jax_function = jax_function

    def clone(self):
        return Z(self.shapes[:], self.names[:], self.onnx_graph, self.jax_function)

    def __add__(self, other):
        if not isinstance(other, Z):
            return NotImplemented

        return Z(
            self.shapes + other.shapes,
            self.names + other.names,
            self.onnx_graph,
            self.jax_function,
        )


def load_plugins():
    plugins = {}
    package = "jax2onnx.plugins"
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        print(f"Loading plugin: {name}")
        module = importlib.import_module(f"{package}.{name}")
        plugins[name] = module
    return plugins


def to_onnx(
    model_file_name,
    component,
    input_shapes,
    output_path="model.onnx",
    params=None,
):
    if params is None:
        params = {}

    # Initialize the ONNX graph
    onnx_graph = OnnxGraph()
    input_names = [f"input_{onnx_graph.next_id()}" for _ in input_shapes]

    z = Z(input_shapes, input_names, onnx_graph)

    # Optional pre-transpose
    z = pre_transpose(z, **params)

    # Convert the model to ONNX
    # call the with a copy of params that has not pre_transpose and post_transpose included
    params_no_transpose = {
        k: v for k, v in params.items() if k not in ["pre_transpose", "post_transpose"]
    }

    if hasattr(component, "to_onnx"):
        z = component.to_onnx(z, **params_no_transpose)
    else:
        raise ValueError(
            "Model does not have a `to_onnx` method and no conversion function was provided."
        )

    # Optional post-transpose
    z = post_transpose(z, **params)
    final_output_shapes = z.shapes
    final_output_names = z.names

    # Remove outputs from onnx_graph.value_info
    onnx_graph.value_info = [
        value_info
        for value_info in onnx_graph.value_info
        if value_info.name not in final_output_names
    ]

    # Define ONNX inputs and outputs
    inputs = [
        oh.make_tensor_value_info(
            input_names[i], onnx.TensorProto.FLOAT, input_shapes[i]
        )
        for i in range(len(input_shapes))
    ]
    outputs = [
        oh.make_tensor_value_info(
            final_output_names[i], onnx.TensorProto.FLOAT, final_output_shapes[i]
        )
        for i in range(len(final_output_names))
    ]

    # Create ONNX graph
    graph_def = oh.make_graph(
        onnx_graph.nodes,
        model_file_name,
        inputs,
        outputs,
        onnx_graph.initializers,
        value_info=onnx_graph.value_info,
    )
    model_def = oh.make_model(
        graph_def,
        producer_name="jax2onnx",
        opset_imports=[oh.make_operatorsetid("", 21)],
    )
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")

    return z


def pre_transpose(z, **params):
    """Applies a pre-transposition to ONNX inputs if specified in `params`."""
    shapes, names, onnx_graph = z.shapes, z.names, z.onnx_graph
    transposed_input_names, shapes_out = names[:], shapes[:]

    if "pre_transpose" in params:
        pre_transpose_perm = params.pop("pre_transpose", [(0, 3, 1, 2)])  # NHWC → NCHW

        transposed_input_names = [
            f"{name}_transposed_{onnx_graph.next_id()}" for name in names
        ]
        for i in range(len(names)):
            onnx_graph.add_node(
                oh.make_node(
                    "Transpose",
                    inputs=[names[i]],
                    outputs=[transposed_input_names[i]],
                    perm=pre_transpose_perm[i],
                    name=f"transpose_input_{onnx_graph.next_id()}",
                )
            )
            x = jnp.zeros(shapes[i])
            shapes_out[i] = jnp.transpose(x, pre_transpose_perm[i]).shape

        onnx_graph.add_local_outputs(shapes_out, transposed_input_names)
        z.shapes, z.names = shapes_out, transposed_input_names

    return z


def post_transpose(z, **params):
    """Applies a post-transposition to ONNX outputs if specified in `params`."""
    output_shapes, output_names, onnx_graph = z.shapes, z.names, z.onnx_graph
    final_output_names, shapes = output_names[:], output_shapes[:]

    if "post_transpose" in params:
        post_transpose_perm = params.pop(
            "post_transpose", [(0, 2, 3, 1)]
        )  # NCHW → NHWC

        final_output_names = [
            f"{name}_transposed_{onnx_graph.next_id()}" for name in output_names
        ]
        for i in range(len(output_names)):
            x = jnp.zeros(output_shapes[i])
            shapes[i] = jnp.transpose(x, post_transpose_perm[i]).shape
            onnx_graph.add_node(
                oh.make_node(
                    "Transpose",
                    inputs=[output_names[i]],
                    outputs=[final_output_names[i]],
                    perm=post_transpose_perm[i],
                    name=f"transpose_output_{onnx_graph.next_id()}",
                )
            )

        onnx_graph.add_local_outputs(shapes, final_output_names)
        z.shapes, z.names = shapes, final_output_names

    return z
