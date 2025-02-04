# file: jax2onnx/onnx_export.py
import onnx
import onnx.helper as oh
import importlib
import pkgutil
import os
import numpy as np
import jax.numpy as jnp
from flax import nnx
from .transpose_utils import transpose_to_onnx, transpose_to_jax, jax_shape_to_onnx_shape, onnx_shape_to_jax_shape

class OnnxGraph:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.value_info = []
        self.counter = [0]

    def add_node(self, node):
        self.nodes.append(node)

    def add_initializer(self, initializer):
        self.initializers.append(initializer)


    def counter_plusplus(self):
        self.counter[0] += 1
        return self.counter[0]

    def add_local_outputs(self, output_shapes, output_names):
        for i in range(len(output_names)):
            self.value_info.append(oh.make_tensor_value_info(output_names[i], onnx.TensorProto.FLOAT, output_shapes[i]))


def load_plugins():
    plugins = {}
    package = "jax2onnx.plugins"
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        print(f"Loading plugin: {name}")
        module = importlib.import_module(f"{package}.{name}")
        plugins[name] = module
    return plugins

import jax

def export_to_onnx(model_file_name, model, input_shapes, output_path="model.onnx", build_onnx_node=None, parameters=None):
    if parameters is None:
        parameters = {}

     # Initialize the ONNX graph
    onnx_graph = OnnxGraph()

    input_names = [f"input_{onnx_graph.counter_plusplus()}" for i in range(len(input_shapes))]



    # Optional pre-transpose
    transposed_input_shapes, transposed_input_names = pre_transpose( input_shapes, input_names, onnx_graph, parameters)

    # Build ONNX node
    output_shapes, output_names = (
        model.build_onnx_node(transposed_input_shapes, transposed_input_names, onnx_graph, parameters)
        if hasattr(model, "build_onnx_node")
        else build_onnx_node(model, transposed_input_shapes, transposed_input_names, onnx_graph, parameters)
    )

    # extract into intermediate_output_tensors and remove outputs according to output_names from onnx value_info
    # intermediate_output_tensors = [value_info for value_info in onnx_graph.value_info if value_info.name in output_names]
    # onnx_graph.value_info = [value_info for value_info in onnx_graph.value_info if value_info.name not in output_names]


    # Optional post-transpose
    final_output_shapes, final_output_names = post_transpose(output_shapes, output_names, onnx_graph, parameters)


    # Remove outputs from onnx_graph.value_info
    onnx_graph.value_info = [value_info for value_info in onnx_graph.value_info if value_info.name not in final_output_names]


    # Define ONNX inputs and outputs
    inputs = [
        oh.make_tensor_value_info(input_names[i], onnx.TensorProto.FLOAT, input_shapes[i])
        for i in range(len(input_shapes))
    ]
    outputs = [
        oh.make_tensor_value_info(final_output_names[i], onnx.TensorProto.FLOAT, final_output_shapes[i])
        for i in range(len(final_output_names))
    ]

    # Create ONNX graph
    graph_def = oh.make_graph(onnx_graph.nodes, model_file_name, inputs, outputs, onnx_graph.initializers, value_info=onnx_graph.value_info)
    model_def = oh.make_model(graph_def, producer_name="jax2onnx", opset_imports=[oh.make_operatorsetid("", 21)])
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")

    return output_shapes


def pre_transpose( shapes, names, onnx_graph, parameters):
    transposed_input_names = names[:]
    shapes_out = shapes[:]
    if "pre_transpose" in parameters:
        pre_transpose_perm = parameters.get("pre_transpose", [(0, 3, 1, 2)])  # Default NHWC → NCHW
        parameters.pop("pre_transpose")
        transposed_input_names = [f"{name}_transposed_{onnx_graph.counter_plusplus()}" for name in names]
        for i in range(len(names)):
            onnx_graph.add_node(
                oh.make_node(
                    "Transpose",
                    inputs=[names[i]],
                    outputs=[transposed_input_names[i]],
                    perm=pre_transpose_perm[i],
                    name=f"transpose_input_{onnx_graph.counter_plusplus()}"
                )
            )
            x = jnp.zeros(shapes[i])
            shape = jnp.transpose(x, pre_transpose_perm[i]).shape
            shapes_out[i] = shape
        onnx_graph.add_local_outputs(shapes_out, transposed_input_names)
    return shapes_out, transposed_input_names

def post_transpose( output_shapes, output_names, onnx_graph, parameters):
    final_output_names = output_names[:]
    shapes = output_shapes[:]
    if "post_transpose" in parameters:
        post_transpose_perm = parameters.get("post_transpose", [(0, 2, 3, 1)])  # Default NCHW → NHWC
        parameters.pop("post_transpose")
        final_output_names = [f"{name}_transposed_{onnx_graph.counter_plusplus()}" for name in output_names]
        for i in range(len(output_names)):
            x = jnp.zeros(output_shapes[i])
            shape = jnp.transpose(x, post_transpose_perm[i]).shape
            shapes[i] = shape
            onnx_graph.add_node(
                oh.make_node(
                    "Transpose",
                    inputs=[output_names[i]],
                    outputs=[final_output_names[i]],
                    perm=post_transpose_perm[i],
                    name=f"transpose_output_{onnx_graph.counter_plusplus()}"
                )
            )
        onnx_graph.add_local_outputs(shapes, final_output_names)
    return shapes, final_output_names
