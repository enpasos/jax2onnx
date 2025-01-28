# file: jax2onnx/onnx_export.py
import onnx
import onnx.helper as oh
import importlib
import pkgutil
import os
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

    def increment_counter(self):
        self.counter[0] += 1

    def get_counter(self):
        return self.counter[0]

    def add_local_outputs(self, jax_outputs, onnx_output_names):
        for i in range(len(onnx_output_names)):
            self.value_info.append(oh.make_tensor_value_info(onnx_output_names[i], onnx.TensorProto.FLOAT, jax_shape_to_onnx_shape(jax_outputs[i].shape)))



def load_plugins():
    plugins = {}
    package = "jax2onnx.plugins"
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        print(f"Loading plugin: {name}")
        module = importlib.import_module(f"{package}.{name}")
        plugins[name] = module
    return plugins

def export_to_onnx(model_file_name, model, jax_inputs, output_path="model.onnx", build_onnx_node=None, parameters=None):
    if parameters is None:
        parameters = {}

    # Convert input shapes to ONNX format
    input_shapes = [jax_shape_to_onnx_shape(jax_input.shape) for jax_input in jax_inputs]



    input_names = [f"input_{i}" for i in range(len(jax_inputs))]

    # Initialize the ONNX graph
    onnx_graph = OnnxGraph()

    # Build ONNX node
    jax_outputs, output_names = (
        model.build_onnx_node(jax_inputs, input_names, onnx_graph, parameters)
        if hasattr(model, "build_onnx_node")
        else build_onnx_node(jax_inputs, input_names, onnx_graph, parameters)
    )


    # Define ONNX inputs and outputs
    inputs = [
        oh.make_tensor_value_info(input_names[i], onnx.TensorProto.FLOAT, input_shapes[i])
        for i in range(len(jax_inputs))
    ]
    outputs = [
        oh.make_tensor_value_info(output_names[i], onnx.TensorProto.FLOAT, jax_shape_to_onnx_shape(jax_outputs[i].shape))
        for i in range(len(output_names))
    ]
    # remove outputs from onnx_graph.value_info
    onnx_graph.value_info = [value_info for value_info in onnx_graph.value_info if value_info.name not in output_names]

    # Create ONNX graph
    graph_def = oh.make_graph(onnx_graph.nodes, model_file_name, inputs, outputs, onnx_graph.initializers, value_info = onnx_graph.value_info)
    model_def = oh.make_model(graph_def, producer_name="jax2onnx", opset_imports=[oh.make_operatorsetid("", 21)])
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")

    return jax_outputs
