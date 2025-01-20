# jax2onnx/onnx_export.py
import onnx
import onnx.helper as oh
import importlib
import pkgutil
import os
from flax import nnx

def load_plugins():
    plugins = {}
    package = 'jax2onnx.plugins'
    plugins_path = os.path.join(os.path.dirname(__file__), 'plugins')
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        print(f"Loading plugin: {name}")
        module = importlib.import_module(f'{package}.{name}')
        plugins[name] = module
    return plugins


def export_to_onnx(model, example_input, output_path="model.onnx", build_onnx_node=False):
    nnx.Module.build_onnx_node = lambda self, example_input, nodes, parameters, counter: None
    plugins = load_plugins()

    example_output = model(example_input)
    input_shape = example_input.shape
    output_shape = example_output.shape

    input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)
    nodes = []
    initializers = []

    counter = [0]
    # in case model.build_onnx_node is not implemented, build_onnx_node must be present and is used to build the ONNX graph
    output_name = model.build_onnx_node(example_input, "input", nodes, initializers, counter) if hasattr(model, "build_onnx_node") \
        else build_onnx_node(example_input, "input", nodes, initializers, counter)
    #output_name = model.build_onnx_node(example_input, "input", nodes, initializers, counter)
    output_tensor = oh.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)

    graph_def = oh.make_graph(
        nodes,
        "NNXExportGraph",
        [input_tensor],
        [output_tensor],
        initializers
    )

    model_def = oh.make_model(graph_def, producer_name="nnx2onnx", opset_imports=[oh.make_operatorsetid("", 21)])
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")


