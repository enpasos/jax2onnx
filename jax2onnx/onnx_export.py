# file: jax2onnx/onnx_export.py
import onnx
import onnx.helper as oh
import importlib
import pkgutil
import os
from flax import nnx
import jax.numpy as jnp
from .transpose_utils import transpose_to_onnx, transpose_to_jax, jax_shape_to_onnx_shape, onnx_shape_to_jax_shape


def load_plugins():
    plugins = {}
    package = "jax2onnx.plugins"
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        print(f"Loading plugin: {name}")
        module = importlib.import_module(f"{package}.{name}")
        plugins[name] = module
    return plugins


def export_to_onnx(model, jax_inputs, output_path="model.onnx", build_onnx_node=None, parameters=None):
    if parameters is None:
        parameters = {}

    nnx.Module.build_onnx_node = lambda self, jax_input, nodes, parameters, counter: None
    plugins = load_plugins()
    jax_output = model(*jax_inputs)


    input_shapes = [jax_shape_to_onnx_shape(jax_input.shape) for jax_input in jax_inputs]

    # for now only one output implemented
    output_shapes = [jax_shape_to_onnx_shape(jax_output.shape) ]


    input_names = [f"input_{i}" for i in range(len(jax_inputs))]
    nodes = []
    initializers = []
    counter = [0]

    # Use the provided build_onnx_node or model-specific node
    output_names = (
        model.build_onnx_node(jax_inputs, input_names, nodes, initializers, counter)
        if hasattr(model, "build_onnx_node")
        else build_onnx_node(jax_inputs, input_names, nodes, parameters, counter)
    )


    inputs = [
        oh.make_tensor_value_info(input_names[i], onnx.TensorProto.FLOAT, input_shapes[i])
        for i in range(len(jax_inputs))
    ]


    outputs = [
        oh.make_tensor_value_info(output_names[i], onnx.TensorProto.FLOAT, output_shapes[i]) for i in range(len(output_names))
    ]

    # Create the ONNX graph
    graph_def = oh.make_graph(nodes, "NNXExportGraph", inputs, outputs, initializers)
    model_def = oh.make_model(
        graph_def,
        producer_name="jax2onnx",
        opset_imports=[oh.make_operatorsetid("", 21)],  # Ensure compatibility with ONNX Runtime
    )
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")

