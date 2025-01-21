# file: jax2onnx/onnx_export.py
import onnx
import onnx.helper as oh
import importlib
import pkgutil
import os
from flax import nnx
import jax.numpy as jnp

def load_plugins():
    plugins = {}
    package = 'jax2onnx.plugins'
    plugins_path = os.path.join(os.path.dirname(__file__), 'plugins')
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        print(f"Loading plugin: {name}")
        module = importlib.import_module(f'{package}.{name}')
        plugins[name] = module
    return plugins


def export_to_onnx(model, jax_inputs, output_path="model.onnx", build_onnx_node=False):
    nnx.Module.build_onnx_node = lambda self, jax_input, nodes, parameters, counter: None
    plugins = load_plugins()



    # wenn jax_input array, dann den ersten Eintrag
    # jax_input = jax_inputs[0] if isinstance(jax_inputs, list) else jax_inputs

    jax_outputs = model(*jax_inputs)


    # iterate over all jax_inputs and convert them to a list of inputTensors called inputs
    inputs = []
    input_names = []
    for i, jax_input in enumerate(jax_inputs):
        input_shape = jax_shape_to_onnx_shape(jax_input.shape)
        intput_name = f"input_{i}"
        input_names.append(intput_name)
        inputs.append(oh.make_tensor_value_info(intput_name, onnx.TensorProto.FLOAT, input_shape))

    nodes = []
    initializers = []

    counter = [0]
    # in case model.build_onnx_node is not implemented, build_onnx_node must be present and is used to build the ONNX graph
    # until now we have n : 1 mapping of jax_inputs to jax_outputs

    output_names = model.build_onnx_node(jax_inputs, input_names, nodes, initializers, counter) if hasattr(model, "build_onnx_node") \
        else build_onnx_node(jax_inputs, input_names, nodes, initializers, counter)

    # iterate over all jax_outputs and convert them to a list of outputTensors called outputs
    outputs = []
    if len(output_names) == 1:
        output_shape = jax_shape_to_onnx_shape(jax_outputs.shape)
        outputs.append(oh.make_tensor_value_info(output_names[0], onnx.TensorProto.FLOAT, output_shape))
    else:
         raise ValueError("Multiple outputs not yet implemented")

    graph_def = oh.make_graph(
        nodes,
        "NNXExportGraph",
        inputs,
        outputs,
        initializers
    )

    model_def = oh.make_model(graph_def, producer_name="nnx2onnx", opset_imports=[oh.make_operatorsetid("", 21)])
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to {output_path}")


# define a function converting von jax shape to onnx shape
# (B, H, W, C) -> (B, C, H, W)
# or
# (B, C) -> (B, C)
# consider different cases
def jax_shape_to_onnx_shape(jax_shape : tuple):
    if len(jax_shape) == 4:
        return tuple([jax_shape[0], jax_shape[3], jax_shape[1], jax_shape[2]])
    else:
        return tuple(jax_shape)

def jax_to_onnx_axes(jax_shape: tuple):
    if len(jax_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        return (0, 3, 1, 2)
    return None  # Return None for unsupported cases


def onnx_to_jax_axes(onnx_shape: tuple):
    if len(onnx_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        return (0, 2, 3, 1)
    return None  # Return None for unsupported cases

# define a function converting von onnx shape to jax shape
# (B, C, H, W) -> (B, H, W, C)
# (B, C) -> (B, C)
# consider different cases
def onnx_shape_to_jax_shape(onnx_shape : tuple):
    if len(onnx_shape) == 4:
        return tuple([onnx_shape[0], onnx_shape[2], onnx_shape[3], onnx_shape[1]])
    else:
        return tuple(onnx_shape)

def transpose_to_onnx(array):
    axes = jax_to_onnx_axes(array.shape)
    if axes is None:
        return array  # Return the original array for unsupported cases
    return jnp.transpose(array, axes)


def transpose_to_jax(array):
    axes = onnx_to_jax_axes(array.shape)
    if axes is None:
        return array  # Return the original array for unsupported cases
    return jnp.transpose(array, axes)
