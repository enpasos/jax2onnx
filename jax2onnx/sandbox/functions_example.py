# file: jax2onnx/sandbox/functions_example.py

import os

import jax.numpy as jnp
import onnx
from flax import nnx
from onnx import GraphProto, TensorProto, helper

from jax2onnx import to_onnx


class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


# === Instantiate and export ===
my_callable = MLPBlock(num_hiddens=256, mlp_dim=512, dropout_rate=0.1, rngs=nnx.Rngs(0))

onnx_model = to_onnx(
    my_callable,
    [("B", 10, 256)],
    #   enable_dynamic_shapes=True,
    #   include_intermediate_shapes=True  # ✅ ensures all intermediate value_info are kept
)

output_path = "./docs/onnx/sandbox/mlp_block.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(onnx_model, output_path)


mlp_graph: GraphProto = onnx_model.graph
param_names = [init.name for init in mlp_graph.initializer]

# === Collect full value_info (input, output, intermediate tensors) ===
# Deduplicate by name
all_value_info = (
    list(mlp_graph.input) + list(mlp_graph.output) + list(mlp_graph.value_info)
)
seen = set()
function_value_infos = []
for vi in all_value_info:
    if vi.name not in seen:
        function_value_infos.append(vi)
        seen.add(vi.name)

# === Define the function ===
mlp_function = helper.make_function(
    domain="custom",
    fname="MLPBlock_001",
    inputs=[i.name for i in mlp_graph.input] + param_names,
    outputs=[o.name for o in mlp_graph.output],
    nodes=mlp_graph.node,
    opset_imports=list(onnx_model.opset_import),
    attributes=[],
    value_info=function_value_infos,  # ✅ full shape-aware tensors
)

# === Create top-level graph ===
input_tensor = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 30])
output_tensor = helper.make_tensor_value_info("Output", TensorProto.FLOAT, [1, 256])

# Parameter inputs (need shape info too)
param_value_infos = [
    helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
    for init in mlp_graph.initializer
]

mlp_node = helper.make_node(
    "MLPBlock_001",
    inputs=["B"] + param_names,
    outputs=["Output"],
    domain="custom",
)

top_graph = helper.make_graph(
    nodes=[mlp_node],
    name="MLPBlockGraph",
    inputs=[input_tensor] + param_value_infos,
    outputs=[output_tensor],
    initializer=mlp_graph.initializer,
)

# === Final model ===
opset_imports = list(onnx_model.opset_import) + [helper.make_opsetid("custom", 1)]

top_model = helper.make_model(
    top_graph,
    opset_imports=opset_imports,
    functions=[mlp_function],
    ir_version=9,
    producer_name="jax2onnx + your-brain",
)

# === Save it ===
output_path = "./docs/onnx/sandbox/onnx_with_mlp_function_option1b.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(top_model, output_path)
print(f"✅ ONNX model (Reshapes + Shapes) saved to:\n{output_path}")
