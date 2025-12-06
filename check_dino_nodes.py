# check_dino_nodes.py

import onnx

model_path = "docs/onnx/examples/eqx_dino/eqx_dinov3_vit_B14_dynamic.onnx"
try:
    model = onnx.load(model_path)
    conv_transpose_nodes = [n for n in model.graph.node if n.op_type == "ConvTranspose"]
    print(f"Found {len(conv_transpose_nodes)} ConvTranspose nodes.")
    for n in conv_transpose_nodes:
        print(f"Node: {n.name}, Inputs: {n.input}, Outputs: {n.output}")
except Exception as e:
    print(f"Error loading model: {e}")
