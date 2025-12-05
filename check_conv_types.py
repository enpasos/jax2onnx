# check_conv_types.py

import onnx

model_path = "docs/onnx/examples/eqx_dino/eqx_dinov3_vit_S14.onnx"
try:
    model = onnx.load(model_path)
    node_types = set()
    for n in model.graph.node:
        node_types.add(n.op_type)
    print(f"Node types found: {node_types}")
    print(f"Functions found: {[f.name for f in model.functions]}")
    
    for n in model.graph.node:
        if n.op_type == "Conv":
            print(f"Conv Node: {n.name}")
            for i, inp in enumerate(n.input):
                # Find the value info for this input
                val_info = next((v for v in model.graph.value_info if v.name == inp), None)
                if not val_info:
                    val_info = next((v for v in model.graph.input if v.name == inp), None)
                if not val_info:
                    val_info = next((v for v in model.graph.initializer if v.name == inp), None)
                    if val_info:
                        print(f"  Input {i}: {inp} (Initializer, dtype={val_info.data_type})")
                        continue
                
                if val_info:
                    print(f"  Input {i}: {inp} (dtype={val_info.type.tensor_type.elem_type})")
                else:
                    print(f"  Input {i}: {inp} (Unknown type)")
            
            for i, out in enumerate(n.output):
                val_info = next((v for v in model.graph.value_info if v.name == out), None)
                if val_info:
                    print(f"  Output {i}: {out} (dtype={val_info.type.tensor_type.elem_type})")
                else:
                    print(f"  Output {i}: {out} (Unknown type)")

except Exception as e:
    print(f"Error loading model: {e}")
