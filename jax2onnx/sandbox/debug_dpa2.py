import onnx

model = onnx.load("docs/onnx/primitives/nn/dpa_debug_one_false.onnx")
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"{inp.name:8s}  → shape {dims}")
