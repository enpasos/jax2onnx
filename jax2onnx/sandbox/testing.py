import onnx

model = onnx.load("docs/onnx/examples/onnx_functions/011_vit_conv_embedding.onnx")

print("=== VALUE INFO BEFORE SHAPE INFERENCE ===")
for vi in list(model.graph.value_info) + list(model.graph.output):
    shape = [
        dim.dim_value if (dim.HasField("dim_value")) else "?"
        for dim in vi.type.tensor_type.shape.dim
    ]
    print(f"{vi.name}: {shape}")

# Now run inference
inferred = onnx.shape_inference.infer_shapes(model)
