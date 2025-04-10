import onnx

# Load an ONNX model from the specified file.
model = onnx.load("docs/onnx/examples/onnx_functions/011_vit_conv_embedding.onnx")

# Print the names and shapes of the model's outputs.
print("== Model Output ==")
for output in model.graph.output:
    print(f"Name: {output.name}")
    dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")
