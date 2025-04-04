import onnx

# model = onnx.load("docs/onnx/examples/onnx_functions/011_vit_conv_embedding.onnx")
model = onnx.load("docs/onnx/examples/onnx_functions/006_one_function_outer.onnx")


def print_single_shape_info(vi):
    shape = [
        dim.dim_value if (dim.HasField("dim_value")) else "?"
        for dim in vi.type.tensor_type.shape.dim
    ]
    print(f"{vi.name}: {shape}")


def print_shape_info(model):
    print("=== INPUT INFO ===")
    for vi in model.graph.input:
        print_single_shape_info(vi)

    print("=== VALUE INFO ===")
    for vi in model.graph.value_info:
        print_single_shape_info(vi)

    print("=== OUTPUT INFO ===")
    for vi in model.graph.output:
        print_single_shape_info(vi)


print_shape_info(model)
print("######### INFERENCE #########")
model = onnx.shape_inference.infer_shapes(model)
print_shape_info(model)
