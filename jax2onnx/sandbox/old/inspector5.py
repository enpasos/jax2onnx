import onnx


def inspect_onnx_model(model_path):
    print(f"Loading ONNX model from {model_path}")
    model = onnx.load(model_path)

    print("\nFunctions in the model:")
    for f in model.functions:
        print(f"- Function: {f.name}")
        print(f"  Inputs: {list(f.input)}")
        print(f"  Outputs: {list(f.output)}")

        print("\n  Nodes in function:")
        for i, node in enumerate(f.node):
            print(f"    Node {i}: {node.op_type} - {node.name}")
            print(f"      Inputs: {list(node.input)}")
            print(f"      Outputs: {list(node.output)}")
        print()


if __name__ == "__main__":
    inspect_onnx_model("docs/onnx/duplicate_param_test.onnx")
