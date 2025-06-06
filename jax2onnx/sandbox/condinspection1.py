import onnx

try:
    model = onnx.load("docs/onnx/primitives/lax/cond_scalar.onnx")
    onnx.checker.check_model(model)
    print("ONNX model is valid.")
except Exception as e:
    print(f"ONNX model is invalid: {e}")
