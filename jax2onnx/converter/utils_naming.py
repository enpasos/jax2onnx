# file: jax2onnx/utils/naming.py


def get_qualified_name(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"
