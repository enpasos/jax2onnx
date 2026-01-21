# check_native_clone.py

import onnx_ir as ir
from onnx import TensorProto


def test_native_clone_metadata():
    graph = ir.Graph(name="test_graph", inputs=[], outputs=[], nodes=[])
    graph.meta["test_key"] = "test_val"

    input_val = ir.Value(
        name="input_0", type=ir.TensorType(TensorProto.FLOAT), shape=ir.Shape([1, 2])
    )
    input_val.meta["val_meta_key"] = "val_meta_val"
    graph.inputs.append(input_val)

    # Try native clone
    if not hasattr(graph, "clone"):
        print("native_clone_metadata: SKIP (graph.clone not found)")
        return

    new_graph = graph.clone(allow_outer_scope_values=True)

    # Check graph meta
    if new_graph.meta.get("test_key") == "test_val":
        print("Graph metadata preserved: YES")
    else:
        print("Graph metadata preserved: NO")

    # Check value meta
    new_input = new_graph.inputs[0]
    if new_input.meta.get("val_meta_key") == "val_meta_val":
        print("Value metadata preserved: YES")
    else:
        print("Value metadata preserved: NO")


if __name__ == "__main__":
    try:
        test_native_clone_metadata()
    except Exception as e:
        print(f"FAIL: {e}")
