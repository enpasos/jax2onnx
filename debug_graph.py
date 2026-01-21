# debug_graph.py

import onnx_ir as ir
from onnx import TensorProto


def debug():
    print("--- Debugging Graph Construction ---")
    try:
        # Create input
        input_val = ir.Value(
            name="input_0",
            type=ir.TensorType(TensorProto.FLOAT),
            shape=ir.Shape([1, 2]),
        )
        # Create output
        output_val = ir.Value(
            name="output_0",
            type=ir.TensorType(TensorProto.FLOAT),
            shape=ir.Shape([1, 2]),
        )

        # Create node
        # Note: Empty domain string as confirmed earlier
        node = ir.Node("", "Identity", inputs=[input_val], outputs=[output_val])

        print("Node created.")
        print(f"Input produced by: {input_val.producer()}")
        print(f"Output produced by: {output_val.producer()}")

        # Create Graph
        print("Creating Graph...")
        ir.Graph(
            inputs=[input_val], outputs=[output_val], nodes=[node], name="test_graph"
        )
        print("Graph created successfully.")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug()
