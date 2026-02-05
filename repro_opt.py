# repro_opt.py

import jax.numpy as jnp
from jax2onnx import to_onnx


def test_optimization():
    # User's case:
    # Input 1x32xHxW
    # Transpose 0, 2, 3, 1 -> 1xHxWx32
    # ReduceMean axes=(1, 2) -> 1x1x1x32 (keepdims=True)
    # Transpose 0, 3, 1, 2 -> 1x32x1x1

    def f(x):
        # x is 1x32x10x20
        # channel last: 1x10x20x32
        y = jnp.transpose(x, (0, 2, 3, 1))
        # Reduce H, W (axes 1, 2)
        z = jnp.mean(y, axis=(1, 2), keepdims=True)
        # Back to NCHW
        w = jnp.transpose(z, (0, 3, 1, 2))
        return w

    input_shape = (1, 32, 10, 20)
    onnx_model = to_onnx(f, inputs=[input_shape])

    # Check for Transpose nodes
    nodes = onnx_model.graph.node
    transpose_nodes = [n for n in nodes if n.op_type == "Transpose"]
    reduce_nodes = [n for n in nodes if n.op_type == "ReduceMean"]

    print(f"Found {len(transpose_nodes)} Transpose nodes.")
    print(f"Found {len(reduce_nodes)} ReduceMean nodes.")

    if len(transpose_nodes) == 0 and len(reduce_nodes) == 1:
        print("SUCCESS: Optimization applied. No Transpose nodes found.")
    else:
        print("FAILURE: Optimization NOT applied.")
        for n in transpose_nodes:
            print(f"  Transpose: {n.name}")
            print(f"    Inputs: {n.input}")
            print(f"    Outputs: {n.output}")
            for attr in n.attribute:
                print(f"    Attr: {attr.name} = {attr.ints}")
        for n in reduce_nodes:
            print(f"  ReduceMean: {n.name}")
            print(f"    Inputs: {n.input}")
            print(f"    Outputs: {n.output}")
            for attr in n.attribute:
                print(f"    Attr: {attr.name}")


if __name__ == "__main__":
    test_optimization()
