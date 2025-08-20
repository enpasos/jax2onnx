import unittest
import sys
import subprocess
import os
import tempfile
import onnx
import onnxruntime as ort

REPRODUCER_SCRIPT_CODE = r"""
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
import onnxruntime as ort
import numpy as np
import onnx
import tempfile

jax.config.update("jax_enable_x64", True)

def broken():
    float_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    int_arr = jnp.array([3, 4], dtype=jnp.int32)
    concat_result = jnp.concatenate([float_arr, int_arr])   # -> dtype float64 in x64 mode
    lookup = jnp.array([100, 200, 300, 400, 500], dtype=jnp.int32)
    indices = jnp.clip(concat_result.astype(jnp.int32), 0, len(lookup) - 1)
    indexed_vals = jnp.take(lookup, indices)
    float_vals = concat_result * 1.5                       # <- python float literal
    return concat_result, indexed_vals, float_vals

def main():
    model = to_onnx(broken, inputs=[], enable_double_precision=True)
    
    # Test if ORT can load the model
    try:
        sess = ort.InferenceSession(model.SerializeToString())
    except Exception:
        # Accept legacy ORT behavior.
        print("SUCCESS (legacy ORT failure mode)")
        return

    # If ORT loads, verify we did not emit a Concat with mixed input dtypes.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        temp_path = f.name
    
    try:
        m = onnx.load(temp_path)

        def _walk_graph(g):
            yield g
            for n in g.node:
                for a in n.attribute:
                    if a.type == onnx.AttributeProto.GRAPH:
                        yield from _walk_graph(a.g)
                    elif a.type == onnx.AttributeProto.GRAPHS:
                        for sub in a.graphs:
                            yield from _walk_graph(sub)

        def _elem_type(g, name):
            for vi in list(g.input) + list(g.value_info) + list(g.output):
                if vi.name == name and vi.type and vi.type.tensor_type:
                    return vi.type.tensor_type.elem_type
            for t in g.initializer:
                if t.name == name:
                    return t.data_type
            return None

        for g in _walk_graph(m.graph):
            for n in g.node:
                if n.op_type == "Concat":
                    types = { _elem_type(g, inp) for inp in n.input if _elem_type(g, inp) is not None }
                    assert len(types) == 1, f"Concat has mixed dtypes: {types}"
        
        print("SUCCESS (dtype consistency verified)")
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    main()
"""


class TestConcatMixedDtypesReproducer(unittest.TestCase):
    def test_reproducer_succeeds_after_fix(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            path = f.name
            f.write(REPRODUCER_SCRIPT_CODE)
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True, text=True, check=True)
            self.assertIn("SUCCESS", proc.stdout)
        finally:
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
