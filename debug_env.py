# debug_env.py

import sys
import inspect
import onnx_ir as ir

print(f"Executable: {sys.executable}")
print(f"Path: {sys.path}")
print(f"onnx_ir file: {ir.__file__}")

try:
    sig = inspect.signature(ir.Graph.clone)
    print(f"Graph.clone signature: {sig}")
except Exception as e:
    print(f"Could not inspect signature: {e}")

try:
    g = ir.Graph([], [], nodes=[])
    print("Graph created successfully")
    try:
        g.clone(allow_outer_scope_values=True)
        print("g.clone(allow_outer_scope_values=True) WORKED")
    except TypeError as e:
        print(f"g.clone(allow_outer_scope_values=True) FAILED: {e}")
except Exception as e:
    print(f"Graph creation failed: {e}")
