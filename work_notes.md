Thanks for the pointer! I've refactored ir_clone.py to use the native onnx_ir._cloner.Cloner as suggested.

The cloner is a private class. I would recommend using the clone() method instead. There is a PR pending: onnx/ir-py#313

Could you check if it fits the need if this project?

    I also found that I needed to extend clone_node and clone_value to explicitly copy type, shape, and metadata_props. The native cloner currently produces fresh values without these properties, which was causing issues with lost JAX tracing metadata (like loop stack extents) in our tests.

That’s strange, will fix
