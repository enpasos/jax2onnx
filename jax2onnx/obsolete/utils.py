from obsolete.convert import OnnxGraph, Z


def retry_with_dynamic_batch_dim(func, self, z: Z, **params):
    """Utility function to retry ONNX conversion with dynamic batch dim handling."""
    onnx_graph: OnnxGraph = z.onnx_graph
    input_shape = z.shapes[0]

    original_input_shape = input_shape
    original_internal_shape_info = onnx_graph.internal_shape_info
    original_dynamic_batch_dim = onnx_graph.dynamic_batch_dim

    try:
        return func(self, z, **params)
    except Exception as e:
        if onnx_graph.dynamic_batch_dim:
            # Temporarily switch off dynamic_batch_dim if an error occurs
            onnx_graph.dynamic_batch_dim = False
            onnx_graph.internal_shape_info = False

            # Replace the batch-dim value with a concrete one (1)
            input_shape = (1,) + tuple(input_shape[1:])
            z.shapes[0] = input_shape

            # Retry the conversion
            z2 = func(self, z, **params)

            # Restore the original dynamic_batch_dim setting
            onnx_graph.dynamic_batch_dim = original_dynamic_batch_dim
            onnx_graph.internal_shape_info = original_internal_shape_info

            # Restore the original input shape
            z.shapes[0] = original_input_shape

            # Ensure z2.shapes is a list of tuples
            z2.shapes = [tuple(shape) for shape in z2.shapes]
            z2.shapes[0] = (original_input_shape[0],) + z2.shapes[0][1:]
            return z2
        else:
            raise e
