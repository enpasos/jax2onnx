from onnx import helper, TensorProto
import numpy as np

ONNX_OPSET_VERSION = 21


class OnnxBuilder:
    def __init__(self, name_counter=0):
        self.name_counter = name_counter
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        # Maintain maps if needed for constants etc.
        self.var_to_name = {}
        self.name_to_const = {}

    def get_unique_name(self, prefix="node"):
        name = f"{prefix}_{self.name_counter}"
        self.name_counter += 1
        return name

    def numpy_dtype_to_onnx(self, dtype):
        """Convert numpy dtype to ONNX data type."""
        if dtype == np.float32:
            return TensorProto.FLOAT
        elif dtype == np.float64:
            return TensorProto.DOUBLE
        elif dtype == np.int32:
            return TensorProto.INT32
        elif dtype == np.int64:
            return TensorProto.INT64
        elif dtype == np.bool_:
            return TensorProto.BOOL
        else:
            return TensorProto.FLOAT

    def get_constant_name(self, val):
        """Convert a value into an ONNX constant and register it."""
        name = self.get_unique_name("const")
        if hasattr(val, "val"):  # handle JAX Literal objects
            actual_val = val.val
        else:
            actual_val = val
        np_val = np.array(actual_val)
        if np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)
        tensor = helper.make_tensor(
            name=name,
            data_type=self.numpy_dtype_to_onnx(np_val.dtype),
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        # Optionally, track constants for re-use:
        self.name_to_const[name] = np_val
        return name

    def add_input(self, name, shape, dtype=np.float32):
        input_def = helper.make_tensor_value_info(
            name, self.numpy_dtype_to_onnx(dtype), shape
        )
        self.inputs.append(input_def)
        return name

    def add_output(self, name, shape, dtype=np.float32):
        output_def = helper.make_tensor_value_info(
            name, self.numpy_dtype_to_onnx(dtype), shape
        )
        self.outputs.append(output_def)
        return name

    def add_value_info(self, name, shape, dtype=np.float32):
        value_info = helper.make_tensor_value_info(
            name, self.numpy_dtype_to_onnx(dtype), shape
        )
        self.value_info.append(value_info)
        return name

    def add_node(self, op_type, inputs, outputs, name=None, **attrs):
        if name is None:
            name = self.get_unique_name(op_type.lower())
        node = helper.make_node(
            op_type, inputs=inputs, outputs=outputs, name=name, **attrs
        )
        self.nodes.append(node)
        return node

    def create_graph(self, graph_name):
        """Build an ONNX graph using the accumulated nodes and definitions."""
        return helper.make_graph(
            nodes=self.nodes,
            name=graph_name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info,
        )

    def create_model(self, graph, producer_name="jaxpr_to_onnx"):
        """Wrap the graph into an ONNX model."""
        return helper.make_model(
            graph,
            producer_name=producer_name,
            opset_imports=[helper.make_opsetid("", ONNX_OPSET_VERSION)],
        )
