from onnx import helper, TensorProto
import numpy as np


class OnnxBuilder:
    def __init__(self, name_counter=0):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.name_counter = name_counter

    def reset(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.name_counter = 0

    def get_unique_name(self, prefix="node"):
        name = f"{prefix}_{self.name_counter}"
        self.name_counter += 1
        return name

    def get_constant_name(self, val):
        name = self.get_unique_name("const")
        np_val = np.array(val)
        if np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)
        tensor = helper.make_tensor(
            name=name,
            data_type=self._numpy_dtype_to_onnx(np_val.dtype),
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        return name

    def add_input(self, name, shape, dtype=np.float32):
        input_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.inputs.append(input_def)

    def add_output(self, name, shape, dtype=np.float32):
        output_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.outputs.append(output_def)

    def add_value_info(self, name, shape, dtype=np.float32):
        value_info = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.value_info.append(value_info)

    def create_node(self, op_type, inputs, outputs, **kwargs):
        return helper.make_node(op_type, inputs, outputs, **kwargs)

    def add_node(self, node):
        self.nodes.append(node)

    def create_graph(self, name):
        return helper.make_graph(
            nodes=self.nodes,
            name=name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info,
        )

    def create_model(self, graph):
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

    def _numpy_dtype_to_onnx(self, dtype):
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
