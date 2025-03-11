# file: jax2onnx/converter/onnx_builder.py

from onnx import helper, TensorProto, NodeProto, ValueInfoProto, ModelProto, GraphProto
import numpy as np
from typing import List, Any, Tuple
from jax.extend.core import Literal


class OnnxBuilder:
    def __init__(self, name_counter: int = 0, opset_version: int = 21) -> None:
        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = []  # TensorProto objects
        self.value_info: List[ValueInfoProto] = []
        self.name_counter: int = name_counter
        self.opset_version: int = opset_version

    def reset(self) -> None:
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.name_counter = 0

    def get_unique_name(self, prefix: str = "node") -> str:
        name = f"{prefix}_{self.name_counter}"
        self.name_counter += 1
        return name

    def get_constant_name(self, val):
        name = self.get_unique_name("const")
        # Unwrap a JAX Literal to get its Python value.

        if isinstance(val, Literal):
            val = val.val
        # Continue with conversion, for example converting to numpy if needed:
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

    def add_input(
        self, name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        input_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.inputs.append(input_def)

    def add_output(
        self, name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        output_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.outputs.append(output_def)

    def add_value_info(
        self, name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        value_info = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.value_info.append(value_info)

    def create_node(
        self, op_type: str, inputs: List[str], outputs: List[str], **kwargs: Any
    ) -> NodeProto:
        return helper.make_node(op_type, inputs, outputs, **kwargs)

    def add_node(self, node: NodeProto) -> None:
        self.nodes.append(node)

    def create_graph(self, name: str) -> GraphProto:
        return helper.make_graph(
            nodes=self.nodes,
            name=name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info,
        )

    def create_model(self, graph: GraphProto) -> ModelProto:
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", self.opset_version)]
        )

    def _numpy_dtype_to_onnx(self, dtype: Any) -> int:
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
