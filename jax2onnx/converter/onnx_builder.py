# file: jax2onnx/converter/onnx_builder.py

from onnx import (
    helper,
    TensorProto,
    NodeProto,
    ValueInfoProto,
    ModelProto,
    GraphProto,
    FunctionProto,
    OperatorSetIdProto,
)
import numpy as np
from typing import Dict, List, Any, Tuple
from jax.extend.core import Literal


class OnnxBuilder:
    def __init__(self, name_counter: int = 0, opset: int = 21) -> None:
        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = []  # TensorProto objects
        self.value_info: List[ValueInfoProto] = []
        self.name_counter: int = name_counter
        self.opset: int = opset
        self.functions: Dict[str, GraphProto] = {}

    def reset(self) -> None:
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.functions.clear()
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
        """Create the final ONNX model, including any registered nested functions."""
        opset_imports = [helper.make_opsetid("", self.opset)]

        # Add `custom` domain opset if functions are present
        if self.functions:
            opset_imports.append(helper.make_opsetid("custom", self.opset))

        model = helper.make_model(
            graph,
            opset_imports=opset_imports,
        )

        # Add FunctionProtos from nested functions
        if self.functions:
            model.functions.extend(self.create_functions())
            print(
                f"🧠 Added {len(model.functions)} nested ONNX function(s): {[f.name for f in model.functions]}"
            )

        return model

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

    def add_function(
        self, name: str, builder: "OnnxBuilder", param_input_names: List[str]
    ) -> None:
        """Registers a nested function correctly as a FunctionProto."""
        function_graph = builder.create_graph(name)

        # Inputs (dynamic inputs)
        inputs = [vi.name for vi in function_graph.input]

        # Outputs
        outputs = [vi.name for vi in function_graph.output]

        # Collect full value_info (input, output, intermediate tensors)
        all_value_info = (
            list(function_graph.input)
            + list(function_graph.output)
            + list(function_graph.value_info)
        )
        seen = set()
        unique_value_infos = []
        for vi in all_value_info:
            if vi.name not in seen:
                unique_value_infos.append(vi)
                seen.add(vi.name)

        # Explicitly add parameter initializers as function inputs with proper shapes/dtypes
        param_value_infos = [
            helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
            for init in builder.initializers
        ]

        function_proto = helper.make_function(
            domain="custom",
            fname=name,
            inputs=inputs + param_input_names,
            outputs=outputs,
            nodes=function_graph.node,
            opset_imports=[helper.make_opsetid("", self.opset)],
            attributes=[],
            value_info=unique_value_infos
            + param_value_infos,  # explicitly define params
        )

        self.functions[name] = function_proto
        print(f"🧩 Stored FunctionProto: {name} with {len(function_graph.node)} nodes")

    def add_function_call_node(
        self,
        function_name: str,
        inputs: List[str],
        outputs: List[str],
    ) -> None:
        """Adds a node calling a nested function by name."""
        node = helper.make_node(
            op_type=function_name,
            inputs=inputs,
            outputs=outputs,
            domain="custom",
        )
        self.nodes.append(node)

    def create_functions(self) -> List[FunctionProto]:
        """Converts stored function graphs into ONNX FunctionProto objects."""
        functions = []
        for func_name, graph in self.functions.items():
            opset = OperatorSetIdProto()
            opset.version = self.opset

            func_proto = FunctionProto()
            func_proto.name = func_name
            func_proto.domain = ""  # Default domain for now
            func_proto.opset_import.extend([opset])

            # Copy input/output names
            func_proto.input.extend([i.name for i in graph.input])
            func_proto.output.extend([o.name for o in graph.output])

            # Add nodes from the function graph
            func_proto.node.extend(graph.node)

            functions.append(func_proto)

        return functions

    def create_function_proto(
        self, graph: GraphProto, domain: str = ""
    ) -> FunctionProto:
        function_proto = FunctionProto()
        function_proto.name = graph.name
        function_proto.domain = domain  # leave domain empty or define clearly
        function_proto.input.extend([inp.name for inp in graph.input])
        function_proto.output.extend([out.name for out in graph.output])
        function_proto.node.extend(graph.node)
        function_proto.value_info.extend(graph.value_info)
        function_proto.opset_import.extend(
            [OperatorSetIdProto(domain="", version=self.opset)]
        )
        return function_proto

    def adjust_dynamic_batch_dimensions(self, input_shapes):
        """
        Adjusts input and output tensor shapes to handle dynamic batch dimensions ("B").
        """
        # Adjust input tensor shapes
        for tensor, input_shape in zip(self.inputs, input_shapes):
            tensor_shape = tensor.type.tensor_type.shape.dim
            for idx, dim in enumerate(input_shape):
                if dim == "B":
                    tensor_shape[idx].dim_param = "B"

        # Collect batch dimension indices
        batch_dims = {
            idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
        }

        # Adjust output tensor shapes
        for tensor in self.outputs:
            tensor_shape = tensor.type.tensor_type.shape.dim
            for idx in batch_dims:
                if idx < len(tensor_shape):
                    tensor_shape[idx].dim_param = "B"
