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
import onnx

from jax2onnx.converter.name_generator import GlobalNameCounter


class OnnxBuilder:
    def __init__(
        self,
        name_counter: GlobalNameCounter = GlobalNameCounter(),
        opset: int = 21,
        model_name: str = "",
    ) -> None:
        self.name_counter: GlobalNameCounter = name_counter
        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = []  # TensorProto objects
        self.value_info: List[ValueInfoProto] = []
        self.opset: int = opset
        self.functions: Dict[str, GraphProto] = {}
        self.model_name: str = model_name  # Added model_name attribute

    def reset(self) -> None:
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.functions.clear()
        self.name_counter = GlobalNameCounter()  # Reset the name counter

    def get_unique_name(self, prefix: str = "node") -> str:
        return self.name_counter.get(prefix)

    def get_constant_name(self, val):
        """
        Creates a globally unique ONNX constant tensor and returns its name.
        """
        # Always generate a unique name (never reuse)
        name = self.get_unique_name("const")

        if isinstance(val, Literal):
            val = val.val

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

    def _add_tensor(
        self,
        collection: List[ValueInfoProto],
        name: str,
        shape: Tuple[int, ...],
        dtype: Any,
    ):
        """
        Generalized method to add a tensor to a specified collection (inputs, outputs, or value_info).
        """
        tensor_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        collection.append(tensor_def)

    def add_input(
        self, name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        self._add_tensor(self.inputs, name, shape, dtype)

    def add_output(
        self, name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        self._add_tensor(self.outputs, name, shape, dtype)

    def add_value_info(
        self, name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        self._add_tensor(self.value_info, name, shape, dtype)

    def create_node(
        self, op_type: str, inputs: List[str], outputs: List[str], **kwargs: Any
    ) -> NodeProto:
        return helper.make_node(op_type, inputs, outputs, **kwargs)

    def add_node(self, node: NodeProto) -> None:
        self.nodes.append(node)

    def _build_graph(self, name: str) -> GraphProto:
        """
        Helper method to build an ONNX graph with the current nodes, inputs, outputs, initializers, and value_info.
        """
        return helper.make_graph(
            nodes=self.nodes,
            name=name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info,
        )

    def create_graph(self, name: str) -> GraphProto:
        """
        Creates an ONNX graph using the current state of the builder.
        """
        return self._build_graph(name)

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

    def create_onnx_model(self, model_name: str) -> onnx.ModelProto:
        """
        Creates the ONNX model by assembling the graph, initializers, and functions.
        """
        graph = self._build_graph(model_name)

        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid("custom", 1),  # Explicitly import custom domain
            ],
            functions=list(self.functions.values()),
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
        """
        Registers a nested function correctly as a FunctionProto.
        """
        function_graph = builder.create_graph(name)

        # Collect inputs and outputs
        inputs = [vi.name for vi in function_graph.input]
        outputs = [vi.name for vi in function_graph.output]

        # Collect unique value_info (input, output, intermediate tensors)
        all_value_info = (
            list(function_graph.input)
            + list(function_graph.output)
            + list(function_graph.value_info)
        )
        unique_value_infos = list({vi.name: vi for vi in all_value_info}.values())

        # Add parameter initializers as function inputs
        param_value_infos = [
            helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
            for init in builder.initializers
        ]

        # Combine unique value_infos and parameter initializers
        combined_value_infos = unique_value_infos + param_value_infos

        # Create the FunctionProto
        function_proto = helper.make_function(
            domain="custom",
            fname=name,
            inputs=inputs + param_input_names,
            outputs=outputs,
            nodes=function_graph.node,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid(
                    "custom", self.opset
                ),  # Ensure custom domain is included
            ],
            attributes=[],
            value_info=combined_value_infos,
        )

        # self.functions[name] = function_proto

        # After storing the function
        self.functions[name] = function_proto
        print(f"🧩 Stored FunctionProto: {name} with {len(function_graph.node)} nodes")
        #
        #         # 🚨 Remove function initializers from global initializer list to maintain SSA form
        #         function_initializer_names = {init.name for init in builder.initializers}
        #         self.initializers = [
        #             init for init in self.initializers if init.name not in function_initializer_names
        # ]

        print(f"🧩 Stored FunctionProto: {name} with {len(function_graph.node)} nodes")

    def add_function_call_node(
        self,
        function_name: str,
        inputs: List[str],
        outputs: List[str],
    ) -> None:
        """Adds a node calling a nested function by name."""
        # Ensure all inputs and outputs are properly connected
        assert inputs, f"Function {function_name} must have inputs."
        assert outputs, f"Function {function_name} must have outputs."

        node = helper.make_node(
            op_type=function_name,
            inputs=inputs,
            outputs=outputs,
            domain="custom",
        )
        self.nodes.append(node)
        print(
            f"🧩 Added function call node: {function_name} with inputs {inputs} and outputs {outputs}"
        )

    def _build_function_proto(
        self, graph: GraphProto, domain: str = ""
    ) -> FunctionProto:
        """
        Helper method to build a FunctionProto from a GraphProto.
        """
        function_proto = FunctionProto()
        function_proto.name = graph.name
        function_proto.domain = domain
        function_proto.input.extend([inp.name for inp in graph.input])
        function_proto.output.extend([out.name for out in graph.output])
        function_proto.node.extend(graph.node)
        function_proto.value_info.extend(graph.value_info)
        function_proto.opset_import.extend(
            [OperatorSetIdProto(domain="", version=self.opset)]
        )
        return function_proto

    def create_functions(self) -> List[FunctionProto]:
        """
        Converts stored function graphs into ONNX FunctionProto objects.
        """
        return [
            self._build_function_proto(graph, "") for graph in self.functions.values()
        ]

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

    def _adjust_tensor_shape(self, tensor, shape, batch_dims):
        tensor_shape = tensor.type.tensor_type.shape.dim
        for idx, dim in enumerate(shape):
            if dim == "B":
                tensor_shape[idx].dim_param = "B"
        for idx in batch_dims:
            if idx < len(tensor_shape):
                tensor_shape[idx].dim_param = "B"

    def adjust_dynamic_batch_dimensions(self, input_shapes):
        """
        Adjusts input and output tensor shapes to handle dynamic batch dimensions ("B").
        """
        batch_dims = {
            idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
        }
        for tensor, input_shape in zip(self.inputs, input_shapes):
            self._adjust_tensor_shape(tensor, input_shape, batch_dims)
        for tensor in self.outputs:
            self._adjust_tensor_shape(tensor, [], batch_dims)

    def filter_unused_initializers(self):
        """
        Ensures all required initializers are included in the graph.
        """
        used_inputs = {i for node in self.nodes for i in node.input}
        # Include all initializers that are used or required by the graph
        self.initializers = [
            init for init in self.initializers if init.name in used_inputs or init.name
        ]
