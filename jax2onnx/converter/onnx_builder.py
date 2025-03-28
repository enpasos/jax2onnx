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


CUSTOM_DOMAIN = "custom"  # "jax2onnx.functions"
CUSTOM_DOMAIN_VERSION = 1  # Start versioning at 1


class OnnxBuilder:
    def __init__(
        self,
        name_counter: GlobalNameCounter,
        opset: int = 21,
        model_name: str = "",
        constant_cache: (
            Dict[Tuple[bytes, np.dtype, Tuple[int, ...]], str] | None
        ) = None,
        initializers: List[Any] | None = None,
    ) -> None:
        self.name_counter: GlobalNameCounter = name_counter
        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = initializers if initializers is not None else []
        self.value_info: List[ValueInfoProto] = []
        self.opset: int = opset
        self.functions: Dict[str, FunctionProto] = {}
        self.model_name: str = model_name
        self.constant_cache: Dict[Tuple[bytes, np.dtype, Tuple[int, ...]], str] = (
            constant_cache if constant_cache is not None else {}
        )
        self.function_name_to_domain: Dict[str, str] = {}

    def get_constant_name(self, val):
        if isinstance(val, Literal):
            val = val.val

        np_val = np.array(val)
        if np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)

        try:
            cache_key = (np_val.tobytes(), np_val.dtype, np_val.shape)
        except AttributeError:
            cache_key = (str(val), type(val), ())

        if cache_key in self.constant_cache:
            return self.constant_cache[cache_key]
        else:
            name = self.get_unique_name("const")
            tensor = helper.make_tensor(
                name=name,
                data_type=self._numpy_dtype_to_onnx(np_val.dtype),
                dims=np_val.shape,
                vals=np_val.flatten().tolist(),
            )
            self.initializers.append(tensor)
            self.constant_cache[cache_key] = name
            return name

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

    def _add_tensor(
        self,
        collection: List[ValueInfoProto],
        name: str,
        shape: Tuple[int, ...],
        dtype: Any,
    ):
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
        return helper.make_graph(
            nodes=self.nodes,
            name=name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info,
        )

    def create_graph(self, name: str) -> GraphProto:
        return self._build_graph(name)

    def create_model(self, graph: GraphProto) -> ModelProto:
        opset_imports = [helper.make_opsetid("", self.opset)]

        if self.functions:
            opset_imports.append(helper.make_opsetid("custom", self.opset))

        model = helper.make_model(
            graph,
            opset_imports=opset_imports,
        )

        if self.functions:
            model.functions.extend(self.create_functions())
            print(
                f"🧠 Added {len(model.functions)} nested ONNX function(s): {[f.name for f in model.functions]}"
            )

        return model

    def create_onnx_model(self, model_name: str) -> onnx.ModelProto:
        graph = self._build_graph(model_name)

        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid(
                    CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION
                ),  # Custom domain for functions
            ],
            functions=list(self.functions.values()),  # Adds the FunctionProtos
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
        function_graph = builder.create_graph(name)
        inputs = [vi.name for vi in function_graph.input]
        outputs = [vi.name for vi in function_graph.output]

        all_value_info = (
            list(function_graph.input)
            + list(function_graph.output)
            + list(function_graph.value_info)
        )
        unique_value_infos = list({vi.name: vi for vi in all_value_info}.values())

        param_value_infos = [
            helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
            for init in builder.initializers
        ]

        combined_value_infos = unique_value_infos + param_value_infos

        function_proto = helper.make_function(
            domain=CUSTOM_DOMAIN,
            fname=name,
            inputs=inputs + param_input_names,
            outputs=outputs,
            nodes=function_graph.node,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid(CUSTOM_DOMAIN, self.opset),
            ],
            attributes=[],
            value_info=combined_value_infos,
        )

        self.functions[name] = function_proto
        print(f"🧩 Stored FunctionProto: {name} with {len(function_graph.node)} nodes")

    def add_function_call_node(
        self,
        function_name: str,
        inputs: List[str],
        outputs: List[str],
    ) -> None:
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
        return [
            self._build_function_proto(graph, "") for graph in self.functions.values()
        ]

    def create_function_proto(
        self, graph: GraphProto, domain: str = ""
    ) -> FunctionProto:
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

    def _adjust_tensor_shape(self, tensor, shape, batch_dims):
        tensor_shape = tensor.type.tensor_type.shape.dim
        for idx, dim in enumerate(shape):
            if dim == "B":
                tensor_shape[idx].dim_param = "B"
        for idx in batch_dims:
            if idx < len(tensor_shape):
                tensor_shape[idx].dim_param = "B"

    def adjust_dynamic_batch_dimensions(self, input_shapes):
        batch_dims = {
            idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
        }
        for tensor, input_shape in zip(self.inputs, input_shapes):
            self._adjust_tensor_shape(tensor, input_shape, batch_dims)
        for tensor in self.outputs:
            self._adjust_tensor_shape(tensor, [], batch_dims)

    def filter_unused_initializers(self):
        used_inputs = {i for node in self.nodes for i in node.input}
        self.initializers = [
            init for init in self.initializers if init.name in used_inputs or init.name
        ]
