# file: jax2onnx/converter/onnx_builder.py
from onnx import (
    helper,
    TensorProto,
    NodeProto,
    ValueInfoProto,
    ModelProto,
    GraphProto,
    FunctionProto,
)
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from jax.extend.core import Literal
import onnx

# === Import BOTH name generators ===
from jax2onnx.converter.name_generator import GlobalNameCounter, UniqueNameGenerator

CUSTOM_DOMAIN = "custom"
CUSTOM_DOMAIN_VERSION = 1


class OnnxBuilder:
    def __init__(
        self,
        name_counter: GlobalNameCounter,
        opset: int = 21,
        model_name: str = "",
        initializers: Optional[List[Any]] = None,
    ) -> None:
        self.name_counter: GlobalNameCounter = name_counter
        self.name_generator: UniqueNameGenerator = UniqueNameGenerator()

        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = initializers if initializers is not None else []
        self.value_info: List[ValueInfoProto] = []
        self.opset: int = opset
        self.functions: Dict[str, FunctionProto] = {}
        self.model_name: str = model_name
        self.function_name_cache: Dict[str, str] = {}
        self.display_name_map: Dict[str, str] = {}

    def get_constant_name(self, val):
        if isinstance(val, Literal):
            val = val.val
        np_val = np.array(val)
        if np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)
        try:
            onnx_dtype = self._numpy_dtype_to_onnx(np_val.dtype)
        except TypeError:
            print(
                f"Warning: Could not convert value {val} to numpy array. Skipping initializer."
            )
            return self.get_unique_name("invalid_const")

        name = self.get_unique_name("const")
        tensor = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        return name

    def reset(self) -> None:
        self.name_counter = GlobalNameCounter()
        self.name_generator = UniqueNameGenerator()
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.functions.clear()
        self.display_name_map.clear()

    def get_unique_name(self, prefix: str = "node") -> str:
        return self.name_counter.get(prefix)

    def get_unique_instance_name(self, base_name: str) -> str:
        return self.name_generator.get(base_name)

    def add_initializer(
        self, name, vals, data_type=helper.TensorProto.INT64, dims=None
    ):
        if dims is None:
            dims = [len(vals)] if isinstance(vals, (list, tuple)) else []
        flat_vals = np.array(vals).flatten().tolist()
        tensor = helper.make_tensor(
            name=name, data_type=data_type, dims=dims, vals=flat_vals
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
        self.filter_unused_initializers()
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
        return self._finalize_model(graph)

    def create_onnx_model(self, model_name: str) -> onnx.ModelProto:
        graph = self._build_graph(model_name)
        return self._finalize_model(graph)

    def _finalize_model(self, graph: GraphProto) -> ModelProto:
        opset_imports = [
            helper.make_opsetid("", self.opset),
            *(
                [helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION)]
                if self.functions
                else []
            ),
        ]

        # ✅ Keep only distinct functions by name (not serialized body)
        unique_function_protos = list(
            {f.name: f for f in self.functions.values()}.values()
        )

        # Optional: Log any duplicate names (should never happen)
        names = [f.name for f in unique_function_protos]
        seen, duplicates = set(), set()
        for n in names:
            if n in seen:
                duplicates.add(n)
            seen.add(n)
        if duplicates:
            print(f"⚠️ Duplicate ONNX functions detected: {sorted(duplicates)}")
        else:
            print("✅ No duplicate ONNX function names")

        model = helper.make_model(
            graph,
            opset_imports=opset_imports,
            functions=unique_function_protos,
        )
        return model

    def _numpy_dtype_to_onnx(self, dtype: Any) -> int:
        try:
            np_dtype = np.dtype(dtype)
        except TypeError:
            return TensorProto.FLOAT
        dtype_map = {
            np.dtype(np.float32): TensorProto.FLOAT,
            np.dtype(np.float64): TensorProto.DOUBLE,
            np.dtype(np.int32): TensorProto.INT32,
            np.dtype(np.int64): TensorProto.INT64,
            np.dtype(np.bool_): TensorProto.BOOL,
            np.dtype(np.int8): TensorProto.INT8,
            np.dtype(np.uint8): TensorProto.UINT8,
        }
        return dtype_map.get(np_dtype, TensorProto.FLOAT)

    def add_function(
        self,
        name: str,
        sub_builder: "OnnxBuilder",
        param_input_names: List[str],
        user_display_name: Optional[str] = None,
        allow_duplicates: bool = False,
    ) -> str:
        # Determine unique internal function name
        if not allow_duplicates and name in self.function_name_cache:
            internal_name = self.function_name_cache[name]
        else:
            if user_display_name:
                internal_name = f"{user_display_name}_def_{self.name_counter.get(user_display_name)}"
            else:
                internal_name = self.get_unique_name("custom_fn_def")
            if name:
                self.function_name_cache[name] = internal_name

        # Create the function graph
        function_graph = sub_builder.create_graph(internal_name + "_graph")
        inputs = [vi.name for vi in function_graph.input]
        outputs = [vi.name for vi in function_graph.output]
        op_type = user_display_name or name

        function_proto = helper.make_function(
            domain=CUSTOM_DOMAIN,
            fname=op_type,
            inputs=inputs + param_input_names,
            outputs=outputs,
            nodes=function_graph.node,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION),
            ],
        )

        # Register function
        self.functions[internal_name] = function_proto

        # Optional alias registration
        if user_display_name and user_display_name != internal_name:
            self.functions[user_display_name] = function_proto
            self.display_name_map[user_display_name] = internal_name
            print(f"🔄 Aliased function added: {user_display_name} -> {internal_name}")

        return internal_name

    def add_function_call_node(
        self,
        function_name: str,
        input_names: List[str],
        output_names: List[str],
        node_name: Optional[str] = None,
        op_type: Optional[str] = None,
        user_display_name: Optional[str] = None,
    ):
        if node_name is None:
            node_name = self.get_unique_instance_name(function_name)
        if op_type is None:
            op_type = user_display_name or self.display_name_map.get(
                function_name, function_name
            )
        node = helper.make_node(
            op_type,
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            domain=CUSTOM_DOMAIN,
        )
        self.nodes.append(node)

    def _adjust_tensor_shape(self, tensor, shape_hint, batch_dims):
        if not tensor.type.HasField(
            "tensor_type"
        ) or not tensor.type.tensor_type.HasField("shape"):
            return
        tensor_dims = tensor.type.tensor_type.shape.dim
        num_tensor_dims = len(tensor_dims)
        for idx, dim_symbol in enumerate(shape_hint):
            if idx < num_tensor_dims and dim_symbol == "B":
                if tensor_dims[idx].HasField("dim_value"):
                    tensor_dims[idx].ClearField("dim_value")
                tensor_dims[idx].dim_param = "B"
        for idx in batch_dims:
            if idx < num_tensor_dims:
                if tensor_dims[idx].HasField("dim_value"):
                    tensor_dims[idx].ClearField("dim_value")
                tensor_dims[idx].dim_param = "B"

    def adjust_dynamic_batch_dimensions(self, input_shapes):
        batch_dims = {
            idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
        }
        if not batch_dims:
            return
        num_hints = len(input_shapes)
        num_inputs = len(self.inputs)
        if num_hints != num_inputs:
            print("Warning: Input shapes hints != model inputs. Skipping.")
        else:
            for tensor, input_shape_hint in zip(self.inputs, input_shapes):
                self._adjust_tensor_shape(tensor, input_shape_hint, batch_dims)
        for tensor in self.outputs:
            self._adjust_tensor_shape(tensor, [], batch_dims)

    def filter_unused_initializers(self):
        used_inputs = {inp for node in self.nodes for inp in node.input}
        for func_proto in self.functions.values():
            for node in func_proto.node:
                used_inputs.update(node.input)
        self.initializers = [
            init for init in self.initializers if init.name in used_inputs
        ]
