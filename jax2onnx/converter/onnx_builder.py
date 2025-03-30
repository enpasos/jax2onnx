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
from typing import Dict, List, Any, Tuple
from jax.extend.core import Literal
import onnx

# === Import BOTH name generators ===
from jax2onnx.converter.name_generator import GlobalNameCounter, UniqueNameGenerator


CUSTOM_DOMAIN = "custom"
CUSTOM_DOMAIN_VERSION = 1


class OnnxBuilder:
    def __init__(
        self,
        name_counter: GlobalNameCounter,  # <<< Keep original argument
        opset: int = 21,
        model_name: str = "",
        initializers: List[Any] | None = None,
    ) -> None:
        # === Keep original name_counter, add new name_generator ===
        self.name_counter: GlobalNameCounter = name_counter  # Keep original name
        self.name_generator: UniqueNameGenerator = (
            UniqueNameGenerator()
        )  # Add new generator instance

        # --- Rest of __init__ remains the same as original ---
        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = initializers if initializers is not None else []
        self.value_info: List[ValueInfoProto] = []
        self.opset: int = opset
        self.functions: Dict[str, FunctionProto] = {}  # Store FunctionProtos
        self.model_name: str = model_name

    def get_constant_name(self, val):
        if isinstance(val, Literal):
            val = val.val
        np_val = np.array(val)
        if np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)

        name = self.get_unique_name("const")  # Uses original name_counter
        tensor = helper.make_tensor(
            name=name,
            # === Uses CORRECTED _numpy_dtype_to_onnx below ===
            data_type=self._numpy_dtype_to_onnx(np_val.dtype),
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        return name

    def reset(self) -> None:
        # === Reset BOTH generators ===
        self.name_counter = GlobalNameCounter()  # Reset original counter
        self.name_generator = UniqueNameGenerator()  # Reset new generator

        # --- Rest of reset remains the same as original ---
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.functions.clear()

    def get_unique_name(self, prefix: str = "node") -> str:
        # === Uses self.name_counter (original behavior) ===
        return self.name_counter.get(prefix)

    # === ADD the new method using the NEW generator ===
    def get_unique_instance_name(self, base_name: str) -> str:
        """Gets a unique name for a type instance (e.g., ONNX function) using the new generator."""
        # --- Uses the NEW name_generator instance ---
        return self.name_generator.get(base_name)

    # === Fix: Add back add_initializer method ===
    def add_initializer(
        self, name, vals, data_type=helper.TensorProto.INT64, dims=None
    ):
        """Add a tensor initializer to the model's list."""
        if dims is None:
            # Try to infer dims if possible, otherwise maybe require it?
            # For simplicity, assume vals is list-like for len()
            dims = (
                [len(vals)] if isinstance(vals, (list, tuple)) else []
            )  # Basic inference

        tensor = helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=dims,
            vals=vals,  # Should be flattened list if multi-dimensional
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
        # === Uses CORRECTED _numpy_dtype_to_onnx below ===
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
        # === Uses CORRECTED filter_unused_initializers below ===
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

    # === UNCONDITIONALLY add Custom Opset in create_model & Simplify Func Handling ===
    def create_model(self, graph: GraphProto) -> ModelProto:
        opset_imports = [
            helper.make_opsetid("", self.opset),
            # Always include custom opset import with correct version
            helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION),
        ]

        model = helper.make_model(
            graph,
            opset_imports=opset_imports,
            # Pass functions directly using the stored FunctionProtos
            functions=list(self.functions.values()),
        )
        # Optional print
        # if self.functions: print(f"🧠 Added {len(model.functions)} functions via create_model: {[f.name for f in model.functions]}")
        return model

    # === UNCONDITIONALLY add Custom Opset in create_onnx_model & Simplify Func Handling ===
    def create_onnx_model(self, model_name: str) -> onnx.ModelProto:
        graph = self._build_graph(model_name)

        opset_imports = [
            helper.make_opsetid("", self.opset),
            # Always include custom opset import with correct version
            helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION),
        ]

        model = helper.make_model(
            graph,
            opset_imports=opset_imports,
            # Pass functions directly using the stored FunctionProtos
            functions=list(self.functions.values()),
        )
        # Optional print
        # if self.functions: print(f"🧠 Added {len(model.functions)} functions via create_onnx_model: {[f.name for f in model.functions]}")
        return model

    # === CORRECT numpy -> ONNX dtype mapping ===
    def _numpy_dtype_to_onnx(self, dtype: Any) -> int:
        try:
            np_dtype = np.dtype(dtype)
        except TypeError:
            print(
                f"Warning: Could not convert dtype '{dtype}' to numpy dtype. Defaulting to ONNX FLOAT."
            )
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
        onnx_type = dtype_map.get(np_dtype)
        if onnx_type is None:
            print(
                f"Warning: Unsupported numpy dtype {np_dtype} encountered. Defaulting to ONNX FLOAT."
            )
            return TensorProto.FLOAT
        return onnx_type

    # === Simplify Function Handling: Modify add_function & Add Custom Opset Import Inside ===
    def add_function(
        self, name: str, builder: "OnnxBuilder", param_input_names: List[str]
    ) -> None:
        if name in self.functions:
            print(f"Warning: Function {name} already exists. Overwriting.")
        function_graph = builder.create_graph(name + "_internal_graph")
        inputs = [vi.name for vi in function_graph.input]
        outputs = [vi.name for vi in function_graph.output]
        function_proto = helper.make_function(
            domain=CUSTOM_DOMAIN,
            fname=name,
            inputs=inputs + param_input_names,
            outputs=outputs,
            nodes=function_graph.node,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid(
                    CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION
                ),  # Use correct version
            ],
        )
        self.functions[name] = function_proto
        # print(f"🧩 Stored FunctionProto: {name} with {len(function_graph.node)} nodes")

    def add_function_call_node(
        self,
        function_name: str,
        inputs: List[str],
        outputs: List[str],
    ) -> None:
        if function_name not in self.functions:
            raise ValueError(
                f"Function '{function_name}' not defined in domain '{CUSTOM_DOMAIN}'. Available: {list(self.functions.keys())}"
            )
        call_node_name = self.get_unique_name(
            f"call_{function_name}"
        )  # Use original name counter
        node = helper.make_node(
            op_type=function_name,
            inputs=inputs,
            outputs=outputs,
            name=call_node_name,
            domain=CUSTOM_DOMAIN,
        )
        self.nodes.append(node)
        # print(f"🧩 Added function call node '{call_node_name}': {function_name}({', '.join(inputs)}) -> {', '.join(outputs)}")

    # === Remove Redundant Function Helpers ===
    # def _build_function_proto( ... ) # Removed
    # def create_functions(self) -> List[FunctionProto]: # Removed
    # def create_function_proto( ... ) # Removed

    # --- adjust_dynamic_batch_dimensions (kept as is) ---
    def _adjust_tensor_shape(self, tensor, shape_hint, batch_dims):
        if not tensor.type.HasField(
            "tensor_type"
        ) or not tensor.type.tensor_type.HasField("shape"):
            return
        tensor_dims = tensor.type.tensor_type.shape.dim
        num_tensor_dims = len(tensor_dims)
        for idx, dim_symbol in enumerate(shape_hint):
            if idx < num_tensor_dims and dim_symbol == "B":
                tensor_dims[idx].ClearField("dim_value")
                tensor_dims[idx].dim_param = "B"
        for idx in batch_dims:
            if idx < num_tensor_dims:
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
            print(
                f"Warning: Input shapes hints ({num_hints}) != model inputs ({num_inputs}). Skipping."
            )
        else:
            for tensor, input_shape_hint in zip(self.inputs, input_shapes):
                self._adjust_tensor_shape(tensor, input_shape_hint, batch_dims)
        for tensor in self.outputs:
            self._adjust_tensor_shape(tensor, [], batch_dims)

    # === CORRECT filter_unused_initializers logic ===
    def filter_unused_initializers(self):
        used_inputs = {inp for node in self.nodes for inp in node.input}
        initializers_count_before = len(self.initializers)
        self.initializers = [
            init for init in self.initializers if init.name in used_inputs
        ]
        initializers_count_before - len(self.initializers)
        # if removed_count > 0: print(f"Removed {removed_count} unused initializers.")
