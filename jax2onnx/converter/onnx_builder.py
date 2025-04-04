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


def _as_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


class OnnxBuilder:
    def __init__(
        self,
        name_counter: GlobalNameCounter,
        name_generator: UniqueNameGenerator,
        opset: int = 21,
        model_name: str = "",
        initializers: Optional[List[Any]] = None,
    ) -> None:
        self.name_counter: GlobalNameCounter = name_counter
        self.name_generator: UniqueNameGenerator = name_generator

        self.nodes: List[NodeProto] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[Any] = initializers if initializers is not None else []
        self.value_info: List[ValueInfoProto] = []
        self.opset: int = opset
        self.functions: Dict[str, FunctionProto] = {}
        self.model_name: str = model_name
        self.display_name_map: Dict[str, str] = {}
        self.value_info_metadata: Dict[str, Tuple[Tuple[int, ...], Any]] = (
            {}
        )  # name -> (shape, dtype)
        self.dtype_env: Dict[str, onnx.TensorProto.DataType] = {}

    def register_value_info_metadata(
        self, name: str, shape: Tuple[int, ...], dtype: Any
    ):
        self.value_info_metadata[name] = (shape, dtype)

    def find_missing_value_info(self) -> List[str]:
        known_names = {vi.name for vi in self.inputs + self.outputs + self.value_info}
        known_names.update(init.name for init in self.initializers)
        node_names = {
            name for n in self.nodes for name in list(n.input) + list(n.output)
        }
        return sorted(name for name in node_names if name not in known_names)

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

        name = self.get_unique_instance_name("const")
        tensor = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)

        # ✅ Ensure shape is a tuple of integers
        self.register_value_info_metadata(
            name, shape=tuple(np_val.shape), dtype=np_val.dtype
        )

        # ✅ Auto-fix fallback value_info for edge cases like tile_repeats, etc.
        self._auto_fix_constant_value_info(name, np_val)

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
        self.value_info_metadata.clear()

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

        # ✅ Auto-fix value_info metadata
        self._auto_fix_constant_value_info(name, np.array(flat_vals).reshape(dims))

        return name

    def _add_tensor(
        self,
        collection: List[ValueInfoProto],
        name: str,
        shape: Optional[Tuple[int, ...]],
        dtype: Any,
    ):
        shape = _as_tuple(shape)
        tensor_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        collection.append(tensor_def)

    def add_input(
        self, name: str, shape: Optional[Tuple[int, ...]], dtype: Any = np.float32
    ) -> None:
        self.dtype_env[name] = dtype
        self._add_tensor(self.inputs, name, shape, dtype)

    def add_output(
        self, name: str, shape: Optional[Tuple[int, ...]], dtype: Any = np.float32
    ) -> None:
        self.dtype_env[name] = dtype
        self._add_tensor(self.outputs, name, shape, dtype)

    def add_value_info(
        self, name: str, shape: Optional[Tuple[int, ...]], dtype: Any = np.float32
    ) -> None:
        self.dtype_env[name] = dtype
        self._add_tensor(self.value_info, name, shape, dtype)

    def create_node(
        self, op_type: str, inputs: List[str], outputs: List[str], **kwargs: Any
    ) -> NodeProto:
        return helper.make_node(op_type, inputs, outputs, **kwargs)

    def add_node(self, node: NodeProto) -> None:
        self.nodes.append(node)

    def _build_graph(self, name: str) -> GraphProto:
        self.filter_unused_initializers()
        missing = self.find_missing_value_info()
        if missing:
            raise RuntimeError(
                f"Missing value_info for: {missing}\n\nConsider adding them using `builder.add_value_info(...)` or `register_value_info_metadata(...)`"
            )
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

        unique_function_protos = list(
            {f.name: f for f in self.functions.values()}.values()
        )

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
        self, name: str, sub_builder: "OnnxBuilder", param_input_names: List[str]
    ) -> str:
        missing = sub_builder.find_missing_value_info()
        if missing:
            raise RuntimeError(
                f"Missing value_info in function '{name}': {missing}\n\nFix the corresponding plugin using `register_value_info_metadata(...)`"
            )

        function_graph = sub_builder.create_graph(name + "_graph")
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
                helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION),
            ],
        )

        self.functions[name] = function_proto

        return name

    def _get_shape(self, vi):
        if hasattr(vi, "type") and hasattr(vi.type, "tensor_type"):
            shape_proto = vi.type.tensor_type.shape
            return [
                d.dim_value if d.HasField("dim_value") else None
                for d in shape_proto.dim
            ]
        return None

    def _get_dtype(self, vi):
        if hasattr(vi, "type") and hasattr(vi.type, "tensor_type"):
            return vi.type.tensor_type.elem_type
        return TensorProto.FLOAT  # default fallback

    def _register_value_info_for_function_inputs_outputs_and_intermediates(
        self, func: onnx.FunctionProto, input_names: List[str], output_names: List[str]
    ):

        # Inputs
        for func_input_name, outer_input_name in zip(func.input, input_names):
            vi = next((v for v in self.value_info if v.name == outer_input_name), None)
            if vi:
                self.add_value_info(
                    func_input_name, self._get_shape(vi), self._get_dtype(vi)
                )
            elif outer_input_name in self.value_info_metadata:
                shape, dtype = self.value_info_metadata[outer_input_name]
                self.add_value_info(func_input_name, shape, dtype)

        # Outputs
        for func_output_name, outer_output_name in zip(func.output, output_names):
            vi = next((v for v in self.value_info if v.name == outer_output_name), None)
            if vi:
                self.add_value_info(
                    func_output_name, self._get_shape(vi), self._get_dtype(vi)
                )
            elif outer_output_name in self.value_info_metadata:
                shape, dtype = self.value_info_metadata[outer_output_name]
                self.add_value_info(func_output_name, shape, dtype)

        # Intermediates
        all_known = set(func.input) | set(func.output)
        for node in func.node:
            for name in list(node.input) + list(node.output):
                if (
                    name
                    and name not in all_known
                    and name not in self.value_info_metadata
                ):
                    self.add_value_info(name, None, TensorProto.FLOAT)

    def _register_value_info_if_missing(self, name: str):
        if name not in self.value_info:
            shape, dtype = self.value_info_metadata.get(name, (None, TensorProto.FLOAT))
            if shape is None:
                # fallback for debugging
                print(f"[WARN] Missing metadata for: {name} — using fallback")
                shape = ()  # or None
            print(
                f"[INFO] Registering value_info: {name}, shape={shape}, dtype={dtype}"
            )
            self.add_value_info(name, shape, dtype)

    def _auto_fix_constant_value_info(self, name: str, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        # Shape + Type
        shape = list(value.shape)
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE.get(value.dtype)

        # 👇 Patch: Special case for scalar lists that are meant to be INT64 (e.g. axes, shape)
        if dtype == TensorProto.FLOAT and name.endswith(
            ("_shape", "_axes", "_repeats", "_starts", "_ends")
        ):
            dtype = TensorProto.INT64
            value = value.astype(np.int64)  # Also patch the initializer

        # Ensure shape is a tuple of integers
        self.register_value_info_metadata(name, shape=tuple(shape), dtype=dtype)

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
            readable_base = (user_display_name or function_name).split(".")[-1]
            node_name = self.get_unique_instance_name(readable_base)
        else:
            node_name = node_name.split(".")[-1]

        func = self.functions.get(function_name)

        # 🧠 Register value info for function inputs, outputs, and intermediates
        if func:
            self._register_value_info_for_function_inputs_outputs_and_intermediates(
                func, input_names, output_names
            )

            # 🧠 Recursively register inner functions too
            for node in func.node:
                if node.op_type in self.functions:
                    inner_func = self.functions[node.op_type]
                    self._register_value_info_for_function_inputs_outputs_and_intermediates(
                        inner_func,
                        input_names=list(node.input),
                        output_names=list(node.output),
                    )

        # 🧠 Ensure all input value info is available
        for input in input_names:
            self._register_value_info_if_missing(input)

        # 🧠 Ensure all output value info is available
        for output in output_names:
            if not any(v.name == output for v in self.value_info):
                dtype = None
                if func:
                    # Try from declared outputs
                    for vi in func.output:
                        if vi == output:
                            # This is a string comparison — probably incorrect
                            continue
                        if hasattr(vi, "name") and vi.name == output:
                            shape = self._get_shape(vi)
                            dtype = self._get_dtype(vi)
                            self.add_value_info(output, shape, dtype)
                            break
                    else:
                        # Try to infer from writer node
                        for node in func.node:
                            if output in node.output:
                                op = node.op_type
                                if op in [
                                    "Shape",
                                    "Size",
                                    "Range",
                                    "Cast",
                                    "Squeeze",
                                    "Unsqueeze",
                                    "Concat",
                                    "Slice",
                                    "ConstantOfShape",
                                ]:
                                    dtype = TensorProto.INT64
                                elif op == "Cast":
                                    for attr in node.attribute:
                                        if attr.name == "to":
                                            dtype = attr.i
                                break

                # If still not added
                if not any(v.name == output for v in self.value_info):
                    if dtype is None:
                        if any(
                            suffix in output
                            for suffix in [
                                "_shape",
                                "_dims",
                                "_axes",
                                "_starts",
                                "_ends",
                                "_repeats",
                            ]
                        ):
                            dtype = TensorProto.INT64
                        else:
                            dtype = TensorProto.FLOAT
                    print(
                        f"⚠️ Warning: output '{output}' missing type info, defaulting to {TensorProto.DataType.Name(dtype)}."
                    )
                    # shape =
                    # self.register_value_info_metadata(output, shape=shape, dtype=dtype)
                    self._register_value_info_if_missing(output)

        # ✅ Create function call node
        node = helper.make_node(
            op_type=op_type or node_name,
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

    def merge_value_info_metadata_from(self, other: "OnnxBuilder"):
        for name, (shape, dtype) in other.value_info_metadata.items():
            if name not in self.value_info_metadata:
                self.value_info_metadata[name] = (shape, dtype)
            else:
                # Optional: warn if there's a mismatch
                existing = self.value_info_metadata[name]
                if existing != (shape, dtype):
                    print(
                        f"⚠️ [merge] Mismatch in value_info for '{name}': existing={existing}, new={(shape, dtype)}"
                    )
