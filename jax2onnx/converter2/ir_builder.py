# file: jax2onnx/converter2/ir_builder.py


from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import onnx_ir as ir


_NP_TO_IR_BASE = {
    np.dtype(np.float32): "FLOAT",
    np.dtype(np.float64): "DOUBLE",
    np.dtype(np.int32): "INT32",
    np.dtype(np.int64): "INT64",
    np.dtype(np.uint8): "UINT8",
    np.dtype(np.int8): "INT8",
    np.dtype(np.uint16): "UINT16",
    np.dtype(np.int16): "INT16",
    np.dtype(np.uint32): "UINT32",
    np.dtype(np.uint64): "UINT64",
    np.dtype(np.bool_): "BOOL",
}


def _dtype_to_ir(dtype: Optional[np.dtype], enable_double: bool) -> "ir.DataType":
    """
    Map numpy dtype to onnx_ir.DataType.
    Floats are normalized by enable_double flag.
    """
    if dtype is None:
        return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
    key = np.dtype(dtype)
    if np.issubdtype(key, np.floating):
        if key == np.float16:
            return ir.DataType.FLOAT16
        if key == np.float32:
            return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
        if key == np.float64:
            return ir.DataType.DOUBLE
        return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
    name = _NP_TO_IR_BASE.get(key)
    if name:
        return getattr(ir.DataType, name)
    if np.issubdtype(key, np.integer):
        return ir.DataType.INT64
    raise TypeError(f"Unsupported dtype: {dtype}")


class IRBuilder:
    """
    Minimal IR graph assembler for converter2.
    Holds a mapping from jaxpr vars to ir.Values, and accumulates nodes/inputs/outputs.
    """

    def __init__(self, *, opset: int, enable_double_precision: bool):
        self.opset = opset
        self.enable_double_precision = enable_double_precision
        self.inputs: list[ir.Value] = []
        self.outputs: list[ir.Value] = []
        self.nodes: list[ir.Node] = []
        self.initializers: list[ir.Value] = []
        self.initializers_by_name: dict[str, ir.Value] = {}
        # Intermediate ValueInfo entries (propagated to ir.Graph)
        self.value_info: list[ir.Value] = []
        self._function_mode: bool = False
        self._var2val: dict[Any, ir.Value] = {}
        self._counters: dict[str, int] = {}
        # optional: symbolic dim origins used by some plugins
        self._sym_origin: dict[str, tuple[ir.Value, int]] = {}

    # ---------- naming ----------
    def fresh_name(self, base: str) -> str:
        i = self._counters.get(base, 0)
        self._counters[base] = i + 1
        return f"{base}_{i}"

    # ---------- values ----------
    def _make_value(
        self, name: str, shape: Tuple[Any, ...], np_dtype: Optional[np.dtype]
    ) -> ir.Value:
        dtype_enum = _dtype_to_ir(np_dtype, self.enable_double_precision)
        return ir.Value(
            name=name, shape=ir.Shape(shape), type=ir.TensorType(dtype_enum)
        )

    # public helpers for initializers (used by FunctionPlugin)
    def add_initializer_from_scalar(self, name: str, value: Any) -> ir.Value:
        arr = np.asarray(value)
        if not self.enable_double_precision and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        tensor = ir.tensor(arr)
        v = ir.Value(
            name=name,
            shape=ir.Shape(arr.shape if arr.shape else ()),
            type=ir.TensorType(_dtype_to_ir(arr.dtype, self.enable_double_precision)),
            const_value=tensor,
        )
        if getattr(self, "_function_mode", False):
            Attr = getattr(ir, "Attr", getattr(ir, "Attribute", None))
            AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
            attributes: list[Any] = []
            if Attr is not None:
                try:
                    if hasattr(Attr, "t"):
                        attributes.append(Attr.t("value", tensor))
                    elif AttrType is not None:
                        attributes.append(Attr("value", AttrType.TENSOR, tensor))
                    else:
                        attributes.append(Attr("value", tensor))
                except Exception:
                    pass
            node = ir.Node(
                op_type="Constant",
                domain="",
                inputs=[],
                outputs=[v],
                name=self.fresh_name("Constant"),
                attributes=attributes,
            )
            self.nodes.append(node)
            return v
        # overwrite-safe: last wins
        self.initializers_by_name[name] = v
        # keep list for stable order
        self.initializers.append(v)
        return v

    def add_initializer_from_array(self, name: str, array: np.ndarray) -> ir.Value:
        return self.add_initializer_from_scalar(name, np.asarray(array))

    # convenient I64 consts for shape ops
    def const_i64(self, name: str, values: Sequence[int]) -> ir.Value:
        arr = np.asarray(values, dtype=np.int64)
        return self.add_initializer_from_array(name, arr)

    # bind graph inputs from specs
    def add_inputs_from_specs(
        self, invars: Sequence[Any], specs: Sequence[Any]
    ) -> None:
        """
        Bind jaxpr invars to graph inputs using the provided input specs.
        """
        for i, (var, spec) in enumerate(zip(invars, specs)):
            if hasattr(spec, "shape"):
                shp = tuple(spec.shape)
                dt = getattr(spec, "dtype", None)
            elif isinstance(spec, (tuple, list)):
                shp = tuple(spec)
                dt = None
            else:
                raise TypeError(f"Unsupported spec for graph input: {type(spec)}")
            v = self._make_value(
                name=f"x{i}",
                shape=shp,
                np_dtype=(np.dtype(dt) if dt is not None else None),
            )
            self._var2val[var] = v
            self.inputs.append(v)

    def get_value_for_var(
        self, var: Any, *, name_hint: Optional[str] = None
    ) -> ir.Value:
        """
        Return an ir.Value for a jaxpr var; create it from aval if needed.
        """
        if var in self._var2val:
            return self._var2val[var]
        aval = getattr(var, "aval", None)
        if aval is None:
            raise ValueError(f"Missing aval for var: {var}")
        shp = tuple(aval.shape)
        try:
            np_dt = np.dtype(aval.dtype)
        except Exception:
            np_dt = None
        v = self._make_value(
            name=name_hint or self.fresh_name("v"), shape=shp, np_dtype=np_dt
        )
        self._var2val[var] = v
        return v

    def add_outputs_from_vars(self, outvars: Sequence[Any]) -> None:
        for i, var in enumerate(outvars):
            v = self.get_value_for_var(var, name_hint=f"y{i}")
            self.outputs.append(v)

    # ---------- nodes ----------
    def add_node_obj(self, node: ir.Node) -> None:
        self.nodes.append(node)

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value],
        outputs: Sequence[ir.Value],
        attributes: Optional[list[ir.Attr]] = None,
        name: Optional[str] = None,
    ) -> ir.Node:
        n = ir.Node(
            op_type=op_type,
            domain="",
            inputs=list(inputs),
            outputs=list(outputs),
            name=name or self.fresh_name(op_type),
            attributes=(attributes or []),
        )
        self.nodes.append(n)
        return n

    # ---------- symbolic dim origin ----------
    def record_symbol_origin(self, sym: str, src_val: ir.Value, axis: int) -> None:
        self._sym_origin[sym] = (src_val, axis)

    def get_symbolic_dim_origin(self, sym: str) -> Optional[tuple[ir.Value, int]]:
        return self._sym_origin.get(sym)

    # ---------- finalize (IR only) ----------
    def to_ir_model(self, *, name: str, ir_version: int = 10) -> "ir.Model":
        graph = ir.Graph(
            inputs=self.inputs,
            outputs=self.outputs,
            nodes=self.nodes,
            initializers=self.initializers,
            name=name or "jax2onnx_ir_graph",
            opset_imports={"": self.opset},
        )
        if self.value_info:
            vi = list(self.value_info)
            if hasattr(graph, "value_info"):
                graph.value_info = vi
            elif hasattr(graph, "_value_info"):
                graph._value_info = vi
        return ir.Model(graph, ir_version=ir_version)
