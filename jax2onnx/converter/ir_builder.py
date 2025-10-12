# jax2onnx/converter/ir_builder.py


from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
from onnx_ir._tape import Builder as _TapeBuilder


def _dtype_to_ir(dtype: Optional[np.dtype], enable_double: bool) -> ir.DataType:
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
    try:
        return ir.DataType.from_numpy(key)
    except Exception as e:
        raise TypeError(f"Unsupported dtype: {dtype}") from e


class IRBuilder:
    """
    Minimal IR graph assembler for converter.
    Holds a mapping from jaxpr vars to ir.Values, and accumulates nodes/inputs/outputs.
    """

    def __init__(self, *, opset: int, enable_double_precision: bool):
        self.opset = opset
        self.enable_double_precision = enable_double_precision
        self._tape_builder = _TapeBuilder()
        self.inputs: list[ir.Value] = []
        self.outputs: list[ir.Value] = []
        self.nodes: list[ir.Node] = []
        self.initializers: list[ir.Value] = []
        self.used_opsets: set[tuple[str, int | None]] = self._tape_builder.used_opsets
        self.initializers_by_name: dict[str, ir.Value] = {}
        # Intermediate ValueInfo entries (propagated to ir.Graph)
        self._function_mode: bool = False
        self._var2val: dict[Any, ir.Value] = {}
        self._counters: dict[str, int] = {}
        # optional: symbolic dim origins used by some plugins
        self._sym_origin: dict[str, tuple[ir.Value, int]] = {}
        self._tape_node_index = 0
        self._tape_initializer_index = 0

    # ---------- naming ----------
    def fresh_name(self, base: str) -> str:
        i = self._counters.get(base, 0)
        self._counters[base] = i + 1
        return f"{base}_{i}"

    def _sync_from_tape_builder(self) -> None:
        tape_nodes = self._tape_builder.nodes
        for node in tape_nodes[self._tape_node_index :]:
            self.nodes.append(node)
        self._tape_node_index = len(tape_nodes)

        tape_initializers = self._tape_builder.initializers
        for value in tape_initializers[self._tape_initializer_index :]:
            init_name = value.name or None
            existing = (
                self.initializers_by_name.get(init_name)
                if init_name is not None
                else None
            )
            if existing is not None:
                try:
                    idx = self.initializers.index(existing)
                    self.initializers[idx] = value
                except ValueError:
                    self.initializers.append(value)
            else:
                self.initializers.append(value)
            if init_name is not None:
                self.initializers_by_name[init_name] = value
        self._tape_initializer_index = len(tape_initializers)

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
        if self._function_mode:
            v = ir.Value(
                name=name,
                shape=ir.Shape(arr.shape if arr.shape else ()),
                type=ir.TensorType(
                    _dtype_to_ir(arr.dtype, self.enable_double_precision)
                ),
                const_value=tensor,
            )
            attributes = [
                Attr("value", AttributeType.TENSOR, tensor),
            ]
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
        v = self._tape_builder.initializer(tensor, name=name)
        self._sync_from_tape_builder()
        # overwrite-safe: last wins
        self.initializers_by_name[name] = v
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
                dt = spec.dtype if hasattr(spec, "dtype") else None
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
        aval = var.aval if hasattr(var, "aval") else None
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

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        tape_builder = object.__getattribute__(self, "_tape_builder")
        try:
            attr = object.__getattribute__(tape_builder, name)
        except AttributeError as err:
            try:
                getattr_hook = object.__getattribute__(
                    type(tape_builder), "__getattr__"
                )
            except AttributeError:
                getattr_hook = None
            if getattr_hook is None:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute {name!r}"
                ) from err
            attr = getattr_hook(tape_builder, name)
        if callable(attr):

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                self._sync_from_tape_builder()
                return result

            return _wrapped
        return attr

    # ---------- symbolic dim origin ----------
    def record_symbol_origin(self, sym: str, src_val: ir.Value, axis: int) -> None:
        self._sym_origin[sym] = (src_val, axis)

    def get_symbolic_dim_origin(self, sym: str) -> Optional[tuple[ir.Value, int]]:
        return self._sym_origin.get(sym)

    def to_ir_model(self, *, name: str, ir_version: int = 11) -> ir.Model:
        self._sync_from_tape_builder()
        graph = ir.Graph(
            inputs=self.inputs,
            outputs=self.outputs,
            nodes=self.nodes,
            initializers=self.initializers,
            name=name or "main_graph",
            opset_imports={"": self.opset},
        )
        return ir.Model(graph, ir_version=ir_version, producer_name="jax2onnx")
