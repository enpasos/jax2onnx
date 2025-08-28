from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

try:
    import onnx_ir as ir
except Exception as e:  # pragma: no cover
    ir = None
    _IR_IMPORT_ERROR = e


_NP_TO_IR_BASE = {
    np.dtype(np.float32): "FLOAT",
    np.dtype(np.float64): "DOUBLE",
    np.dtype(np.int32): "INT32",
    np.dtype(np.int64): "INT64",
    np.dtype(np.uint8): "UINT8",
    np.dtype(np.int8): "INT8",
    np.dtype(np.uint16): "UINT16",
    np.dtype(np.int16): "INT16",
    np.dtype(np.bool_): "BOOL",
}


def _dtype_to_ir(dtype: Optional[np.dtype], enable_double: bool) -> "ir.DataType":
    """
    Map numpy dtype to onnx_ir.DataType.
    Floats are normalized by enable_double flag.
    """
    if dtype is None or np.issubdtype(dtype, np.floating):
        return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
    key = np.dtype(dtype)
    name = _NP_TO_IR_BASE.get(key)
    if not name:
        # Default to FLOAT/FLOAT64 policy for unknown floatlike, else INT64 is a safe integer default.
        return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
    return getattr(ir.DataType, name)


class IRBuilder:
    """
    Minimal IR graph assembler for converter2.
    Holds a mapping from jaxpr vars to ir.Values, and accumulates nodes/inputs/outputs.
    """

    def __init__(self, *, opset: int, enable_double_precision: bool):
        if ir is None:
            raise ImportError("onnx_ir is required") from _IR_IMPORT_ERROR
        self.opset = opset
        self.enable_double_precision = enable_double_precision
        self.inputs: List[ir.Value] = []
        self.outputs: List[ir.Value] = []
        self.nodes: List[ir.Node] = []
        self.initializers: List[ir.Value] = []
        self._var2val: Dict[Any, ir.Value] = {}
        self._counters: Dict[str, int] = {}

    # ---------- naming ----------
    def fresh_name(self, base: str) -> str:
        i = self._counters.get(base, 0)
        self._counters[base] = i + 1
        return f"{base}{i}"

    # ---------- values ----------
    def _make_value(
        self, name: str, shape: Tuple[Any, ...], np_dtype: Optional[np.dtype]
    ) -> ir.Value:
        dtype_enum = _dtype_to_ir(np_dtype, self.enable_double_precision)
        return ir.Value(
            name=name, shape=ir.Shape(shape), type=ir.TensorType(dtype_enum)
        )

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
        name = name_hint or self.fresh_name("v")
        v = self._make_value(name=name, shape=shp, np_dtype=np_dt)
        self._var2val[var] = v
        return v

    def add_outputs_from_vars(self, outvars: Sequence[Any]) -> None:
        for i, var in enumerate(outvars):
            v = self.get_value_for_var(var, name_hint=f"y{i}")
            self.outputs.append(v)

    # ---------- nodes ----------
    def add_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value],
        outputs: Sequence[ir.Value],
        **attrs,
    ):
        node = ir.node(
            op_type=op_type,
            inputs=list(inputs),
            outputs=list(outputs),
            attributes=(attrs or None),
        )
        self.nodes.append(node)

    # ---------- finalize ----------
    def to_model_proto(self, *, name: str, ir_version: int = 10) -> "onnx.ModelProto":
        graph = ir.Graph(
            inputs=self.inputs,
            outputs=self.outputs,
            nodes=self.nodes,
            initializers=self.initializers,
            name=name or "jax2onnx_ir_graph",
            opset_imports={"": self.opset},
        )
        model = ir.Model(graph, ir_version=ir_version)

        # Serialize via onnx_ir -> load as ModelProto (keeps caller API unchanged)
        import tempfile
        import os

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp_path = f.name
            ir.save(model, tmp_path)
            return onnx.load_model(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
