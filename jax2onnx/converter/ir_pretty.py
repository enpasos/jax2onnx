# jax2onnx/converter/ir_pretty.py

from __future__ import annotations

from collections.abc import Mapping, Sequence as SequenceABC
from typing import Any, Iterable, List, Sequence, cast
import numpy as np

import onnx_ir as ir

from jax2onnx.converter.function_scope import FunctionDef  # noqa: F401


# ------------------------------ helpers ------------------------------


def _indent(lines: Iterable[str], n: int = 2) -> List[str]:
    pad = " " * n
    return [pad + s for s in lines]


def _graph_initializers(graph: ir.Graph) -> Iterable[ir.Value]:
    initializers_obj = graph.initializers
    if hasattr(initializers_obj, "values"):
        return cast(Iterable[ir.Value], initializers_obj.values())
    return cast(Iterable[ir.Value], initializers_obj)


def _dtype_str(v_or_type: Any) -> str:
    # ir.TensorType(elem_type=ir.DataType.FLOAT) or elem_type directly
    dtype: Any | None
    if isinstance(v_or_type, ir.Value):
        dtype = v_or_type.dtype
    elif hasattr(v_or_type, "dtype"):
        dtype = cast(Any, v_or_type).dtype
    else:
        dtype = v_or_type
    if isinstance(dtype, ir.DataType):
        return dtype.name
    if dtype is None:
        return "tensor(?)"
    return str(dtype)


def _shape_str(v_or_shape: Any) -> str:
    dims: Sequence[Any] | None = None
    if isinstance(v_or_shape, ir.Value):
        shape = v_or_shape.shape
        dims = shape.dims if shape else None
    elif isinstance(v_or_shape, ir.Shape):
        dims = v_or_shape.dims
    elif isinstance(v_or_shape, (list, tuple)):
        dims = v_or_shape
    if dims is None:
        return "?"
    formatted: List[str] = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            formatted.append(str(int(d)))
        elif d is None:
            formatted.append("?")
        else:
            s = str(d)
            formatted.append(s if s else "?")
    return "×".join(formatted)


def _value_meta(v: Any) -> str:
    if isinstance(v, ir.Value):
        return f"{_dtype_str(v)} [{_shape_str(v)}]"
    return "tensor(?) [?]"


def _vname(x: Any) -> str:
    # Accept either ir.Value or plain string names
    if x is None:
        return ""
    if isinstance(x, ir.Value):
        return x.name or ""
    if isinstance(x, str):
        return x
    return str(x)


def _short_tensor(val: Any) -> str:
    """
    Best-effort summary of a tensor-ish attribute payload (const/Attr).
    """
    # onnx_ir tensor wrapper often has .to_numpy()
    if hasattr(val, "to_numpy"):
        try:
            arr = val.to_numpy()
            return f"tensor({arr.dtype}, shape={tuple(arr.shape)})"
        except Exception:
            pass
    # raw numpy
    try:
        arr = np.asarray(val)
        if arr.ndim == 0:
            return f"scalar({arr.dtype}, {arr.item()})"
        return f"tensor({arr.dtype}, shape={tuple(arr.shape)})"
    except Exception:
        pass
    # fallback
    s = str(val)
    return s if len(s) <= 64 else (s[:61] + "...")


def _attr_kv_lines(attrs: Any) -> List[str]:
    """
    Render attribute sequences/mappings into "key: value" strings.
    """
    lines: List[str] = []
    if attrs is None:
        return lines

    def _append(name: str, value: Any) -> None:
        key = name or "attr"
        if key == "value":
            lines.append(f"{key}: {_short_tensor(value)}")
        else:
            lines.append(f"{key}: {value}")

    if isinstance(attrs, Mapping):
        for key, raw_value in attrs.items():
            if isinstance(raw_value, ir.Attr):
                _append(raw_value.name or key, raw_value.value)
            else:
                _append(key, raw_value)
        return lines

    if isinstance(attrs, SequenceABC) and not isinstance(attrs, (str, bytes)):
        for entry in attrs:
            if isinstance(entry, ir.Attr):
                _append(entry.name, entry.value)
            elif isinstance(entry, tuple) and len(entry) == 2:
                name, value = entry
                _append(str(name), value)
            else:
                candidate_name = entry.name if hasattr(entry, "name") else ""
                candidate_value = entry.value if hasattr(entry, "value") else entry
                _append(candidate_name, candidate_value)
    return lines


# ----------------------------- IR model ------------------------------


def format_ir_model(m: ir.Model, *, show_initializers: bool = True) -> str:
    """
    Pretty-print an onnx_ir.Model without converting to ONNX.
    Includes graph I/O, initializers (optional), and node list with attributes.
    """
    g = m.graph
    graph_name = g.name or "<unnamed>"
    out: List[str] = []
    out.append(
        f"IR-Model name={graph_name} ir={m.ir_version} opset={dict(m.opset_imports)}"
    )
    out.append("Graph:")

    # Inputs
    graph_inputs = list(g.inputs)
    if graph_inputs:
        out.append("  Inputs:")
        for v in graph_inputs:
            out.append(f"    - {v.name}: {_value_meta(v)}")

    # Outputs
    graph_outputs = list(g.outputs)
    if graph_outputs:
        out.append("  Outputs:")
        for v in graph_outputs:
            out.append(f"    - {v.name}: {_value_meta(v)}")

    # Initializers
    inits = list(_graph_initializers(g))
    if show_initializers and inits:
        out.append(f"  Initializers: {len(inits)}")
        for v in inits:
            name = v.name
            if not name:
                out.append(f"    - <unnamed initializer>: {_value_meta(v)}")
                continue
            cv = v.const_value
            meta = _value_meta(v)
            if cv is not None:
                meta += f"  <- {_short_tensor(cv)}"
            out.append(f"    - {name}: {meta}")

    # Nodes
    nodes = list(g.all_nodes())
    out.append(f"  Nodes: {len(nodes)}")
    for i, n in enumerate(nodes):
        nm = n.name or ""
        dom = n.domain or ""
        node_inputs = ", ".join(_vname(v) for v in n.inputs if v is not None)
        node_outputs = ", ".join(_vname(v) for v in n.outputs if v is not None)
        out.append(f"    [{i}] {n.op_type}{'(' + dom + ')' if dom else ''}  {nm}")
        out.extend(_indent([f"inputs:  {node_inputs}", f"outputs: {node_outputs}"]))
        attr_lines = _attr_kv_lines(n.attributes)
        if attr_lines:
            out.extend(_indent(["attributes:"] + _indent(attr_lines), n=6))
    return "\n".join(out)


# ------------------------ converter FunctionDef ----------------------


def format_function_defs(funcs: Sequence["FunctionDef"]) -> str:
    """
    Pretty-print converter FunctionDefs (from function_scope.py) directly from
    their IR nodes/values, without converting to ONNX.
    """
    out: List[str] = []
    for f in funcs:
        out.append(f"Function '{f.name}' (domain={f.domain or ''})")
        # signature
        sig_in = ", ".join(_vname(v) for v in (f.inputs or []))
        sig_out = ", ".join(_vname(v) for v in (f.outputs or []))
        out.append(f"  signature: ({sig_in}) -> ({sig_out})")
        # I/O meta
        if f.inputs:
            out.append("  inputs:")
            for val in f.inputs:
                out.append(f"    - {val.name}: {_value_meta(val)}")
        if f.outputs:
            out.append("  outputs:")
            for val in f.outputs:
                out.append(f"    - {val.name}: {_value_meta(val)}")
        # nodes
        nodes = list(f.nodes or [])
        out.append(f"  body nodes: {len(nodes)}")
        for i, n in enumerate(nodes):
            nm = n.name or ""
            dom = n.domain or ""
            ins = ", ".join(_vname(v) for v in n.inputs if v is not None)
            outs2 = ", ".join(_vname(v) for v in n.outputs if v is not None)
            out.append(f"    [{i}] {n.op_type}{'(' + dom + ')' if dom else ''}  {nm}")
            out.extend(_indent([f"inputs:  {ins}", f"outputs: {outs2}"]))
            attr_lines = _attr_kv_lines(n.attributes)
            if attr_lines:
                out.extend(_indent(["attributes:"] + _indent(attr_lines), n=6))
        # late attribute overrides (if any)
        ao = f.attr_overrides
        if ao:
            out.append("  late-attr-overrides:")
            for node_name, override_map in ao.items():
                out.append(f"    - {node_name}:")
                for key, attr_value in (override_map or {}).items():
                    if key == "value":
                        out.append(f"        {key}: {_short_tensor(attr_value)}")
                    else:
                        out.append(f"        {key}: {attr_value}")
        out.append("")  # blank line between functions
    return "\n".join(out).rstrip()


# ------------------------------ convenience ------------------------------


def print_ir_model(m: ir.Model, *, show_initializers: bool = True) -> None:
    str = format_ir_model(m, show_initializers=show_initializers)
    print(str)


def print_function_defs(funcs: Sequence[FunctionDef]) -> None:
    print(format_function_defs(funcs))
