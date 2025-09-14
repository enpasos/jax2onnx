# jax2onnx/utils/ir_pretty.py
from __future__ import annotations

from typing import Iterable, List, Sequence, Any
import numpy as np

import onnx_ir as ir

from jax2onnx.converter2.function_scope import FunctionDef  # noqa: F401


# ------------------------------ helpers ------------------------------


def _indent(lines: Iterable[str], n: int = 2) -> List[str]:
    pad = " " * n
    return [pad + s for s in lines]


def _dtype_str(v_or_type: Any) -> str:
    # ir.TensorType(elem_type=ir.DataType.FLOAT) or elem_type directly
    try:
        if isinstance(v_or_type, ir.TensorType):
            et = getattr(v_or_type, "elem_type", None)
        else:
            et = v_or_type
        return str(et) if et is not None else "tensor(?)"
    except Exception:
        return "tensor(?)"


def _shape_str(v_or_shape: Any) -> str:
    try:
        shp = v_or_shape
        # ir.Value -> .shape (ir.Shape) -> .dims
        if hasattr(v_or_shape, "shape"):
            shp = getattr(v_or_shape, "shape")
        dims = getattr(shp, "dims", None)
        if dims is None and isinstance(shp, (list, tuple)):
            dims = shp
        if dims is None:
            return "?"

        def _fmt(d):
            if isinstance(d, (int, np.integer)):
                return str(int(d))
            s = str(d) if d is not None else ""
            return s if s else "?"

        return "×".join(_fmt(d) for d in dims)
    except Exception:
        return "?"


def _value_meta(v: Any) -> str:
    t = getattr(v, "type", None)
    return f"{_dtype_str(t)} [{_shape_str(v)}]"


def _vname(x: Any) -> str:
    # Accept either ir.Value or plain string names
    if isinstance(x, str):
        return x
    return getattr(x, "name", "")


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
    ir.Node.attributes may be:
      - dict[str, python_value or tensor-like]
      - list[ir.Attr] with .name and .value
    """
    lines: List[str] = []
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            if k == "value":
                lines.append(f"{k}: {_short_tensor(v)}")
            else:
                lines.append(f"{k}: {v}")
    elif isinstance(attrs, (list, tuple)):
        for a in attrs:
            k = getattr(a, "name", "") or getattr(a, "key", "") or "attr"
            val = getattr(a, "value", a)
            if k == "value":
                lines.append(f"{k}: {_short_tensor(val)}")
            else:
                lines.append(f"{k}: {val}")
    return lines


# ----------------------------- IR model ------------------------------


def format_ir_model(m: ir.Model, *, show_initializers: bool = True) -> str:
    """
    Pretty-print an onnx_ir.Model without converting to ONNX.
    Includes graph I/O, initializers (optional), and node list with attributes.
    """
    g = m.graph
    out: List[str] = []
    out.append(
        f"IR-Model name={getattr(g, 'name', '<unnamed>')} ir={getattr(m, 'ir_version', '')} opset={getattr(m, 'opset_imports', {})}"
    )
    out.append("Graph:")

    # Inputs
    ins = list(getattr(g, "inputs", []) or [])
    if ins:
        out.append("  Inputs:")
        for v in ins:
            out.append(f"    - {v.name}: {_value_meta(v)}")

    # Outputs
    outs = list(getattr(g, "outputs", []) or [])
    if outs:
        out.append("  Outputs:")
        for v in outs:
            out.append(f"    - {v.name}: {_value_meta(v)}")

    # Initializers
    inits = list(getattr(g, "initializers", []) or [])
    if show_initializers and inits:
        out.append(f"  Initializers: {len(inits)}")
        for v in inits:
            # tolerate strings and mixed objects
            if isinstance(v, str):
                out.append(f"    - {v}: <initializer>")
                continue
            name = getattr(v, "name", None)
            if not name:
                out.append(f"    - <unnamed initializer>: {_value_meta(v)}")
                continue
            cv = getattr(v, "const_value", None)
            meta = _value_meta(v)
            if cv is not None:
                meta += f"  <- {_short_tensor(cv)}"
            out.append(f"    - {name}: {meta}")

    # Nodes
    nodes = list(getattr(g, "nodes", []) or [])
    out.append(f"  Nodes: {len(nodes)}")
    for i, n in enumerate(nodes):
        nm = getattr(n, "name", "") or ""
        dom = getattr(n, "domain", "") or ""
        ins = ", ".join(_vname(v) for v in (getattr(n, "inputs", []) or []))
        outs2 = ", ".join(_vname(v) for v in (getattr(n, "outputs", []) or []))
        out.append(f"    [{i}] {n.op_type}{'(' + dom + ')' if dom else ''}  {nm}")
        out.extend(_indent([f"inputs:  {ins}", f"outputs: {outs2}"]))
        attr_lines = _attr_kv_lines(getattr(n, "attributes", None))
        if attr_lines:
            out.extend(_indent(["attributes:"] + _indent(attr_lines), n=6))
    return "\n".join(out)


# ------------------------ converter2 FunctionDef ----------------------


def format_function_defs(funcs: Sequence["FunctionDef"]) -> str:
    """
    Pretty-print converter2 FunctionDefs (from function_scope.py) directly from
    their IR nodes/values, without converting to ONNX.
    """
    out: List[str] = []
    for f in funcs:
        out.append(f"Function '{f.name}' (domain={f.domain or ''})")
        # signature
        sig_in = ", ".join(v.name for v in (f.inputs or []))
        sig_out = ", ".join(v.name for v in (f.outputs or []))
        out.append(f"  signature: ({sig_in}) -> ({sig_out})")
        # I/O meta
        if f.inputs:
            out.append("  inputs:")
            for v in f.inputs:
                out.append(f"    - {v.name}: {_value_meta(v)}")
        if f.outputs:
            out.append("  outputs:")
            for v in f.outputs:
                out.append(f"    - {v.name}: {_value_meta(v)}")
        # nodes
        nodes = list(getattr(f, "nodes", []) or [])
        out.append(f"  body nodes: {len(nodes)}")
        for i, n in enumerate(nodes):
            nm = getattr(n, "name", "") or ""
            dom = getattr(n, "domain", "") or ""
            ins = ", ".join(
                getattr(v, "name", "") for v in (getattr(n, "inputs", []) or [])
            )
            outs2 = ", ".join(
                getattr(v, "name", "") for v in (getattr(n, "outputs", []) or [])
            )
            out.append(f"    [{i}] {n.op_type}{'(' + dom + ')' if dom else ''}  {nm}")
            out.extend(_indent([f"inputs:  {ins}", f"outputs: {outs2}"]))
            attr_lines = _attr_kv_lines(getattr(n, "attributes", None))
            if attr_lines:
                out.extend(_indent(["attributes:"] + _indent(attr_lines), n=6))
        # late attribute overrides (if any)
        ao = getattr(f, "attr_overrides", None)
        if ao:
            out.append("  late-attr-overrides:")
            for node_name, kv in ao.items():
                out.append(f"    - {node_name}:")
                for k, v in (kv or {}).items():
                    if k == "value":
                        out.append(f"        {k}: {_short_tensor(v)}")
                    else:
                        out.append(f"        {k}: {v}")
        out.append("")  # blank line between functions
    return "\n".join(out).rstrip()


# ------------------------------ convenience ------------------------------


def print_ir_model(m: ir.Model, *, show_initializers: bool = True) -> None:
    str = format_ir_model(m, show_initializers=show_initializers)
    print(str)


def print_function_defs(funcs: Sequence[FunctionDef]) -> None:
    print(format_function_defs(funcs))
