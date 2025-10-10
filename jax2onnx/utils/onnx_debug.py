# jax2onnx/utils/onnx_debug.py

from __future__ import annotations
from typing import Iterable, Tuple, Optional
import onnx
from onnx import AttributeProto, onnx_ml_pb2 as om


def _shape_tuple(vi_type: om.TypeProto) -> Optional[Tuple[object, ...]]:
    if not vi_type.HasField("tensor_type") or not vi_type.tensor_type.HasField("shape"):
        return None
    dims = []
    for d in vi_type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return tuple(dims)


def _value_info_map(g: onnx.GraphProto):
    # Map tensor name -> (shape tuple or None)
    mp = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        mp[vi.name] = _shape_tuple(vi.type)
    for init in g.initializer:
        mp[init.name] = tuple(init.dims)
    for n in g.node:
        if n.op_type == "Constant" and n.output:
            for a in n.attribute:
                if a.name == "value" and a.type == AttributeProto.TENSOR and a.t:
                    mp[n.output[0]] = tuple(a.t.dims)
    return mp


def _is_broadcast_compatible(
    a: Tuple[object, ...], b: Tuple[object, ...]
) -> Optional[bool]:
    """Best-effort static check. Returns True/False if decisive, or None if unknown dims prevent a decision."""
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None  # rank-only or unknown rank somewhere
    for da, db in zip(a, b):
        if da in (None, "") or db in (None, ""):
            return None  # unknown at this axis → cannot decide here
        if da == db:
            continue
        if da == 1 or db == 1:
            continue
        return False
    return True


def _walk_loops(
    g: onnx.GraphProto, where="(main)"
) -> Iterable[Tuple[str, str, Tuple[object, ...], Tuple[object, ...]]]:
    """Yield (path, node_name, in0_shape, in1_shape) for each Mul we can see inside Loop/Scan/If subgraphs."""
    mp = _value_info_map(g)

    def _in_shapes(n):
        ins = [i for i in n.input if i]
        shp = [mp.get(i) for i in ins]
        return shp

    for n in g.node:
        if n.op_type == "Mul" and len([i for i in n.input if i]) >= 2:
            s0, s1 = _in_shapes(n)[:2]
            yield where, (n.name or "Mul"), s0, s1

        # Recurse into subgraphs
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH and a.g:
                yield from _walk_loops(a.g, f"{where} → {n.name or n.op_type}.body")
            elif a.type == AttributeProto.GRAPHS and a.graphs:
                for i, sg in enumerate(a.graphs):
                    yield from _walk_loops(sg, f"{where} → {n.name or n.op_type}[{i}]")
    return


def dump_and_flag_incompatible_mul_pairs(model: onnx.ModelProto) -> int:
    """
    Print every Mul inside nested bodies and return the count of pairs that are
    statically **incompatible** (4 vs 5 etc., not covered by 1/broadcast).
    """
    bad = 0
    for where, nm, s0, s1 in _walk_loops(model.graph):
        compat = _is_broadcast_compatible(s0, s1)
        print(f"[MUL] {where} :: {nm} :: {s0} × {s1}  → compatible? {compat}")
        if compat is False:
            bad += 1
    return bad
