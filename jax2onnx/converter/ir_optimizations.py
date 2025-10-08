# jax2onnx/converter/ir_optimizations.py

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence as SequenceABC
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Set,
    Iterable,
    Any,
    Callable,
    Sequence,
    TypeAlias,
    cast,
)
import copy
import os
import numpy as np

import onnx_ir as ir
from onnx_ir import AttributeType as IRAttrType
from onnx_ir import convenience as ir_convenience

# ---------------- Config ----------------

ALLOWED_ELEMENTWISE_OPS: Set[str] = {
    "elu",
    "gelu",
    "relu",
    "sigmoid",
    "tanh",
    "leakyrelu",
    "identity",
    "cast",
    "castlike",
    "not",
}

ALLOWED_ELEMWISE: Set[str] = {
    "Elu",
    "Gelu",
    "Relu",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Identity",
    "Cast",
    "CastLike",
    "Not",
}

# Unary ops that do not change data shape/dtype (used for propagation)
UNARY_DATAFLOW_OPS: Set[str] = {
    "Gelu",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Dropout",
    "LeakyRelu",
    "Identity",
    "Cast",
    "CastLike",
}

DEBUG: bool = bool(int(os.getenv("JAX2ONNX_IROPT_DEBUG", "0")))
RSH_DEBUG: bool = bool(int(os.getenv("JAX2ONNX_RSH_DEBUG", "0")))
TRN_DEBUG: bool = bool(int(os.getenv("JAX2ONNX_TRN_DEBUG", "0")))
DCE_DEBUG: bool = bool(int(os.getenv("JAX2ONNX_DCE_DEBUG", "0")))
TM_DEBUG: bool = bool(int(os.getenv("JAX2ONNX_TM_DEBUG", "0")))

# ---------------- Type aliases ----------------

NodeList: TypeAlias = List[ir.Node]
NodeSeq: TypeAlias = Sequence[ir.Node]
ValueList: TypeAlias = List[ir.Value]
ValueSeq: TypeAlias = Sequence[ir.Value]

# ---------------- Debug ----------------


def _dbg(*a: object) -> None:
    if DEBUG:
        print("[iropt]", *a)


def _dbg_tm(*a: object) -> None:
    if TM_DEBUG:
        print("[tm-inline]", *a)


# ---------------- Public helper shims (restored for unit tests) ----------------


def _is_elem(op_type: str) -> bool:
    """
    Return True if op_type is a benign elementwise op (case-insensitive).
    """
    if not isinstance(op_type, str):
        return False
    return op_type.lower() in ALLOWED_ELEMENTWISE_OPS


def _get_perm_attr(node: ir.Node) -> Optional[List[int]]:
    """
    Return the Transpose 'perm' attribute as a list of ints, or None.
    """
    attr_obj = _get_attr(node, "perm")
    candidates: Optional[SequenceABC] = None

    if attr_obj is None:
        attrs_raw = getattr(node, "attributes", None)
        if isinstance(attrs_raw, SequenceABC):
            for entry in attrs_raw:
                if getattr(entry, "name", None) == "perm":
                    attr_obj = entry
                    break

    if isinstance(attr_obj, ir.Attr):
        ints_candidate: Optional[SequenceABC] = None
        try:
            as_ints = attr_obj.as_ints()
        except Exception:
            as_ints = None
        if as_ints:
            ints_candidate = list(as_ints)
        if ints_candidate is None:
            value = attr_obj.value
            if isinstance(value, SequenceABC):
                ints_candidate = value
        candidates = ints_candidate
    elif isinstance(attr_obj, SequenceABC):
        candidates = attr_obj
    else:
        ints_value = getattr(attr_obj, "ints", None)
        if isinstance(ints_value, SequenceABC):
            candidates = ints_value
        else:
            value = getattr(attr_obj, "value", None)
            if isinstance(value, SequenceABC):
                candidates = value

    if candidates is None:
        return None
    try:
        return [int(x) for x in candidates]
    except Exception:
        return None


def _perms_compose_identity(p1: Sequence[int], p2: Sequence[int]) -> bool:
    """
    Return True if composing p1 after p2 yields identity.
    (i.e., composed[i] = p1[p2[i]] equals range(len(p1)))
    """
    if not (isinstance(p1, list) and isinstance(p2, list)):
        return False
    if len(p1) != len(p2):
        return False
    try:
        composed = [p1[p] for p in p2]
        return composed == list(range(len(p1)))
    except Exception:
        return False


def _value_identity(
    value_or_name: Optional[object],
) -> Tuple[Optional["ir.Value"], Optional[str]]:
    if value_or_name is None:
        return None, None
    if isinstance(value_or_name, ir.Value):
        return value_or_name, _v_name(value_or_name)
    if isinstance(value_or_name, str):
        return None, value_or_name or None
    return None, None


def _has_input_name_or_obj(
    node: object, name: Optional[str], obj: Optional[object]
) -> bool:
    """
    Return True if 'node' has an input that matches either the given name
    (by .name on Value or string equality) or the given object identity.
    """
    ins = _node_inputs(node)
    if obj is not None:
        for iv in ins:
            if iv is obj:
                return True
    ref_value, ref_name = _value_identity(obj)
    target_name = name or ref_name
    for iv in ins:
        if ref_value is not None and iv is ref_value:
            return True
        if target_name:
            ivn = _v_name(iv)
            if ivn == target_name:
                return True
            # If inputs are plain strings in this build
            if isinstance(iv, str) and iv == target_name:
                return True
    return False


def _count_consumers(
    nodes: Sequence[object], name: Optional[str], obj: Optional[object]
) -> int:
    """
    Count how many nodes consume the given value (by name or object).
    """
    # Prefer IR API when available
    ref_value, ref_name = _value_identity(obj)
    if ref_value is not None:
        cons = ref_value.consumers()
        if isinstance(cons, (list, tuple)):
            return len(cons)

    target_name = name or ref_name
    c = 0
    for n in nodes:
        if _has_input_name_or_obj(n, target_name, ref_value or obj):
            c += 1
    return c


def _find_next_consumer_idx(
    nodes: Sequence[object], start_idx: int, name: Optional[str], obj: Optional[object]
) -> Optional[int]:
    """
    Find the index of the next node (after start_idx) that consumes the given
    value (by name or object). Return None if not found.
    """
    # Prefer the IR API when available, falling back to the legacy scan.
    ref_value, ref_name = _value_identity(obj)
    target_name = name or ref_name
    if ref_value is not None:
        consumers = ref_value.consumers()
        if isinstance(consumers, (list, tuple)):
            node_index = {n: idx for idx, n in enumerate(nodes)}
            for c in consumers:
                idx = node_index.get(c)
                if idx is not None and idx > start_idx:
                    return idx
    for i in range(start_idx + 1, len(nodes)):
        if _has_input_name_or_obj(nodes[i], target_name, ref_value or obj):
            return i
    return None


# ---------------- IR helpers ----------------


def _v_name(v: ir.Value | None) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    name = v.name
    return name or None


def _value_dtype_code(val: Optional[ir.Value]) -> Optional[int]:
    if val is None or isinstance(val, str):
        return None
    # Value may expose dtype directly or via .type
    dtype = val.dtype
    if isinstance(dtype, ir.DataType):
        return int(dtype.value)
    if isinstance(dtype, (int, np.integer)):
        return int(dtype)
    tensor_type = val.type
    if isinstance(tensor_type, ir.TensorType):
        elem_dtype = tensor_type.dtype
        if isinstance(elem_dtype, ir.DataType):
            return int(elem_dtype.value)
        if isinstance(elem_dtype, (int, np.integer)):
            return int(elem_dtype)
    return None


def _node_outputs(n: ir.Node) -> ValueList:
    return list(cast(ValueSeq, n.outputs))


def _node_output(n: ir.Node) -> Optional["ir.Value"]:
    outs = _node_outputs(n)
    return outs[0] if outs else None


def _node_inputs(n: ir.Node) -> ValueList:
    return list(cast(ValueSeq, n.inputs))


def _set_node_inputs(n: ir.Node, new_ins: Sequence["ir.Value"]) -> None:
    for idx, val in enumerate(new_ins):
        n.replace_input_with(idx, val)


def _rebuild_node_with_inputs(n: "ir.Node", new_ins: Sequence["ir.Value"]) -> "ir.Node":
    """
    Construct a fresh ir.Node with the same op_type/name/domain/outputs/attributes,
    but with the provided inputs. This avoids copy-on-write pitfalls in some
    onnx_ir builds where mutating `.inputs`/`.input` doesn't persist.
    """
    _set_node_inputs(n, new_ins)
    return n


def _shape_tuple(v: Optional["ir.Value"]) -> Optional[Tuple[int, ...]]:
    if v is None:
        return None
    shp = v.shape
    if shp is None:
        return None
    dims = shp.dims if isinstance(shp, ir.Shape) else tuple(shp)
    tuple_dims: List[int] = []
    for d in dims:
        if isinstance(d, int):
            tuple_dims.append(d)
        else:
            tuple_dims.append(-1)
    return tuple(tuple_dims)


def _shape_dims_key(shape: object) -> Optional[Tuple[str, ...]]:
    """Return a hashable key representing the shape's dimensions."""
    if shape is None:
        return None
    if isinstance(shape, ir.Shape):
        dims = shape.dims
    else:
        try:
            dims = tuple(shape)
        except Exception:
            return None
    key: List[str] = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            key.append(f"int:{int(d)}")
        else:
            key.append(f"repr:{repr(d)}")
    return tuple(key)


def _clone_shape_obj(shape: object | None) -> ir.Shape | Tuple[Any, ...] | None:
    """Best-effort clone of an onnx_ir Shape object."""
    if shape is None:
        return None
    try:
        if isinstance(shape, ir.Shape):
            dims = shape.dims
        else:
            dims = tuple(shape)
    except Exception:
        try:
            return copy.deepcopy(shape)
        except Exception:
            return None
    norm_dims: List[Any] = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            norm_dims.append(int(d))
        else:
            norm_dims.append(str(d))
    try:
        return ir.Shape(tuple(norm_dims))
    except Exception:
        return tuple(norm_dims)


def _shapes_compatible(a: Optional["ir.Value"], b: Optional["ir.Value"]) -> bool:
    ta, tb = _shape_tuple(a), _shape_tuple(b)
    if ta is None or tb is None or len(ta) != len(tb):
        return False
    for da, db in zip(ta, tb):
        if da == -1 or db == -1:
            continue
        if da != db:
            return False
    return True


def _get_node_seq_and_setter(
    graph: ir.Graph,
) -> Tuple[List["ir.Node"], Callable[[List["ir.Node"]], None]]:
    container = graph._nodes
    snapshot = list(container)

    def _persist(updated_nodes: List[ir.Node]) -> None:
        for existing in list(container):
            container.remove(existing)
        container.extend(updated_nodes)

    return snapshot, _persist


def _replace_everywhere(
    nodes: List["ir.Node"],
    old_v: Optional["ir.Value"],
    old_name: Optional[str],
    new_v: "ir.Value",
) -> None:
    if old_v is not None:
        ir_convenience.replace_all_uses_with(old_v, new_v)
        return
    for m in nodes:
        ins = _node_inputs(m)
        changed = False
        for i, iv in enumerate(ins):
            if (old_v is not None and iv is old_v) or (
                old_name and _v_name(iv) == old_name
            ):
                ins[i] = new_v
                changed = True
            elif isinstance(iv, str) and old_name and iv == old_name:
                new_name = _v_name(new_v)
                ins[i] = new_name if new_name is not None else ""
                changed = True
        if changed:
            _set_node_inputs(m, ins)


def _replace_in_graph_outputs(
    graph: ir.Graph,
    old_v: Optional["ir.Value"],
    old_name: Optional[str],
    new_v: "ir.Value",
) -> None:
    if old_v is not None:
        ir_convenience.replace_all_uses_with(old_v, new_v)

    outputs = graph.outputs
    replaced = False
    for idx, ov in enumerate(outputs):
        if (old_v is not None and ov is old_v) or (
            old_name and _v_name(ov) == old_name
        ):
            outputs[idx] = new_v
            replaced = True

    if not replaced:
        return


def _build_use_maps(
    nodes: NodeSeq,
) -> Tuple[Dict[int, int], Dict[str, int], Dict[int, Set[int]], Dict[str, Set[int]]]:
    prod_by_obj: Dict[int, int] = {}
    prod_by_name: Dict[str, int] = {}
    cons_by_obj: Dict[int, Set[int]] = defaultdict(set)
    cons_by_name: Dict[str, Set[int]] = defaultdict(set)
    for i, n in enumerate(nodes):
        for ov in _node_outputs(n):
            if ov is None:
                continue
            prod_by_obj[id(ov)] = i
            nm = _v_name(ov)
            if nm:
                prod_by_name[nm] = i
    for i, n in enumerate(nodes):
        for iv in _node_inputs(n):
            if iv is None:
                continue
            cons_by_obj[id(iv)].add(i)
            nm = _v_name(iv)
            if nm:
                cons_by_name[nm].add(i)
    return prod_by_obj, prod_by_name, cons_by_obj, cons_by_name


def _unique_consumer(
    cons_by_obj: Dict[int, Set[int]],
    cons_by_name: Dict[str, Set[int]],
    val: Optional["ir.Value"],
) -> Optional[int]:
    if val is None:
        return None
    nm = _v_name(val)
    candidates = set(cons_by_obj.get(id(val), set()))
    if nm:
        candidates |= set(cons_by_name.get(nm, set()))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _producer_idx_for(
    val: Optional["ir.Value"],
    prod_obj: Dict[int, int],
    prod_name: Dict[str, int],
) -> Optional[int]:
    if val is None:
        return None
    nm = _v_name(val)
    return prod_obj.get(id(val)) or (prod_name.get(nm) if nm else None)


def _all_consumers(
    cons_by_obj: Dict[int, Set[int]],
    cons_by_name: Dict[str, Set[int]],
    v: Optional["ir.Value"],
) -> Set[int]:
    if v is None:
        return set()
    nm = _v_name(v)
    consumers = set(cons_by_obj.get(id(v), set()))
    if nm:
        consumers |= set(cons_by_name.get(nm, set()))
    return consumers


# ---------------- Attr access ----------------


def _get_attr(node: ir.Node, name: str) -> Optional[ir.Attr]:
    attrs = node.attributes
    if isinstance(attrs, Mapping):
        candidate = attrs.get(name)
        return candidate if isinstance(candidate, ir.Attr) else None
    if isinstance(attrs, SequenceABC):
        for item in attrs:
            if isinstance(item, ir.Attr) and item.name == name:
                return item
    return None


def _attr_to_int(attr: Any) -> Optional[int]:
    if attr is None:
        return None
    if isinstance(attr, ir.Attr):
        try:
            return int(attr.as_int())
        except Exception:
            try:
                ints = list(attr.as_ints())
            except Exception:
                ints = None
            if ints:
                first = ints[0]
                if isinstance(first, (int, np.integer)):
                    return int(first)
            value = attr.value
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, SequenceABC) and value:
                first = value[0]
                if isinstance(first, (int, np.integer)):
                    return int(first)
        return None
    if isinstance(attr, (int, np.integer)):
        return int(attr)
    if isinstance(attr, SequenceABC) and attr:
        first = attr[0]
        if isinstance(first, (int, np.integer)):
            return int(first)
    return None


def _collect_value_dtypes(graph: ir.Graph, nodes: NodeSeq) -> Dict[str, int]:
    type_map: Dict[str, int] = {}

    def _record(val: Optional[ir.Value]) -> None:
        name = _v_name(val)
        code = _value_dtype_code(val)
        if name and code is not None:
            type_map.setdefault(name, code)

    for value in graph.inputs:
        _record(value)
    for value in graph.outputs:
        _record(value)
    init_container = graph.initializers
    if isinstance(init_container, Mapping):
        init_values = init_container.values()
    else:
        init_values = init_container
    for init in init_values:
        if isinstance(init, ir.Value):
            _record(init)

    for node in nodes:
        for ov in _node_outputs(node):
            _record(ov)
        # also inspect inputs, in case they carry dtype metadata
        for iv in _node_inputs(node):
            _record(iv)

    return type_map


# ---------------- Cast cleanup ----------------


def remove_redundant_casts_ir(graph) -> None:
    nodes, persist = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        dtype_map = _collect_value_dtypes(graph, nodes)
        prod_obj, prod_name, cons_by_obj, cons_by_name = _build_use_maps(nodes)
        for idx, n in enumerate(nodes):
            if n.op_type != "Cast":
                continue
            ins = _node_inputs(n)
            outs = _node_outputs(n)
            if not ins or not outs:
                continue
            target_attr = _get_attr(n, "to")
            target_code = _attr_to_int(target_attr)
            if target_code is None:
                continue
            src_dtype = _value_dtype_code(ins[0])
            if src_dtype is None:
                src_name = _v_name(ins[0])
                if src_name and src_name in dtype_map:
                    src_dtype = dtype_map[src_name]
            if src_dtype is None:
                if DEBUG:
                    _dbg(
                        "skip Cast (unknown dtype)",
                        _v_name(ins[0]),
                        "target",
                        target_code,
                    )
                continue
            if src_dtype != target_code:
                # Try folding consecutive Cast→Cast when net dtype is identity.
                out_val = outs[0]
                consumer_idx = _unique_consumer(cons_by_obj, cons_by_name, out_val)
                if consumer_idx is not None:
                    next_node = nodes[consumer_idx]
                    if next_node.op_type == "Cast":
                        next_outs = _node_outputs(next_node)
                        next_ins = _node_inputs(next_node)
                        if next_outs and next_ins:
                            next_target = _attr_to_int(_get_attr(next_node, "to"))
                            if (
                                next_target is not None
                                and next_target == src_dtype
                                and _unique_consumer(cons_by_obj, cons_by_name, out_val)
                                == consumer_idx
                            ):
                                final_out = next_outs[0]
                                src_val = ins[0]
                                _replace_everywhere(
                                    nodes, final_out, _v_name(final_out), src_val
                                )
                                _replace_in_graph_outputs(
                                    graph, final_out, _v_name(final_out), src_val
                                )
                                kill = sorted([idx, consumer_idx], reverse=True)
                                for k in kill:
                                    nodes.pop(k)
                                persist(nodes)
                                changed = True
                                break
                if DEBUG:
                    _dbg(
                        "skip Cast (dtype mismatch)",
                        _v_name(ins[0]),
                        src_dtype,
                        "→",
                        target_code,
                    )
                continue
            src_val = ins[0]
            out_val = outs[0]
            _replace_everywhere(nodes, out_val, _v_name(out_val), src_val)
            _replace_in_graph_outputs(graph, out_val, _v_name(out_val), src_val)
            nodes = nodes[:idx] + nodes[idx + 1 :]
            persist(nodes)
            changed = True
            break
    persist(nodes)


# ---------------- Transpose folding ----------------


def _transpose_perm(node) -> Optional[List[int]]:
    return _get_perm_attr(node)


def remove_redundant_transpose_pairs_ir(graph) -> None:
    nodes, persist = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        _prod_obj, _prod_name, cons_by_obj, cons_by_name = _build_use_maps(nodes)
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if n.op_type != "Transpose":
                i += 1
                continue
            T1 = n
            T1_out = _node_output(T1)
            nxt_idx = _unique_consumer(cons_by_obj, cons_by_name, T1_out)
            if nxt_idx is None:
                i += 1
                continue
            chain_idx: List[int] = [i]
            allowed_idx: List[int] = []
            cur_idx = nxt_idx
            T2_idx: Optional[int] = None
            steps = 0
            while steps < 8:
                steps += 1
                m = nodes[cur_idx]
                if m.op_type in ALLOWED_ELEMWISE:
                    chain_idx.append(cur_idx)
                    allowed_idx.append(cur_idx)
                    cur_val = _node_output(m)
                    nxt_idx = _unique_consumer(cons_by_obj, cons_by_name, cur_val)
                    if nxt_idx is None:
                        break
                    cur_idx = nxt_idx
                    continue
                if m.op_type == "Transpose":
                    chain_idx.append(cur_idx)
                    T2_idx = cur_idx
                break
            if T2_idx is None:
                i += 1
                continue
            T2 = nodes[T2_idx]
            perm1 = _transpose_perm(T1)
            perm2 = _transpose_perm(T2)
            if perm1 is None or perm2 is None or len(perm1) != len(perm2):
                i += 1
                continue
            composed = [perm1[p] for p in perm2]
            if composed != list(range(len(composed))):
                i += 1
                continue
            if TRN_DEBUG:
                print(
                    "[transposefold]",
                    [nodes[k].op_type for k in chain_idx],
                    "perm1",
                    perm1,
                    "perm2",
                    perm2,
                )
            t1_in = (_node_inputs(T1) or [None])[0]
            if t1_in is None:
                i += 1
                continue
            if allowed_idx:
                first_allowed = nodes[allowed_idx[0]]
                last_allowed = nodes[allowed_idx[-1]]
                _replace_everywhere(
                    [first_allowed], _node_output(T1), _v_name(_node_output(T1)), t1_in
                )
                new_src = _node_output(last_allowed) or t1_in
            else:
                new_src = t1_in
            old_out = _node_output(T2)
            _replace_everywhere(nodes, old_out, _v_name(old_out), new_src)
            _replace_in_graph_outputs(graph, old_out, _v_name(old_out), new_src)
            kill = {i, T2_idx}
            nodes = [m for k, m in enumerate(nodes) if k not in kill]
            persist(nodes)
            changed = True
            break
    persist(nodes)


# ---------------- Reshape folding ----------------


def remove_redundant_reshape_pairs_ir(graph) -> None:
    nodes, persist = _get_node_seq_and_setter(graph)
    if not nodes:
        return

    def _producer_idx_for_local(val, pbo, pbn):
        return _producer_idx_for(val, pbo, pbn)

    changed = True
    while changed:
        changed = False
        prod_obj, prod_name, _cbo, _cbn = _build_use_maps(nodes)
        i = 0
        while i < len(nodes):
            T2 = nodes[i]
            if T2.op_type != "Reshape":
                i += 1
                continue
            v = (_node_inputs(T2) or [None])[0]
            allowed_idxs: List[int] = []
            T1_idx: Optional[int] = None
            steps = 0
            while v is not None and steps < 8:
                steps += 1
                p_idx = _producer_idx_for_local(v, prod_obj, prod_name)
                if p_idx is None:
                    break
                p = nodes[p_idx]
                if p.op_type in ALLOWED_ELEMWISE:
                    allowed_idxs.append(p_idx)
                    v = (_node_inputs(p) or [None])[0]
                    continue
                if p.op_type == "Reshape":
                    T1_idx = p_idx
                break
            if T1_idx is None:
                i += 1
                continue
            T1 = nodes[T1_idx]
            src = (_node_inputs(T1) or [None])[0]
            dst = _node_output(T2)
            if not _shapes_compatible(src, dst):
                i += 1
                continue
            allowed_fwd = list(reversed(allowed_idxs))
            if RSH_DEBUG:
                print(
                    "[reshapefold/up]",
                    [nodes[k].op_type for k in ([T1_idx] + allowed_fwd + [i])],
                    "src",
                    _shape_tuple(src),
                    "dst",
                    _shape_tuple(dst),
                )
            if allowed_fwd:
                first_allowed = nodes[allowed_fwd[0]]
                last_allowed = nodes[allowed_fwd[-1]]
                _replace_everywhere(
                    [first_allowed], _node_output(T1), _v_name(_node_output(T1)), src
                )
                new_src = _node_output(last_allowed) or src
            else:
                new_src = src
            old_out = _node_output(T2)
            _replace_everywhere(nodes, old_out, _v_name(old_out), new_src)
            _replace_in_graph_outputs(graph, old_out, _v_name(old_out), new_src)
            kill = {T1_idx, i}
            nodes = [m for k, m in enumerate(nodes) if k not in kill]
            persist(nodes)
            changed = True
            break
    persist(nodes)


# ---------------- Shape propagation helpers ----------------


def _copy_shape_only(dst: Optional["ir.Value"], src: Optional["ir.Value"]) -> bool:
    """Copy shape metadata from src → dst when dst is missing/unknown."""
    if dst is None or src is None:
        return False
    try:
        s_shp = src.shape
    except AttributeError:
        return False
    if s_shp is None:
        return False
    try:
        d_shp = dst.shape
    except AttributeError:
        d_shp = None
    s_key = _shape_dims_key(s_shp)
    d_key = _shape_dims_key(d_shp) if d_shp is not None else None
    if s_key is None:
        return False
    if d_key == s_key:
        return False
    cloned = _clone_shape_obj(s_shp)
    if cloned is None:
        return False
    try:
        dst.shape = cloned
        return True
    except Exception:
        return False


def _copy_shape_dtype(dst: Optional["ir.Value"], src: Optional["ir.Value"]) -> bool:
    """
    Copy shape & dtype from src -> dst if present; return True if anything changed.
    """
    if dst is None or src is None:
        return False
    changed = False
    try:
        s_shp = src.shape
    except AttributeError:
        s_shp = None
    try:
        d_shp = dst.shape
    except AttributeError:
        d_shp = None
    if s_shp is not None:
        s_key = _shape_dims_key(s_shp)
        d_key = _shape_dims_key(d_shp) if d_shp is not None else None
        if s_key is not None and s_key != d_key:
            cloned = _clone_shape_obj(s_shp)
            if cloned is not None:
                try:
                    dst.shape = cloned
                    changed = True
                except Exception:
                    pass
    try:
        s_ty = src.type
    except AttributeError:
        s_ty = None
    try:
        d_ty = dst.type
    except AttributeError:
        d_ty = None
    if s_ty is not None and s_ty is not d_ty:
        try:
            dst.type = s_ty
            changed = True
        except Exception:
            pass
    return changed


def propagate_unary_shapes_ir(graph) -> None:
    """
    For known unary dataflow ops, set the first output's shape & dtype = first input's,
    when output metadata is missing/unknown. This helps preserve batch symbols across
    elementwise ops (e.g., BxN through Dropout/Gelu/etc.).
    """
    nodes, persist = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = False
    for n in nodes:
        op = n.op_type
        if op not in UNARY_DATAFLOW_OPS:
            continue
        ins = _node_inputs(n)
        outs = _node_outputs(n)
        if not ins or not outs:
            continue
        if op in {"Cast", "CastLike"}:
            if _copy_shape_only(outs[0], ins[0]):
                changed = True
            continue
        if _copy_shape_dtype(outs[0], ins[0]):
            changed = True
    if changed:
        persist(nodes)


# ---------------- Dropout.training_mode constant inlining ----------------


def _find_producer_idx(
    nodes: List["ir.Node"], val_or_name: Optional[object]
) -> Optional[int]:
    """
    Return the index of the node that produces the given tensor.
    Accepts either a Value object or a tensor name (str).
    Tries object identity first, then falls back to name-based matching.
    """
    if val_or_name is None:
        return None
    # Prefer IR API when available
    if isinstance(val_or_name, ir.Value):
        prod = val_or_name.producer()
        if prod is not None:
            for idx, n in enumerate(nodes):
                if n is prod:
                    return idx
    # Object identity match
    for idx, n in enumerate(nodes):
        for ov in _node_outputs(n):
            if ov is val_or_name:
                return idx
    # Name-based match
    name: Optional[str]
    if isinstance(val_or_name, str):
        name = val_or_name
    else:
        name = _v_name(val_or_name)  # type: ignore[arg-type]
    if not name:
        return None
    for idx, n in enumerate(nodes):
        for ov in _node_outputs(n):
            if _v_name(ov) == name:
                return idx
    return None


def _literal_bool_array(value: str) -> Optional[np.ndarray]:
    normalized = value.strip().lower()
    if normalized == "true":
        return np.asarray(True, dtype=np.bool_)
    if normalized == "false":
        return np.asarray(False, dtype=np.bool_)
    return None


def _to_numpy_from_attr(attr: ir.Attr) -> Optional[np.ndarray]:
    attr_type = attr.type
    if isinstance(attr_type, str):
        try:
            attr_type = IRAttrType[attr_type.upper()]
        except KeyError:
            attr_type = None
    if attr_type is IRAttrType.FLOAT:
        try:
            return np.asarray(attr.as_float())
        except Exception:
            return None
    if attr_type is IRAttrType.FLOATS:
        try:
            return np.asarray(tuple(attr.as_floats()))
        except Exception:
            return None
    if attr_type is IRAttrType.INT:
        try:
            return np.asarray(attr.as_int())
        except Exception:
            return None
    if attr_type is IRAttrType.INTS:
        try:
            return np.asarray(tuple(attr.as_ints()))
        except Exception:
            return None
    if attr_type is IRAttrType.STRING:
        try:
            string_value = attr.as_string()
        except Exception:
            return None
        bool_arr = _literal_bool_array(string_value)
        if bool_arr is not None:
            return bool_arr
        return np.asarray(string_value)
    if attr_type is IRAttrType.STRINGS:
        try:
            strings = tuple(attr.as_strings())
        except Exception:
            return None
        if len(strings) == 1:
            bool_arr = _literal_bool_array(strings[0])
            if bool_arr is not None:
                return bool_arr
        return np.asarray(strings)
    if attr_type is IRAttrType.TENSOR:
        try:
            return _to_numpy_from_any(attr.as_tensor())
        except Exception:
            return None
    if attr_type is IRAttrType.TENSORS:
        try:
            tensors = tuple(attr.as_tensors())
        except Exception:
            return None
        if len(tensors) == 1:
            return _to_numpy_from_any(tensors[0])
        collected = [
            _to_numpy_from_any(tensor) for tensor in tensors if tensor is not None
        ]
        if not collected or any(arr is None for arr in collected):
            return None
        try:
            return np.stack([np.asarray(arr) for arr in collected])
        except Exception:
            return None
    value = attr.value
    if value is not None:
        return _to_numpy_from_any(value)
    return None


def _to_numpy_from_any(x: object) -> Optional[np.ndarray]:
    """
    Convert common IR payload carriers (Values, Tensors, Attrs, numpy scalars) into
    numpy arrays so scalar booleans can be read without falling back to proto shims.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, np.generic):
        return np.asarray(x)
    if isinstance(x, (bool, int, float, complex, np.bool_, np.integer, np.floating)):
        return np.asarray(x)
    if isinstance(x, str):
        bool_arr = _literal_bool_array(x)
        if bool_arr is not None:
            return bool_arr
        return np.asarray(x)
    if isinstance(x, ir.Value):
        return _to_numpy_from_any(x.const_value)
    if isinstance(x, ir.Tensor):
        try:
            return np.asarray(x.numpy())
        except Exception:
            return None
    if isinstance(x, ir.Attr):
        return _to_numpy_from_attr(x)
    if isinstance(x, SequenceABC) and not isinstance(x, (bytes, bytearray)):
        try:
            return np.asarray(tuple(x))
        except Exception:
            return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.dtype == object and arr.size == 1:
        try:
            return _to_numpy_from_any(arr.reshape(()).item())
        except Exception:
            return None
    return arr


def _as_scalar_bool(payload: object) -> Optional[bool]:
    if isinstance(payload, (bool, np.bool_)):
        return bool(payload)
    arr = _to_numpy_from_any(payload)
    if arr is None:
        return None
    try:
        return bool(np.asarray(arr).reshape(()).astype(np.bool_).item())
    except Exception:
        return None


def _read_scalar_bool_from_value_or_constant(
    nodes: List["ir.Node"], v_or_name: Optional[object]
) -> Optional[bool]:
    """Resolve a scalar boolean carried by a value or Constant producer."""
    if v_or_name is None:
        return None

    if isinstance(v_or_name, ir.Value):
        val = _as_scalar_bool(v_or_name.const_value)
        if val is not None:
            _dbg_tm("read Value-const:", type(v_or_name).__name__, "→", val)
            return val

    producer_idx = _find_producer_idx(nodes, v_or_name)
    if producer_idx is None:
        return None
    node = nodes[producer_idx]
    if node is None or node.op_type != "Constant":
        return None

    attr = _get_attr(node, "value")
    if isinstance(attr, ir.Attr):
        if attr.type is IRAttrType.TENSOR:
            tensor = attr.as_tensor()
            val = _as_scalar_bool(tensor)
            if val is not None:
                _dbg_tm("read Const-attr tensor →", val)
                return val
        val = _as_scalar_bool(attr.value)
        if val is not None:
            _dbg_tm("read Const-attr value →", val)
            return val
    elif attr is not None:
        val = _as_scalar_bool(attr)
        if val is not None:
            _dbg_tm("read Const-attr payload →", val)
            return val

    for output in _node_outputs(node):
        if output is None:
            continue
        val = _as_scalar_bool(output.const_value)
        if val is not None:
            _dbg_tm("read Const output →", val)
            return val
    return None


def _missing_bool_value() -> "ir.Value":
    return ir.Value(name="", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(()))


def inline_dropout_training_mode_constants_ir(graph) -> None:
    """
    Constant-only inlining for Dropout.training_mode:
      - If training_mode is a constant False → drop it (make input #2 missing)
      - If training_mode is Not(True)        → drop it (make input #2 missing)
    This preserves dynamic (graph-input) cases: they are NOT inlined.
    """
    nodes, persist = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = False
    del_not_names: Set[str] = set()
    del_not_idx: Set[int] = set()
    for idx, n in enumerate(nodes):
        if n.op_type != "Dropout":
            continue
        ins = _node_inputs(n)
        if len(ins) < 3:
            continue
        tm = ins[2]
        _dbg_tm(
            "Dropout@",
            idx,
            "tm input:",
            (tm if isinstance(tm, str) else _v_name(tm)),
            "type:",
            type(tm).__name__,
        )
        # Prefer handling the Not(True) pattern so we can drop the Not producer.
        pidx = _find_producer_idx(nodes, tm)
        if pidx is not None:
            producer = nodes[pidx]
            if producer is not None and producer.op_type == "Not":
                _dbg_tm("tm producer is Not @", pidx)
                not_in = (_node_inputs(producer) or [None])[0]
                if isinstance(not_in, ir.Value) and not_in.is_graph_input():
                    _dbg_tm("Not input is dynamic graph input; skipping")
                    _dbg_tm("Not input could not be proven True; nv=", None)
                    continue
                nv = _read_scalar_bool_from_value_or_constant(nodes, not_in)
                if nv is not None and bool(nv) is True:
                    miss = _missing_bool_value()
                    ins_new = list(ins)
                    ins_new[2] = miss
                    old_not_out = _node_output(producer)
                    _replace_everywhere(nodes, old_not_out, _v_name(old_not_out), miss)
                    _replace_in_graph_outputs(
                        graph, old_not_out, _v_name(old_not_out), miss
                    )
                    _set_node_inputs(n, ins_new)
                    nodes[idx] = n
                    changed = True
                    out_v = _node_output(producer)
                    out_name = _v_name(out_v)
                    if out_name:
                        del_not_names.add(out_name)
                    del_not_idx.add(pidx)
                    continue
                _dbg_tm("Not input could not be proven True; nv=", nv)

        # Case A removed: we only inline Not(True) patterns to preserve
        # call-param wiring in inference graphs.
    if changed:
        _dbg_tm("changed detected; del_not_idx=", sorted(del_not_idx))
        # Explicitly delete orphan Not producers we identified
        # Remove any Not nodes whose outputs have no remaining consumers
        if del_not_names:
            still_needed: Set[str] = set()
            for i, m in enumerate(nodes):
                if i in del_not_idx:
                    continue
                for iv in _node_inputs(m):
                    nm = _v_name(iv)
                    if nm:
                        still_needed.add(nm)
            del_not_names = {nm for nm in del_not_names if nm not in still_needed}

        new_nodes: List[ir.Node] = []
        for i, m in enumerate(nodes):
            if i in del_not_idx:
                continue
            out_names = {_v_name(ov) for ov in _node_outputs(m)}
            if del_not_names and (out_names & del_not_names):
                if TRN_DEBUG or os.getenv("JAX2ONNX_TM_DEBUG"):
                    print("[tm-inline] removed orphan Not")
                continue
            new_nodes.append(m)
        nodes = new_nodes
        persist(nodes)


# ---------------- Graph IO (for DCE/prune) ----------------


def _graph_inputs_list(graph: ir.Graph) -> List["ir.Value"]:
    return list(graph.inputs)


def _graph_outputs_list(graph: ir.Graph) -> List["ir.Value"]:
    return list(graph.outputs)


# ---------------- DCE ----------------


def remove_dead_nodes_ir(graph) -> None:
    nodes, persist = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    prod_obj, prod_name, _cbo, _cbn = _build_use_maps(nodes)
    worklist: List["ir.Value"] = [
        v for v in _graph_outputs_list(graph) if v is not None
    ]
    used_names: Set[str] = set()
    _collect_used_value_names(graph, used_names)
    if used_names:
        seen_names: Set[str] = set()
        for name in used_names:
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            idx = prod_name.get(name)
            if idx is None:
                continue
            for ov in _node_outputs(nodes[idx]):
                if _v_name(ov) == name:
                    worklist.append(ov)
    live_nodes: Set[int] = set()
    while worklist:
        v = worklist.pop()
        idx = prod_obj.get(id(v))
        if idx is None:
            nm = _v_name(v)
            if nm:
                idx = prod_name.get(nm)
        if idx is None or idx in live_nodes:
            continue
        live_nodes.add(idx)
        for iv in _node_inputs(nodes[idx]):
            if iv is not None:
                worklist.append(iv)
    if len(live_nodes) == len(nodes):
        return
    new_nodes = [n for i, n in enumerate(nodes) if i in live_nodes]
    if DCE_DEBUG:
        dropped = [n.op_type for i, n in enumerate(nodes) if i not in live_nodes]
        print("[dce] removed", len(nodes) - len(new_nodes), "nodes:", dropped)
    persist(new_nodes)


# ---------------- Prune unused graph inputs (top graph only) ----------------


def _attr_kind(attr: object) -> Optional[str]:
    if attr is None:
        return None
    if isinstance(attr, ir.Attr):
        atype = attr.type
        if isinstance(atype, IRAttrType):
            return atype.name
        if isinstance(atype, str):
            return atype.upper()
        return None
    if isinstance(attr, IRAttrType):
        return attr.name
    if isinstance(attr, str):
        return attr.upper()
    return None


def _iter_node_attrs(node: ir.Node) -> Iterable[ir.Attr]:
    attrs = node.attributes
    if isinstance(attrs, Mapping):
        return [attr for attr in attrs.values() if isinstance(attr, ir.Attr)]
    if isinstance(attrs, SequenceABC):
        return [attr for attr in attrs if isinstance(attr, ir.Attr)]
    return []


def _collect_used_value_names(graph, used: Set[str]) -> None:
    """Record names that are consumed from an *outer* scope.

    A name is considered "used" for the parent when it appears as an input to
    a node but is not defined within the current graph (i.e. not produced by a
    node, declared as a graph input, or introduced as an initializer). This
    mirrors ONNX's lexical scoping rules for control-flow/function bodies.
    """

    nodes, _ = _get_node_seq_and_setter(graph)
    if not nodes:
        nodes = []

    local_defs: Set[str] = set()
    for g_in in _graph_inputs_list(graph):
        nm = _v_name(g_in)
        if nm:
            local_defs.add(nm)

    for node in nodes:
        for ov in _node_outputs(node):
            nm = _v_name(ov)
            if nm:
                local_defs.add(nm)

    for node in nodes:
        for iv in _node_inputs(node):
            nm = _v_name(iv)
            if nm and nm not in local_defs:
                used.add(nm)

        for attr in _iter_node_attrs(node):
            kind = _attr_kind(attr)
            if kind == "GRAPH":
                if not isinstance(attr, ir.Attr):
                    continue
                try:
                    sub_graph = attr.as_graph()
                except Exception:
                    continue
                if sub_graph is not None:
                    _collect_used_value_names(sub_graph, used)
            elif kind == "GRAPHS":
                if not isinstance(attr, ir.Attr):
                    continue
                try:
                    sub_graphs = tuple(attr.as_graphs())
                except Exception:
                    sub_graphs = ()
                for sub in sub_graphs:
                    _collect_used_value_names(sub, used)


def prune_unused_graph_inputs_ir(graph) -> None:
    """
    Remove graph inputs that are not consumed by any node and are not graph outputs.
    (We do NOT run this on function bodies to avoid changing function signatures.)
    """
    nodes, _ = _get_node_seq_and_setter(graph)
    used: Set[str] = set()
    _collect_used_value_names(graph, used)
    for ov in _graph_outputs_list(graph):
        nm = _v_name(ov)
        if nm:
            used.add(nm)

    output_names = {nm for nm in (_v_name(v) for v in _graph_outputs_list(graph)) if nm}

    def _should_always_keep(name: Optional[str]) -> bool:
        if not name:
            return False
        # Preserve positional graph inputs that correspond to original JAX
        # function arguments (named ``in_<index>`` by IRContext.add_input_for_invar).
        if name.startswith("in_"):
            suffix = name[3:]
            if suffix.isdigit():
                return True
        return False

    inputs_container = graph.inputs
    original_inputs = list(inputs_container)
    keep: List[ir.Value] = []
    removed: List[str] = []
    for value in original_inputs:
        name = _v_name(value)
        if not name:
            keep.append(value)
            continue

        should_keep = False
        if _should_always_keep(name):
            should_keep = True
        elif name in output_names:
            should_keep = True
        elif _count_consumers(nodes or [], name, value) > 0:
            should_keep = True
        elif name in used:
            should_keep = True

        if should_keep:
            keep.append(value)
        else:
            removed.append(name)

    if removed and DEBUG:
        _dbg(f"prune_unused_graph_inputs_ir removed: {removed}")

    if keep != original_inputs:
        inputs_container[:] = keep


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def optimize_graph(ir_model: ir.Model) -> ir.Model:
    # Top graph
    try:
        gr = ir_model.graph
        remove_redundant_casts_ir(gr)
        remove_redundant_transpose_pairs_ir(gr)
        remove_redundant_reshape_pairs_ir(gr)
        inline_dropout_training_mode_constants_ir(gr)
        propagate_unary_shapes_ir(gr)
        remove_redundant_casts_ir(gr)
        remove_dead_nodes_ir(gr)
        prune_unused_graph_inputs_ir(gr)
    except Exception as _e:
        _dbg("optimize_graph: top-graph pass skipped:", _e)

    # Function bodies – do NOT prune function inputs (signature!)
    try:
        funcs_container = ir_model.functions
        if isinstance(funcs_container, dict):
            values: Iterable[Any] = funcs_container.values()
        elif funcs_container is None:
            values = ()
        else:
            values = funcs_container
        for fn in values:
            fgr = fn.graph
            try:
                remove_redundant_casts_ir(fgr)
                remove_redundant_transpose_pairs_ir(fgr)
                remove_redundant_reshape_pairs_ir(fgr)
                inline_dropout_training_mode_constants_ir(fgr)
                propagate_unary_shapes_ir(fgr)
                remove_redundant_casts_ir(fgr)
                remove_dead_nodes_ir(fgr)
            except Exception as _fe:
                _dbg("optimize_graph: function pass skipped:", _fe)
    except Exception as _e:
        _dbg("optimize_graph: functions traversal skipped:", _e)

    return ir_model
