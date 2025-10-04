# jax2onnx/converter/ir_optimizations.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Iterable, Any
import copy
import os
import numpy as np

import onnx_ir as ir
from onnx_ir import AttributeType as IRAttrType

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

ALLOWED_ELEMWISE = {
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

DEBUG = bool(int(os.getenv("JAX2ONNX_IROPT_DEBUG", "0")))
RSH_DEBUG = bool(int(os.getenv("JAX2ONNX_RSH_DEBUG", "0")))
TRN_DEBUG = bool(int(os.getenv("JAX2ONNX_TRN_DEBUG", "0")))
DCE_DEBUG = bool(int(os.getenv("JAX2ONNX_DCE_DEBUG", "0")))
TM_DEBUG = bool(int(os.getenv("JAX2ONNX_TM_DEBUG", "0")))

# ---------------- Debug ----------------


def _dbg(*a):
    if DEBUG:
        print("[iropt]", *a)


def _dbg_tm(*a):
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


def _get_perm_attr(node) -> Optional[List[int]]:
    """
    Return the Transpose 'perm' attribute as a list of ints, or None.
    Works with nodes exposing .attributes (list of Attr with .name/.ints)
    or generic IR nodes where _get_attr('perm') is available.
    """
    a = _get_attr(node, "perm")
    if a is None:
        # try attributes list directly (python objects used by tests)
        attrs = getattr(node, "attributes", None)
        if isinstance(attrs, list):
            for att in attrs:
                if getattr(att, "name", None) == "perm":
                    if hasattr(att, "ints"):
                        return list(getattr(att, "ints"))
                    try:
                        seq = list(att)
                        if all(isinstance(x, int) for x in seq):
                            return seq
                    except Exception:
                        pass
        return None
    # Attr object with .ints
    ints = getattr(a, "ints", None)
    if ints is not None:
        return list(ints)
    # Maybe list-like
    try:
        seq = list(a)
        if all(isinstance(x, int) for x in seq):
            return seq
    except Exception:
        pass
    return None


def _perms_compose_identity(p1: List[int], p2: List[int]) -> bool:
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


def _has_input_name_or_obj(node, name: Optional[str], obj) -> bool:
    """
    Return True if 'node' has an input that matches either the given name
    (by .name on Value or string equality) or the given object identity.
    """
    ins = _node_inputs(node)
    for iv in ins:
        if obj is not None and iv is obj:
            return True
        if name:
            ivn = _v_name(iv)
            if ivn == name:
                return True
            # If inputs are plain strings in this build
            if isinstance(iv, str) and iv == name:
                return True
    return False


def _count_consumers(nodes: List[object], name: Optional[str], obj) -> int:
    """
    Count how many nodes consume the given value (by name or object).
    """
    c = 0
    for n in nodes:
        if _has_input_name_or_obj(n, name, obj):
            c += 1
    return c


def _find_next_consumer_idx(
    nodes: List[object], start_idx: int, name: Optional[str], obj
) -> Optional[int]:
    """
    Find the index of the next node (after start_idx) that consumes the given
    value (by name or object). Return None if not found.
    """
    for i in range(start_idx + 1, len(nodes)):
        if _has_input_name_or_obj(nodes[i], name, obj):
            return i
    return None


# ---------------- IR helpers ----------------


def _v_name(v: ir.Value | None) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    nm = getattr(v, "name", None)
    return nm if isinstance(nm, str) and nm != "" else None


def _value_dtype_code(val: Optional[ir.Value]) -> Optional[int]:
    if val is None or isinstance(val, str):
        return None
    # Value may expose dtype directly or via .type
    dt = getattr(val, "dtype", None)
    if isinstance(dt, ir.DataType):
        return int(dt.value)
    if isinstance(dt, (int, np.integer)):
        return int(dt)
    ty = getattr(val, "type", None)
    if ty is not None:
        tdt = getattr(ty, "dtype", None)
        if isinstance(tdt, ir.DataType):
            return int(tdt.value)
        if isinstance(tdt, (int, np.integer)):
            return int(tdt)
        elem = getattr(ty, "elem_type", None)
        if isinstance(elem, ir.DataType):
            return int(elem.value)
        if isinstance(elem, (int, np.integer)):
            return int(elem)
    return None


def _node_outputs(n) -> List["ir.Value"]:
    outs = getattr(n, "outputs", None)
    if outs is None:
        outs = getattr(n, "output", [])  # defensive
    return list(outs)


def _node_output(n) -> Optional["ir.Value"]:
    outs = _node_outputs(n)
    return outs[0] if outs else None


def _node_inputs(n) -> List["ir.Value"]:
    ins = getattr(n, "inputs", None)
    if ins is None:
        ins = getattr(n, "input", [])  # defensive
    return list(ins)


def _set_node_inputs(n, new_ins: List["ir.Value"]) -> None:
    """
    Robustly write a node's inputs across onnx_ir variants that use either
    `.inputs` (Value objects) or `.input` (Value objects or names).
    """
    # 1) Try the canonical `inputs` attribute first
    try:
        n.inputs = tuple(new_ins)
        wrote_inputs = True
    except Exception:
        wrote_inputs = False
        try:
            n.inputs = list(new_ins)
            wrote_inputs = True
        except Exception:
            pass
    # 2) Then try `input`. If the current field stores strings, convert.
    cur = getattr(n, "input", None)
    if cur is not None:
        try:
            if isinstance(cur, (list, tuple)) and (
                len(cur) == 0 or isinstance(cur[0], str)
            ):
                names = []
                for v in new_ins:
                    nm = _v_name(v)
                    names.append(nm if nm is not None else "")
                try:
                    n.input = tuple(names)
                except Exception:
                    n.input = list(names)
            else:
                # assume Value-like containers
                try:
                    n.input = tuple(new_ins)
                except Exception:
                    n.input = list(new_ins)
        except Exception:
            # best effort; if `inputs` was written above that's already enough
            if not wrote_inputs:
                raise


def _rebuild_node_with_inputs(n: "ir.Node", new_ins: List["ir.Value"]) -> "ir.Node":
    """
    Construct a fresh ir.Node with the same op_type/name/domain/outputs/attributes,
    but with the provided inputs. This avoids copy-on-write pitfalls in some
    onnx_ir builds where mutating `.inputs`/`.input` doesn't persist.
    """
    op_type = getattr(n, "op_type", "")
    domain = getattr(n, "domain", "")
    name = getattr(n, "name", "")
    outs = _node_outputs(n)
    # Try to carry attributes through (handle common spellings)
    attrs = getattr(n, "attributes", None)
    if attrs is None:
        attrs = getattr(n, "attribute", None)
    if attrs is None:
        attrs = []
    try:
        new_node = ir.Node(
            op_type=op_type,
            domain=domain,
            inputs=list(new_ins),
            outputs=list(outs),
            name=name,
            attributes=list(attrs) if isinstance(attrs, (list, tuple)) else attrs,
            num_outputs=len(outs) if hasattr(n, "num_outputs") else None,
        )
    except Exception:
        # Fall back to a simpler ctor without attributes/num_outputs if needed
        new_node = ir.Node(
            op_type=op_type,
            domain=domain,
            inputs=list(new_ins),
            outputs=list(outs),
            name=name,
        )
    return new_node


def _shape_tuple(v: Optional["ir.Value"]) -> Optional[Tuple]:
    if v is None:
        return None
    shp = getattr(v, "shape", None)
    if shp is None:
        return None
    out = []
    for d in shp:
        out.append(d if isinstance(d, int) else -1)
    return tuple(out)


def _shape_dims_key(shape) -> Optional[Tuple[str, ...]]:
    """Return a hashable key representing the shape's dimensions."""
    if shape is None:
        return None
    dims = getattr(shape, "dims", None)
    if dims is None:
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


def _clone_shape_obj(shape):
    """Best-effort clone of an onnx_ir Shape object."""
    if shape is None:
        return None
    try:
        return copy.deepcopy(shape)
    except Exception:
        pass

    dims = getattr(shape, "dims", None)
    if dims is None:
        try:
            dims = tuple(shape)
        except Exception:
            return None
    norm_dims: List[Any] = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            norm_dims.append(int(d))
        else:
            try:
                norm_dims.append(int(d))
            except Exception:
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
    graph,
) -> Tuple[List["ir.Node"], Optional[Tuple[object, str]]]:
    """
    Return (nodes_container, (parent, attr)) such that:
      • If the underlying container is a mutable list, we return that **live** list
        so in-place mutations persist immediately.
      • Otherwise we return a Python list copy and the caller should call _set_nodes
        to persist changes.
    """
    for attr in ("nodes", "_nodes", "node"):
        if hasattr(graph, attr):
            cont = getattr(graph, attr)
            # If it's already a mutable Python list, use it directly (no copy).
            if isinstance(cont, list):
                return cont, (graph, attr)
            # Else, fall back to a list copy with write-back responsibility.
            try:
                return list(cont), (graph, attr)
            except Exception:
                # As a last resort, expose an empty list with a setter; caller will set later.
                return [], (graph, attr)
    return [], None


def _set_nodes(store: Optional[Tuple[object, str]], nodes: List["ir.Node"]) -> None:
    if store is None:
        return
    parent, _ = store
    # Persist to ALL known containers to avoid copy-on-write / mirror mismatches.
    for attr in ("nodes", "_nodes", "node"):
        if hasattr(parent, attr):
            try:
                setattr(parent, attr, nodes)
            except Exception:
                # Try in-place replace if container is a mutable sequence
                try:
                    seq = getattr(parent, attr)
                    if isinstance(seq, list):
                        seq.clear()
                        seq.extend(nodes)
                except Exception:
                    pass


def _replace_everywhere(
    nodes: List["ir.Node"],
    old_v: Optional["ir.Value"],
    old_name: Optional[str],
    new_v: "ir.Value",
) -> None:
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
            if hasattr(m, "replace_input_with"):
                try:
                    for idx, iv in enumerate(ins):
                        if iv is new_v:
                            m.replace_input_with(idx, new_v)
                except Exception:
                    pass
            try:
                _set_node_inputs(m, ins)
            except Exception:
                # Fallback for pure string inputs
                names = []
                for iv in ins:
                    if isinstance(iv, str):
                        names.append(iv)
                    else:
                        nm = _v_name(iv)
                        names.append(nm if nm is not None else "")
                try:
                    m.input = tuple(names)
                except Exception:
                    m.input = list(names)


def _replace_in_graph_outputs(
    graph, old_v: Optional["ir.Value"], old_name: Optional[str], new_v: "ir.Value"
) -> None:
    for attr in ("outputs", "output"):
        outs = getattr(graph, attr, None)
        if outs is None:
            continue
        try:
            lst = list(outs)
        except Exception:
            continue
        changed = False
        for i, ov in enumerate(lst):
            if (old_v is not None and ov is old_v) or (
                old_name and _v_name(ov) == old_name
            ):
                lst[i] = new_v
                changed = True
        if changed:
            try:
                setattr(graph, attr, lst)
            except Exception:
                # last resort: rename
                for i, ov in enumerate(getattr(graph, attr, [])):
                    if (old_v is not None and ov is old_v) or (
                        old_name and _v_name(ov) == old_name
                    ):
                        try:
                            ov.name = _v_name(new_v)
                        except Exception:
                            pass


def _build_use_maps(nodes: List["ir.Node"]):
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
    cons_by_obj, cons_by_name, val: Optional["ir.Value"]
) -> Optional[int]:
    if val is None:
        return None
    nm = _v_name(val)
    S = set(cons_by_obj.get(id(val), set()))
    if nm:
        S |= set(cons_by_name.get(nm, set()))
    return next(iter(S)) if len(S) == 1 else None


def _producer_idx_for(val: Optional["ir.Value"], prod_obj, prod_name) -> Optional[int]:
    if val is None:
        return None
    nm = _v_name(val)
    return prod_obj.get(id(val)) or (prod_name.get(nm) if nm else None)


def _all_consumers(cons_by_obj, cons_by_name, v: Optional["ir.Value"]) -> Set[int]:
    if v is None:
        return set()
    nm = _v_name(v)
    S = set(cons_by_obj.get(id(v), set()))
    if nm:
        S |= set(cons_by_name.get(nm, set()))
    return S


# ---------------- Attr access ----------------


def _get_attr(node, name: str):
    """
    Robustly fetch a node attribute 'name' across onnx_ir variants:
      • dict-like or mapping containers (Attributes)
      • sequences of Attrs
      • direct attribute on the node
    """
    for attr_name in ("attributes", "attrs", "_attrs"):
        d = getattr(node, attr_name, None)
        if d is None:
            continue
        # Mapping-like with .get
        if hasattr(d, "get"):
            try:
                a = d.get(name)
                if a is not None:
                    return a
            except Exception:
                pass
        # Mapping-like via __getitem__
        try:
            a = d[name]  # type: ignore[index]
            return a
        except Exception:
            pass
        # Iterable of Attrs or (key, value) pairs
        try:
            it = d.values() if hasattr(d, "values") else d
            for item in it:
                an = getattr(item, "name", getattr(item, "key", None))
                if an == name:
                    return item
        except Exception:
            pass
    # Fallback: explicit sequence 'attribute'
    seq = getattr(node, "attribute", None)
    if isinstance(seq, (list, tuple)):
        for a in seq:
            if getattr(a, "name", None) == name:
                return a
    # Last resort: direct attribute
    if hasattr(node, name):
        return getattr(node, name)
    return None


def _attr_to_int(attr: Any) -> Optional[int]:
    if attr is None:
        return None
    if isinstance(attr, (int, np.integer)):
        return int(attr)
    for field in ("i", "value", "int_value", "int", "int32", "int64"):
        if hasattr(attr, field):
            val = getattr(attr, field)
            if isinstance(val, (int, np.integer)):
                return int(val)
    # Some attrs expose .ints with single element
    for field in ("ints", "values"):
        seq = getattr(attr, field, None)
        if isinstance(seq, (list, tuple)) and len(seq) == 1:
            val = seq[0]
            if isinstance(val, (int, np.integer)):
                return int(val)
    try:
        seq = list(attr)
        if len(seq) == 2 and seq[0] == "to" and isinstance(seq[1], (int, np.integer)):
            return int(seq[1])
        if len(seq) == 1 and isinstance(seq[0], (int, np.integer)):
            return int(seq[0])
    except Exception:
        pass
    return None


def _collect_value_dtypes(graph, nodes: List["ir.Node"]) -> Dict[str, int]:
    type_map: Dict[str, int] = {}

    def _record(val):
        name = _v_name(val)
        code = _value_dtype_code(val)
        if name and code is not None:
            type_map.setdefault(name, code)

    for attr in ("inputs", "input"):
        vals = getattr(graph, attr, None)
        if vals is None:
            continue
        try:
            for v in vals:
                _record(v)
        except Exception:
            pass

    for attr in ("outputs", "output"):
        vals = getattr(graph, attr, None)
        if vals is None:
            continue
        try:
            for v in vals:
                _record(v)
        except Exception:
            pass

    inits = getattr(graph, "initializer", None)
    if inits is not None:
        try:
            for init in inits:
                name = getattr(init, "name", None)
                dtype = getattr(init, "data_type", None)
                if isinstance(name, str) and isinstance(dtype, (int, np.integer)):
                    type_map.setdefault(name, int(dtype))
        except Exception:
            pass

    for node in nodes:
        for ov in _node_outputs(node):
            _record(ov)
        # also inspect inputs, in case they carry dtype metadata
        for iv in _node_inputs(node):
            _record(iv)

    return type_map


# ---------------- Cast cleanup ----------------


def remove_redundant_casts_ir(graph) -> None:
    nodes, store = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        dtype_map = _collect_value_dtypes(graph, nodes)
        prod_obj, prod_name, cons_by_obj, cons_by_name = _build_use_maps(nodes)
        for idx, n in enumerate(nodes):
            if getattr(n, "op_type", None) != "Cast":
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
                    if getattr(next_node, "op_type", None) == "Cast":
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
                                _set_nodes(store, nodes)
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
            _set_nodes(store, nodes)
            changed = True
            break
    _set_nodes(store, nodes)


# ---------------- Transpose folding ----------------


def _transpose_perm(node) -> Optional[List[int]]:
    a = _get_attr(node, "perm")
    if a is None:
        return None
    ints = getattr(a, "ints", None)
    if ints is not None:
        return [int(v) for v in ints]
    if hasattr(a, "as_ints"):
        try:
            ints = a.as_ints()
            if ints is not None:
                return [int(v) for v in ints]
        except Exception:
            pass
    value = getattr(a, "value", None)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    try:
        seq = list(a)
        if all(isinstance(x, int) for x in seq):
            return seq
    except Exception:
        pass
    if isinstance(a, dict) and "ints" in a and isinstance(a["ints"], (list, tuple)):
        return [int(v) for v in a["ints"]]
    return None


def remove_redundant_transpose_pairs_ir(graph) -> None:
    nodes, store = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        _prod_obj, _prod_name, cons_by_obj, cons_by_name = _build_use_maps(nodes)
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if getattr(n, "op_type", None) != "Transpose":
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
            _set_nodes(store, nodes)
            changed = True
            break
    _set_nodes(store, nodes)


# ---------------- Reshape folding ----------------


def remove_redundant_reshape_pairs_ir(graph) -> None:
    nodes, store = _get_node_seq_and_setter(graph)
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
            if getattr(T2, "op_type", None) != "Reshape":
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
            _set_nodes(store, nodes)
            changed = True
            break
    _set_nodes(store, nodes)


# ---------------- Shape propagation helpers ----------------


def _copy_shape_only(dst: Optional["ir.Value"], src: Optional["ir.Value"]) -> bool:
    """Copy shape metadata from src → dst when dst is missing/unknown."""
    if dst is None or src is None:
        return False
    s_shp = getattr(src, "shape", None)
    if s_shp is None:
        return False
    d_shp = getattr(dst, "shape", None)
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
    s_shp = getattr(src, "shape", None)
    d_shp = getattr(dst, "shape", None)
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
    s_ty = getattr(src, "type", None)
    d_ty = getattr(dst, "type", None)
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
    nodes, store = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = False
    for n in nodes:
        op = getattr(n, "op_type", "")
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
        _set_nodes(store, nodes)


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


def _to_numpy_from_any(x: object) -> Optional[np.ndarray]:
    """
    Best-effort conversion of various IR/ONNX payloads (including onnx_ir wrappers)
    to a NumPy array:
      - numpy-like / scalar                → np.asarray(x)
      - has .numpy or .to_numpy()         → np.asarray(that)
      - has common payload attrs          → np.asarray(getattr(x, ...))
      - TensorProto-like (raw_data + dims)→ decoded buffer
      - typed *_data fields               → np.asarray(field, dtype)
      - last resort                       → np.asarray(x) (may still fail)
    """
    # Direct adapters that expose numpy-like payloads (call before np.asarray)
    for accessor in ("numpy", "to_numpy"):
        try:
            attr = getattr(x, accessor, None)
            if attr is None:
                continue
            data = attr() if callable(attr) else attr
            arr = np.asarray(data)
            if isinstance(arr, np.ndarray) or np.isscalar(arr):
                return arr
        except Exception:
            pass

    # Fast path: already array/scalar-ish
    try:
        arr0 = np.asarray(x)
        if isinstance(arr0, np.ndarray) or np.isscalar(arr0):
            # If it's an object 0D array, attempt unwrap
            if isinstance(arr0, np.ndarray) and arr0.dtype == object and arr0.size == 1:
                try:
                    inner = arr0.reshape(()).item()
                    inner_arr = np.asarray(inner)
                    if isinstance(inner_arr, np.ndarray) or np.isscalar(inner_arr):
                        return inner_arr
                except Exception:
                    pass
            return arr0
    except Exception:
        pass

    # onnx_ir tensor wrapper heuristics: probe typical payload attrs
    for payload_attr in ("value", "data", "array", "_array", "val"):
        try:
            payload = getattr(x, payload_attr, None)
            if payload is not None:
                return np.asarray(payload)
        except Exception:
            pass

    # TensorProto-like decoding (raw_data / dims / data_type)
    try:
        raw = getattr(x, "raw_data", None)
        if raw is not None:
            dt = int(getattr(x, "data_type", 0))
            # minimal map: FLOAT=1, BOOL=9, DOUBLE=11
            dt_map = {1: np.float32, 9: np.bool_, 11: np.float64}
            dtype = dt_map.get(dt, np.uint8)
            arr = np.frombuffer(raw, dtype=dtype)
            dims = tuple(getattr(x, "dims", ()))
            return arr.reshape(dims) if dims else (arr[0] if arr.size == 1 else arr)
        # fallback to typed data fields (include common ones)
        typed_fields = (
            ("float_data", np.float32),
            ("double_data", np.float64),
            ("int32_data", np.int32),
            ("int64_data", np.int64),
            ("bool_data", np.bool_),
        )
        for fld, dtype in typed_fields:
            if getattr(x, fld, None):
                arr = np.asarray(getattr(x, fld), dtype=dtype)
                dims = tuple(getattr(x, "dims", ()))
                return arr.reshape(dims) if dims else (arr[0] if arr.size == 1 else arr)
    except Exception:
        pass

    # Last resort
    try:
        arr_last = np.asarray(x)
        if (
            isinstance(arr_last, np.ndarray)
            and arr_last.dtype == object
            and arr_last.size == 1
        ):
            try:
                val = arr_last.reshape(()).item()
                # common string representations
                if isinstance(val, str):
                    lv = val.strip().lower()
                    if lv in ("true", "false"):
                        return np.asarray(lv == "true", dtype=np.bool_)
                return np.asarray(val)
            except Exception:
                pass
        return arr_last
    except Exception:
        return None


def _read_scalar_bool_from_value_or_constant(
    nodes: List["ir.Node"], v_or_name: Optional[object]
) -> Optional[bool]:
    """
    Try to read a scalar bool from:
      • a Value object carrying const payload, or
      • its Constant producer (found via object or name), or
      • a string tensor name produced by a Constant.
    """
    if v_or_name is None:
        return None
    # Value carries a const?
    if not isinstance(v_or_name, str):
        for attr in ("const_value", "value", "data", "numpy"):
            x = getattr(v_or_name, attr, None)
            if x is None:
                continue
            if isinstance(x, (bool, np.bool_)):
                val = bool(x)
                _dbg_tm("read Value-const (bool)", type(v_or_name).__name__, "→", val)
                return val
            arr = _to_numpy_from_any(x)
            if arr is not None:
                try:
                    val = bool(np.asarray(arr).reshape(()).astype(np.bool_).item())
                    _dbg_tm("read Value-const:", type(v_or_name).__name__, "→", val)
                    return val
                except Exception:
                    pass
    # Producer Constant?
    pidx = _find_producer_idx(nodes, v_or_name)
    if pidx is None:
        return None
    n = nodes[pidx]
    if getattr(n, "op_type", "") != "Constant":
        return None
    _dbg_tm(
        "producer Constant idx",
        pidx,
        "for",
        v_or_name if isinstance(v_or_name, str) else _v_name(v_or_name),
    )
    a = _get_attr(n, "value")
    if a is None:
        # Some builds store Constant attributes in mapping-like containers
        # or as sequences of Attr / (key, payload) pairs.
        for attr_name in ("attributes", "attribute"):
            al = getattr(n, attr_name, None)
            if al is None:
                continue
            # Normalize to an iterable of items
            try:
                it = al.values() if hasattr(al, "values") else al
            except Exception:
                it = []
            for item in it:
                # (key, payload) pair
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and item[0] == "value"
                ):
                    payload = item[1]
                    if isinstance(payload, (bool, np.bool_)):
                        val = bool(payload)
                        _dbg_tm("read Const-attr (pair-bool) →", val)
                        return val
                    arr = _to_numpy_from_any(payload)
                    if arr is not None:
                        try:
                            val = bool(
                                np.asarray(arr).reshape(()).astype(np.bool_).item()
                            )
                            _dbg_tm("read Const-attr (pair) →", val)
                            return val
                        except Exception:
                            continue
                # object with .name/.value-like fields
                name_k = getattr(item, "name", getattr(item, "key", None))
                if name_k == "value":
                    for field in ("value", "t", "i", "ints", "f", "floats", "s"):
                        if hasattr(item, field):
                            payload = getattr(item, field)
                            if isinstance(payload, (bool, np.bool_)):
                                val = bool(payload)
                                _dbg_tm("read Const-attr (obj.", field, " bool) →", val)
                                return val
                            arr = _to_numpy_from_any(payload)
                            if arr is not None:
                                try:
                                    val = bool(
                                        np.asarray(arr)
                                        .reshape(())
                                        .astype(np.bool_)
                                        .item()
                                    )
                                    _dbg_tm("read Const-attr (obj.", field, ") →", val)
                                    return val
                                except Exception:
                                    continue
        # Could not find a 'value' attribute in any known form
        return None
    # Known fields
    for field in ("value", "t", "i", "ints", "f", "floats", "s"):
        if hasattr(a, field):
            payload = getattr(a, field)
            if isinstance(payload, (bool, np.bool_)):
                val = bool(payload)
                _dbg_tm("read Const-attr (", field, " bool) →", val)
                return val
            arr = _to_numpy_from_any(payload)
            if arr is not None:
                try:
                    val = bool(np.asarray(arr).reshape(()).astype(np.bool_).item())
                    _dbg_tm("read Const-attr (", field, ") →", val)
                    return val
                except Exception:
                    continue
    # Fallback: list-like attr
    try:
        seq = list(a)
        if len(seq) >= 2:
            payload = seq[1]
            if isinstance(payload, (bool, np.bool_)):
                val = bool(payload)
                _dbg_tm("read Const-attr (seq[1] bool) →", val)
                return val
            arr = _to_numpy_from_any(payload)
            if arr is not None:
                val = bool(np.asarray(arr).reshape(()).astype(np.bool_).item())
                _dbg_tm("read Const-attr (seq[1]) →", val)
                return val
    except Exception:
        pass
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
    nodes, store = _get_node_seq_and_setter(graph)
    if not nodes:
        return
    changed = False
    del_not_nodes: List[ir.Node] = []
    del_not_names: Set[str] = set()
    for idx, n in enumerate(nodes):
        if n is None:
            continue
        if getattr(n, "op_type", "") != "Dropout":
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
        # Case A: training_mode itself constant False
        val = _read_scalar_bool_from_value_or_constant(nodes, tm)
        if val is not None and bool(val) is False:
            miss = _missing_bool_value()
            ins_new = list(ins)
            ins_new[2] = miss
            try:
                _set_node_inputs(n, ins_new)
                nodes[idx] = n
            except Exception:
                nodes[idx] = _rebuild_node_with_inputs(n, ins_new)
            changed = True
            continue
        # Case B: training_mode is Not(True)
        pidx = _find_producer_idx(nodes, tm)
        if pidx is None:
            continue
        p = nodes[pidx]
        if getattr(p, "op_type", "") != "Not":
            continue
        _dbg_tm("tm producer is Not @", pidx)
        not_in = (_node_inputs(p) or [None])[0]
        nv = _read_scalar_bool_from_value_or_constant(nodes, not_in)
        if nv is not None and bool(nv) is True:
            miss = _missing_bool_value()
            ins_new = list(ins)
            ins_new[2] = miss
            old_not_out = _node_output(p)
            _replace_everywhere(nodes, old_not_out, _v_name(old_not_out), miss)
            _replace_in_graph_outputs(graph, old_not_out, _v_name(old_not_out), miss)
            try:
                _set_node_inputs(n, ins_new)
                nodes[idx] = n
            except Exception:
                nodes[idx] = _rebuild_node_with_inputs(n, ins_new)
            changed = True
            del_not_nodes.append(p)
            out_v = _node_output(p)
            out_name = _v_name(out_v)
            if out_name:
                del_not_names.add(out_name)
            try:
                nodes[pidx] = None
            except Exception:
                pass
        else:
            _dbg_tm("Not input could not be proven True; nv=", nv)
    if changed:
        # Explicitly delete orphan Not producers we identified
        # Remove any Not nodes whose outputs have no remaining consumers
        if del_not_names:
            still_needed: Set[str] = set()
            for m in nodes:
                if m is None:
                    continue
                for iv in _node_inputs(m):
                    nm = _v_name(iv)
                    if nm:
                        still_needed.add(nm)
            del_not_names = {nm for nm in del_not_names if nm not in still_needed}

        new_nodes: List[ir.Node] = []
        for m in nodes:
            if m is None:
                continue
            out_names = {_v_name(ov) for ov in _node_outputs(m)}
            if any(m is rem for rem in del_not_nodes) or (
                del_not_names and (out_names & del_not_names)
            ):
                if TRN_DEBUG or os.getenv("JAX2ONNX_TM_DEBUG"):
                    print("[tm-inline] removed orphan Not")
                continue
            new_nodes.append(m)
        nodes = new_nodes
        _set_nodes(store, nodes)


# ---------------- Graph IO (for DCE/prune) ----------------


def _graph_inputs_list(graph) -> List["ir.Value"]:
    for attr in ("inputs", "input"):
        ins = getattr(graph, attr, None)
        if ins is not None:
            try:
                return list(ins)
            except Exception:
                pass
    return []


def _graph_outputs_list(graph) -> List["ir.Value"]:
    for attr in ("outputs", "output"):
        outs = getattr(graph, attr, None)
        if outs is not None:
            try:
                return list(outs)
            except Exception:
                pass
    return []


# ---------------- DCE ----------------


def remove_dead_nodes_ir(graph) -> None:
    nodes, store = _get_node_seq_and_setter(graph)
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
    _set_nodes(store, new_nodes)


# ---------------- Prune unused graph inputs (top graph only) ----------------


def _attr_kind(attr: object) -> Optional[str]:
    if attr is None:
        return None
    atype = getattr(attr, "type", None)
    if atype is None:
        return None
    if isinstance(atype, str):
        return atype.upper()
    name = getattr(atype, "name", None)
    if isinstance(name, str):
        return name.upper()
    try:
        val = int(atype)
    except Exception:
        return None
    for label in ("GRAPH", "GRAPHS"):
        try:
            enum_val = getattr(IRAttrType, label)
        except AttributeError:
            continue
        try:
            if val == int(enum_val.value):
                return label
        except Exception:
            if val == int(enum_val):
                return label
    return None


def _iter_node_attrs(node: object) -> Iterable[object]:
    raw = getattr(node, "attributes", None)
    if raw:
        if hasattr(raw, "values"):
            for val in raw.values():
                yield val
        elif isinstance(raw, dict):
            for val in raw.values():
                yield val
        else:
            for val in raw:
                yield val
    raw_alt = getattr(node, "attribute", None)
    if raw_alt:
        for val in raw_alt:
            yield val


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
                sub = getattr(attr, "value", None)
                if sub is None:
                    sub = getattr(attr, "g", None)
                if sub is not None:
                    _collect_used_value_names(sub, used)
            elif kind == "GRAPHS":
                subs = getattr(attr, "value", None)
                if subs is None:
                    subs = getattr(attr, "graphs", None)
                if subs is None:
                    continue
                try:
                    iterator = list(subs)
                except Exception:
                    iterator = subs
                for sub in iterator or []:
                    if sub is None:
                        continue
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

    for attr in ("inputs", "input"):
        arr = getattr(graph, attr, None)
        if arr is None:
            continue
        try:
            lst = list(arr)
        except Exception:
            continue
        keep = []
        removed = []
        for v in lst:
            nm = _v_name(v)
            if not nm:
                keep.append(v)
                continue

            should_keep = False
            if _should_always_keep(nm):
                should_keep = True
            elif nm in output_names:
                should_keep = True
            elif _count_consumers(nodes or [], nm, v) > 0:
                should_keep = True
            elif nm in used:
                should_keep = True

            if should_keep:
                keep.append(v)
            else:
                removed.append(nm)
        if removed and DEBUG:
            _dbg(f"prune_unused_graph_inputs_ir removed: {removed}")
        try:
            setattr(graph, attr, keep)
        except Exception:
            try:
                arr[:] = keep
            except Exception:
                try:
                    arr.clear()
                    arr.extend(keep)
                except Exception:
                    pass
        break


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def optimize_graph(ir_model: ir.Model) -> ir.Model:
    # Top graph
    try:
        gr = getattr(ir_model, "graph", None)
        if gr is not None:
            remove_redundant_casts_ir(gr)
            remove_redundant_transpose_pairs_ir(gr)
            remove_redundant_reshape_pairs_ir(gr)
            # Constant-only: inline training_mode when provably false / Not(True)
            inline_dropout_training_mode_constants_ir(gr)
            propagate_unary_shapes_ir(gr)
            remove_redundant_casts_ir(gr)
            remove_dead_nodes_ir(gr)
            prune_unused_graph_inputs_ir(gr)
    except Exception as _e:
        _dbg("optimize_graph: top-graph pass skipped:", _e)

    # Function bodies – do NOT prune function inputs (signature!)
    try:
        funcs = getattr(ir_model, "functions", None) or getattr(
            ir_model, "_functions", None
        )
        if isinstance(funcs, dict):
            values: Iterable[Any] = funcs.values()
        elif funcs is None:
            values = ()
        else:
            values = funcs
        for fn in values:
            fgr = getattr(fn, "graph", None)
            if fgr is not None:
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
