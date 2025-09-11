# jax2onnx/converter2/ir_optimizations.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
import os
import numpy as np

import onnx_ir as ir

# ---------------- Config ----------------

ALLOWED_ELEMENTWISE_OPS: Set[str] = {
    "elu", "gelu", "relu", "sigmoid", "tanh", "dropout",
    "leakyrelu", "identity", "cast", "castlike", "not",
}

ALLOWED_ELEMWISE = {
    "Elu","Gelu","Relu","Sigmoid","Tanh","Dropout",
    "LeakyRelu","Identity","Cast","CastLike","Not",
}

DEBUG = bool(int(os.getenv("JAX2ONNX_IROPT_DEBUG", "0")))
RSH_DEBUG = bool(int(os.getenv("JAX2ONNX_RSH_DEBUG", "0")))
TRN_DEBUG = bool(int(os.getenv("JAX2ONNX_TRN_DEBUG", "0")))
DCE_DEBUG = bool(int(os.getenv("JAX2ONNX_DCE_DEBUG", "0")))
DRPT_DEBUG = bool(int(os.getenv("JAX2ONNX_DRPT_DEBUG", "0")))

# ---------------- Debug ----------------

def _dbg(*a):
    if DEBUG:
        print("[iropt]", *a)

def _d_dropout(*a):
    if DRPT_DEBUG:
        print("[dropout-inline]", *a)

# ---------------- IR helpers ----------------

def _v_name(v: ir.Value | None) -> Optional[str]:
    if v is None:
        return None
    nm = getattr(v, "name", None)
    return nm if isinstance(nm, str) and nm != "" else None

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

def _get_node_seq_and_setter(graph) -> Tuple[List["ir.Node"], Optional[Tuple[object, str]]]:
    for attr in ("nodes", "_nodes", "node"):
        if hasattr(graph, attr):
            nodes = list(getattr(graph, attr))
            return nodes, (graph, attr)
    return [], None

def _set_nodes(store: Optional[Tuple[object, str]], nodes: List["ir.Node"]) -> None:
    if store is None:
        return
    parent, attr = store
    try:
        setattr(parent, attr, nodes)
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
            if (old_v is not None and iv is old_v) or (old_name and _v_name(iv) == old_name):
                ins[i] = new_v
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
                m.inputs = tuple(ins)
            except Exception:
                try:
                    m.inputs = list(ins)
                except Exception:
                    pass

def _replace_in_graph_outputs(graph, old_v: Optional["ir.Value"], old_name: Optional[str], new_v: "ir.Value") -> None:
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
            if (old_v is not None and ov is old_v) or (old_name and _v_name(ov) == old_name):
                lst[i] = new_v
                changed = True
        if changed:
            try:
                setattr(graph, attr, lst)
            except Exception:
                # last resort: rename
                for i, ov in enumerate(getattr(graph, attr, [])):
                    if (old_v is not None and ov is old_v) or (old_name and _v_name(ov) == old_name):
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

def _unique_consumer(cons_by_obj, cons_by_name, val: Optional["ir.Value"]) -> Optional[int]:
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
    for attr_name in ("attributes", "attrs", "_attrs"):
        d = getattr(node, attr_name, None)
        if isinstance(d, dict) and name in d:
            return d[name]
    seq = getattr(node, "attribute", None)
    if isinstance(seq, (list, tuple)):
        for a in seq:
            if getattr(a, "name", None) == name:
                return a
    if hasattr(node, name):
        return getattr(node, name)
    return None

# ---------------- Transpose folding ----------------

def _transpose_perm(node) -> Optional[List[int]]:
    a = _get_attr(node, "perm")
    if a is None:
        return None
    ints = getattr(a, "ints", None)
    if ints is not None:
        return list(ints)
    try:
        seq = list(a)
        if all(isinstance(x, int) for x in seq):
            return seq
    except Exception:
        pass
    if isinstance(a, dict) and "ints" in a and isinstance(a["ints"], (list, tuple)):
        return list(a["ints"])
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
                print("[transposefold]", [nodes[k].op_type for k in chain_idx], "perm1", perm1, "perm2", perm2)
            t1_in = (_node_inputs(T1) or [None])[0]
            if t1_in is None:
                i += 1
                continue
            if allowed_idx:
                first_allowed = nodes[allowed_idx[0]]
                last_allowed = nodes[allowed_idx[-1]]
                _replace_everywhere([first_allowed], _node_output(T1), _v_name(_node_output(T1)), t1_in)
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
                print("[reshapefold/up]", [nodes[k].op_type for k in ([T1_idx] + allowed_fwd + [i])],
                      "src", _shape_tuple(src), "dst", _shape_tuple(dst))
            if allowed_fwd:
                first_allowed = nodes[allowed_fwd[0]]
                last_allowed  = nodes[allowed_fwd[-1]]
                _replace_everywhere([first_allowed], _node_output(T1), _v_name(_node_output(T1)), src)
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

# ---------------- Dropout.training_mode inlining (missing-input sentinel) ----------------

def _unwrap_bool_chain(nodes, start_v, prod_obj, prod_name, max_depth=6):
    """Unwrap Not/Identity/Cast/CastLike; return (origin_v, origin_pidx, not_parity, chain_ops)."""
    parity = 0
    v = start_v
    chain = []
    depth = 0
    while v is not None and depth < max_depth:
        pidx = _producer_idx_for(v, prod_obj, prod_name)
        if pidx is None:
            return v, None, parity, chain
        n = nodes[pidx]
        op = n.op_type
        chain.append(op)
        if op == "Not":
            parity ^= 1
            v = (_node_inputs(n) or [None])[0]
            depth += 1
            continue
        if op in {"Identity","Cast","CastLike"}:
            v = (_node_inputs(n) or [None])[0]
            depth += 1
            continue
        return v, pidx, parity, chain
    return v, _producer_idx_for(v, prod_obj, prod_name), parity, chain

def _missing_input_value() -> "ir.Value":
    # ONNX denotes missing optional inputs by an empty string.
    return ir.Value(name="", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(()))

def inline_dropout_training_mode_ir(graph) -> None:
    """
    If training_mode resolves to:
      • constant False  (or Not(True)) → replace input #2 with a *missing* input (empty name).
      • a scalar-bool graph input used only to drive this flag through Not/Identity/Cast* → same.
    This preserves node count (no Constant), keeps Dropout present, and DCE removes Not.
    """
    nodes, store = _get_node_seq_and_setter(graph)
    if not nodes:
        return

    prod_obj, prod_name, cons_by_obj, cons_by_name = _build_use_maps(nodes)
    graph_inputs = set(id(v) for v in _graph_inputs_list(graph))

    changed = False
    i = 0
    while i < len(nodes):
        n = nodes[i]
        if getattr(n, "op_type", None) != "Dropout":
            i += 1
            continue

        ins = _node_inputs(n)
        if len(ins) < 3:
            i += 1
            continue
        tm = ins[2]

        origin_v, origin_pidx, parity, chain = _unwrap_bool_chain(nodes, tm, prod_obj, prod_name)

        # Case 1: constant path
        cval = None
        for attr in ("const_value", "value", "data", "numpy"):
            x = getattr(origin_v, attr, None) if origin_v is not None else None
            if x is not None:
                try:
                    cval = bool(np.asarray(x).reshape(()).astype(np.bool_).item())
                except Exception:
                    pass
                break

        if cval is not None:
            eff = (not cval) if parity == 1 else bool(cval)
            _d_dropout(f"Dropout@{i}: tm constant chain={chain} cval={cval} parity={parity} -> eff={eff}")
            if eff is False:
                miss = _missing_input_value()
                if hasattr(n, "replace_input_with"):
                    try:
                        n.replace_input_with(2, miss)
                    except Exception:
                        pass
                ins2 = _node_inputs(n)
                try:
                    ins2[2] = miss
                    n.inputs = tuple(ins2)
                except Exception:
                    try:
                        n.inputs = list(ins2)
                    except Exception:
                        pass
                changed = True
            i += 1
            continue

        # Case 2: origin graph input used only for this corridor → inline to missing
        if origin_pidx is None and id(origin_v) in graph_inputs:
            uses = _all_consumers(cons_by_obj, cons_by_name, origin_v)
            _d_dropout(f"Dropout@{i}: tm from graph input {_v_name(origin_v)} chain={chain} parity={parity} consumers={sorted(list(uses))}")
            ok = len(uses) <= 1 or (len(uses) == 2 and i in uses)
            if ok:
                miss = _missing_input_value()
                if hasattr(n, "replace_input_with"):
                    try:
                        n.replace_input_with(2, miss)
                    except Exception:
                        pass
                ins2 = _node_inputs(n)
                try:
                    ins2[2] = miss
                    n.inputs = tuple(ins2)
                except Exception:
                    try:
                        n.inputs = list(ins2)
                    except Exception:
                        pass
                changed = True
            i += 1
            continue

        _d_dropout(f"Dropout@{i}: did not inline (origin_pidx={origin_pidx}, chain={chain})")
        i += 1

    if changed:
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
    worklist: List["ir.Value"] = [v for v in _graph_outputs_list(graph) if v is not None]
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

def prune_unused_graph_inputs_ir(graph) -> None:
    """
    Remove graph inputs that are not consumed by any node and are not graph outputs.
    (We do NOT run this on function bodies to avoid changing function signatures.)
    """
    # Collect used names from node inputs and graph outputs
    nodes, _ = _get_node_seq_and_setter(graph)
    used: Set[str] = set()
    for n in nodes:
        for iv in _node_inputs(n):
            nm = _v_name(iv)
            if nm:
                used.add(nm)
    for ov in _graph_outputs_list(graph):
        nm = _v_name(ov)
        if nm:
            used.add(nm)

    # Read inputs container
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
            # Keep if used or unnamed (shouldn't happen) or empty name
            if nm and (nm in used):
                keep.append(v)
            elif not nm:
                keep.append(v)
            else:
                removed.append(nm)
        if removed and DEBUG:
            _dbg(f"prune_unused_graph_inputs_ir removed: {removed}")
        try:
            setattr(graph, attr, keep)
        except Exception:
            # fall back to in-place clear+extend if list-like
            try:
                arr[:] = keep
            except Exception:
                pass
        break  # we set one of inputs/input; don't set both

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def optimize_graph(ir_model: ir.Model) -> ir.Model:
    # Top graph
    try:
        gr = getattr(ir_model, "graph", None)
        if gr is not None:
            remove_redundant_transpose_pairs_ir(gr)
            remove_redundant_reshape_pairs_ir(gr)
            inline_dropout_training_mode_ir(gr)
            remove_dead_nodes_ir(gr)
            prune_unused_graph_inputs_ir(gr)
    except Exception as _e:
        _dbg("optimize_graph: top-graph pass skipped:", _e)

    # Function bodies – do NOT prune function inputs (signature!)
    try:
        funcs = getattr(ir_model, "functions", None) or getattr(ir_model, "_functions", None)
        values = funcs.values() if isinstance(funcs, dict) else funcs
        for fn in values:
            fgr = getattr(fn, "graph", None)
            if fgr is not None:
                try:
                    remove_redundant_transpose_pairs_ir(fgr)
                    remove_redundant_reshape_pairs_ir(fgr)
                    inline_dropout_training_mode_ir(fgr)
                    remove_dead_nodes_ir(fgr)
                except Exception as _fe:
                    _dbg("optimize_graph: function pass skipped:", _fe)
    except Exception as _e:
        _dbg("optimize_graph: functions traversal skipped:", _e)

    return ir_model
