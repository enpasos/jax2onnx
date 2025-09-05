from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional

import onnx_ir as ir

# ---------------- Config ----------------

ALLOWED_ELEMENTWISE_OPS: Set[str] = {
    # lowercased
    "elu",
    "gelu",
    "relu",
    "sigmoid",
    "tanh",
    "dropout",
    "leakyrelu",
    "identity",
    "cast",
    "castlike",
    # add more benign ops as you encounter them, e.g. "clip"
}

DEBUG = True  # set True temporarily to print reasons

# ---------------- Debug ----------------


def _dbg(*a):
    if DEBUG:
        print("[iropt]", *a)


# ---------------- IR access helpers ----------------


def _is_elem(op_type: str) -> bool:
    return op_type.lower() in ALLOWED_ELEMENTWISE_OPS


def _value_name(v: ir.Value | None) -> Optional[str]:
    if v is None:
        return None
    nm = getattr(v, "name", None)
    return nm if isinstance(nm, str) and nm != "" else None


def _node_outputs(n: ir.Node) -> List[ir.Value]:
    return list(getattr(n, "outputs", []) or [])


def _node_output(n: ir.Node) -> Optional[ir.Value]:
    outs = _node_outputs(n)
    return outs[0] if outs else None


def _node_inputs(n: ir.Node) -> List[ir.Value]:
    return list(getattr(n, "inputs", []) or [])


def _get_perm_attr(node: ir.Node) -> Optional[List[int]]:
    """Best-effort read of a Transpose 'perm' attribute across ir variants."""
    attrs = getattr(node, "attributes", None)
    if not attrs:
        return None

    def _attr_name(a) -> Optional[str]:
        for fld in ("name", "key", "attr_name", "_name", "_key"):
            nm = getattr(a, fld, None)
            if isinstance(nm, str) and nm:
                return nm
        # some irs wrap the proto
        proto = getattr(a, "proto", None)
        if proto is not None:
            nm = getattr(proto, "name", None)
            if isinstance(nm, str) and nm:
                return nm
        return None

    def _as_int_list(x) -> Optional[List[int]]:
        if x is None:
            return None
        try:
            return [int(v) for v in list(x)]
        except Exception:
            return None

    for a in attrs:
        nm = _attr_name(a)
        if nm != "perm":
            continue

        # direct fields on the attribute
        for fld in ("ints", "int64s", "vals", "value", "data"):
            val = getattr(a, fld, None)
            ints = _as_int_list(val)
            if ints is not None:
                return ints

        # nested object under .value with various payload names
        val = getattr(a, "value", None)
        if val is not None:
            for fld in ("ints", "int64s", "vals", "data"):
                ints = _as_int_list(getattr(val, fld, None))
                if ints is not None:
                    return ints
            # methods on the nested value
            for meth in ("as_tuple", "to_list", "tolist", "to_tuple"):
                f = getattr(val, meth, None)
                if callable(f):
                    ints = _as_int_list(f())
                    if ints is not None:
                        return ints

        # methods on the attribute itself
        for meth in ("as_tuple", "to_list", "tolist", "to_tuple"):
            f = getattr(a, meth, None)
            if callable(f):
                ints = _as_int_list(f())
                if ints is not None:
                    return ints

        if DEBUG:
            # last resort: show what we found on this attribute
            fields = {
                k: getattr(a, k)
                for k in dir(a)
                if not k.startswith("_") and not callable(getattr(a, k))
            }
            _dbg("perm attr could not be parsed; fields:", fields)

    return None


def _apply_perm(perm: List[int], axes: List[int]) -> List[int]:
    # ONNX: out[i] = in[perm[i]]
    return [axes[perm[i]] for i in range(len(perm))]


def _perms_compose_identity(p1: List[int] | None, p2: List[int] | None) -> bool:
    if p1 is None or p2 is None or len(p1) != len(p2):
        return False
    axes = list(range(len(p1)))
    return _apply_perm(p2, _apply_perm(p1, axes)) == axes


# ---------------- shape helpers (fallback when perm is missing) ----------------


def _shape_tuple(v: ir.Value | None) -> Optional[Tuple]:
    """
    Return a tuple of dims for v.shape if we can read it; elements are ints or strings.
    Unknown/unreadable -> None.
    """
    if v is None:
        return None
    sh = getattr(v, "shape", None)
    if sh is None:
        return None
    # common patterns across ir variants
    for fld in ("dims", "shape", "_dims"):
        dims = getattr(sh, fld, None)
        if isinstance(dims, (list, tuple)):
            try:
                return tuple(
                    (
                        int(d)
                        if isinstance(d, bool) or getattr(d, "__int__", None)
                        else str(d)
                    )
                    for d in dims
                )
            except Exception:
                # fall through
                pass
    # final attempt: iterable shape itself
    try:
        return tuple(
            int(d) if isinstance(d, bool) or getattr(d, "__int__", None) else str(d)
            for d in list(sh)
        )
    except Exception:
        return None


def _shapes_equal(a: ir.Value | None, b: ir.Value | None) -> bool:
    ta, tb = _shape_tuple(a), _shape_tuple(b)
    if ta is None or tb is None:
        return False
    if len(ta) != len(tb):
        return False
    # strict elementwise comparison (int==int, str==str)
    return all(x == y for x, y in zip(ta, tb))


# ---------------- node list binding ----------------


def _get_node_seq_and_setter(ctx) -> Tuple[List[ir.Node], Optional[Tuple[object, str]]]:
    """Return a live reference to the IR node list and (parent, attr) to persist changes."""

    def _candidate(obj):
        if obj is None:
            return None
        for attr in ("nodes", "_nodes", "ops", "_ops", "operations", "_operations"):
            seq = getattr(obj, attr, None)
            if isinstance(seq, list) and (not seq or isinstance(seq[0], ir.Node)):
                return seq, (obj, attr)
        return None

    cand = _candidate(ctx)
    if cand:
        return cand
    bld = getattr(ctx, "builder", None)
    cand = _candidate(bld)
    if cand:
        return cand
    g = getattr(bld, "graph", None) or getattr(bld, "_graph", None)
    cand = _candidate(g)
    if cand:
        return cand
    seq = (
        getattr(ctx, "_nodes", [])
        if isinstance(getattr(ctx, "_nodes", []), list)
        else []
    )
    return seq, None


def _get_output_lists(ctx) -> List[List[ir.Value]]:
    """Collect IR output lists to rewire graph/model outputs."""
    outs: List[List[ir.Value]] = []
    for obj in (
        ctx,
        getattr(ctx, "builder", None),
        getattr(getattr(ctx, "builder", None), "graph", None),
        getattr(getattr(ctx, "builder", None), "_graph", None),
    ):
        if obj is None:
            continue
        for attr in ("outputs", "_outputs"):
            seq = getattr(obj, attr, None)
            if isinstance(seq, list) and (not seq or isinstance(seq[0], ir.Value)):
                outs.append(seq)
    return outs


# ---------------- consumer matching (name OR object) ----------------


def _has_input_name_or_obj(
    n: ir.Node, name: Optional[str], obj: Optional[ir.Value]
) -> bool:
    for v in _node_inputs(n):
        if obj is not None and v is obj:
            return True
        if name is not None and _value_name(v) == name:
            return True
    return False


def _count_consumers(
    nodes: List[ir.Node], name: Optional[str], obj: Optional[ir.Value]
) -> int:
    return sum(1 for n in nodes if _has_input_name_or_obj(n, name, obj))


def _find_next_consumer_idx(
    nodes: List[ir.Node], start_idx: int, name: Optional[str], obj: Optional[ir.Value]
) -> Optional[int]:
    j = start_idx + 1
    while j < len(nodes):
        if _has_input_name_or_obj(nodes[j], name, obj):
            return j
        j += 1
    return None


# ---------------- rewiring ----------------


def _replace_all_inputs_and_outputs(
    ctx,
    nodes: List[ir.Node],
    old_name: Optional[str],
    old_obj: Optional[ir.Value],
    new_v: ir.Value,
) -> None:
    """Rewire every consumer (by name OR obj) and any graph outputs, and var->value map."""
    for m in nodes:
        ins = tuple(getattr(m, "inputs", []) or [])
        for idx, iv in enumerate(ins):
            if (old_obj is not None and iv is old_obj) or (
                old_name is not None and _value_name(iv) == old_name
            ):
                if hasattr(m, "replace_input_with"):
                    m.replace_input_with(idx, new_v)  # <-- PASS INDEX
                else:
                    # fallback for very old backends
                    tmp = list(ins)
                    tmp[idx] = new_v
                    try:
                        m.inputs = tuple(tmp)
                    except Exception:
                        pass
        # nothing else needed; replace_input_with updates m.inputs internally
    for out_list in _get_output_lists(ctx):
        for i, ov in enumerate(list(out_list)):
            if (old_obj is not None and ov is old_obj) or (
                old_name is not None and _value_name(ov) == old_name
            ):
                out_list[i] = new_v
    var2val = getattr(ctx, "_var2val", None)
    if isinstance(var2val, dict):
        for k, v in list(var2val.items()):
            if (old_obj is not None and v is old_obj) or (
                old_name is not None and _value_name(v) == old_name
            ):
                var2val[k] = new_v


def _rewire_model_outputs(ctx, nodes: List[ir.Node]) -> None:
    """After structural rewrites, ensure model/graph outputs reference live tensors."""
    live_by_name: Dict[str, ir.Value] = {}
    live_objs: set[ir.Value] = set()
    for n in nodes:
        for v in _node_outputs(n):
            if v is None:
                continue
            live_objs.add(v)
            nm = _value_name(v)
            if nm:
                live_by_name[nm] = v
    for out_list in _get_output_lists(ctx):
        for i, ov in enumerate(list(out_list)):
            if ov in live_objs:
                continue
            nm = _value_name(ov)
            if nm and nm in live_by_name:
                out_list[i] = live_by_name[nm]


# ---------------- one fold attempt ----------------


def _try_fold_from(ctx, nodes: List[ir.Node], i: int) -> Tuple[bool, int]:
    """
    Try to fold a chain starting at nodes[i] if it's a Transpose.
    Returns (did_fold, next_index_to_continue_from).
    """
    t1 = nodes[i]
    if t1.op_type != "Transpose":
        return False, i + 1

    t1_out = _node_output(t1)
    t1_out_name = _value_name(t1_out)
    if t1_out is None and t1_out_name is None:
        _dbg("skip: t1 has no output")
        return False, i + 1

    # Ensure t1 output is unbranched (safety)
    if _count_consumers(nodes, t1_out_name, t1_out) != 1:
        _dbg("skip: t1 output branches", t1_out_name)
        return False, i + 1

    # Follow the real consumer chain (name or object), skipping unrelated nodes
    chain: List[ir.Node] = [t1]
    cur_name, cur_obj = t1_out_name, t1_out
    pos = i
    while True:
        nxt_idx = _find_next_consumer_idx(nodes, pos, cur_name, cur_obj)
        if nxt_idx is None:
            _dbg("stop: no next consumer for", cur_name)
            return False, i + 1
        nxt = nodes[nxt_idx]
        if _is_elem(nxt.op_type):
            chain.append(nxt)
            ev = _node_output(nxt)
            cur_name = _value_name(ev) or cur_name  # propagate if named
            cur_obj = ev or cur_obj  # keep object flow
            pos = nxt_idx
            if _count_consumers(nodes, cur_name, cur_obj) != 1:
                _dbg("abort chain: elem output branches", cur_name)
                return False, i + 1
            continue
        if nxt.op_type == "Transpose":
            chain.append(nxt)
        else:
            _dbg("stop: non-elem non-transpose", nxt.op_type)
            return False, i + 1
        break  # hit t2

    if len(chain) < 2 or chain[-1].op_type != "Transpose":
        return False, i + 1

    t2 = chain[-1]
    p1, p2 = _get_perm_attr(t1), _get_perm_attr(t2)
    if not _perms_compose_identity(p1, p2):
        # ----- Fallback: if both perms are missing, use shape identity check -----
        if p1 is None and p2 is None:
            t1_in = (_node_inputs(t1) or [None])[0]
            t2_out = _node_output(t2)
            if _shapes_equal(t1_in, t2_out):
                _dbg("fallback: shapes equal, folding", _shape_tuple(t1_in))
                # proceed to rewire and delete below
            else:
                _dbg(
                    "skip: perms missing and shapes differ",
                    _shape_tuple(t1_in),
                    _shape_tuple(t2_out),
                )
                return False, i + 1
        else:
            _dbg("skip: perms not identity", p1, p2, "on", [n.op_type for n in chain])
            return False, i + 1

    # Rewire
    t1_in = (_node_inputs(t1) or [None])[0]
    t2_out = _node_output(t2)
    if t1_in is None or t2_out is None:
        _dbg("skip: missing t1_in/t2_out")
        return False, i + 1

    if len(chain) > 2:
        # first elem consumes T1 input instead of T1 output
        first_elem = chain[1]
        ins = tuple(getattr(first_elem, "inputs", []) or [])
        for idx, iv in enumerate(ins):
            if iv is t1_out or _value_name(iv) == t1_out_name:
                if hasattr(first_elem, "replace_input_with"):
                    first_elem.replace_input_with(
                        idx, t1_in
                    )  # <-- PASS INDEX, not Value
                else:
                    # fallback for very old backends
                    tmp = list(ins)
                    tmp[idx] = t1_in
                    try:
                        first_elem.inputs = tuple(tmp)
                    except Exception:
                        pass
                break
        last_elem_out = _node_output(chain[-2]) or t1_in
        _replace_all_inputs_and_outputs(
            ctx,
            nodes,
            old_name=_value_name(t2_out),
            old_obj=t2_out,
            new_v=last_elem_out,
        )
    else:
        _replace_all_inputs_and_outputs(
            ctx, nodes, old_name=_value_name(t2_out), old_obj=t2_out, new_v=t1_in
        )

    # Remove (higher index first)
    idx_t2 = nodes.index(t2)
    del nodes[idx_t2]
    del nodes[i]
    _dbg("FOLD:", [n.op_type for n in chain])
    return True, i  # continue from current index


# ---------------- public pass ----------------


def remove_redundant_transpose_pairs_ir(ctx) -> None:
    """Fold Transpose → [pure elementwise]* → Transpose when perms compose to identity."""
    nodes, store = _get_node_seq_and_setter(ctx)
    if not nodes:
        return

    _dbg(
        "nodes:", len(nodes), "transpose:", sum(n.op_type == "Transpose" for n in nodes)
    )

    changed_any = True
    while changed_any:
        changed_any = False
        i = 0
        while i < len(nodes):
            did, nxt = _try_fold_from(ctx, nodes, i)
            if did:
                changed_any = True
                if store is not None:
                    parent, attr = store
                    setattr(parent, attr, nodes)
                i = nxt
            else:
                i = nxt

    _rewire_model_outputs(ctx, nodes)
    if store is not None:
        parent, attr = store
        setattr(parent, attr, nodes)
        parent, attr = store
        setattr(parent, attr, nodes)
