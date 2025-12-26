# jax2onnx/converter/ir_optimizations.py

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence as SequenceABC
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Set,
    Iterable,
    Any,
    Sequence,
    TypeAlias,
    cast,
    Union,
)
import os
import numpy as np

import onnx_ir as ir
from onnx_ir import AttributeType as IRAttrType
from onnx_ir.passes import common as common_passes


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
ArrayND = np.ndarray[Any, np.dtype[Any]]


def _as_ndarray(value: object, *, dtype: np.dtype[Any] | None = None) -> ArrayND:
    """Typed wrapper around np.asarray to satisfy mypy."""
    return cast(ArrayND, np.asarray(value, dtype=dtype))


# ---------------- Debug ----------------


def _dbg(*a: object) -> None:
    if DEBUG:
        print("[iropt]", *a)


def _dbg_tm(*a: object) -> None:
    if TM_DEBUG:
        print("[tm-inline]", *a)


# ---------------- Public helper shims (restored for unit tests) ----------------


def _get_perm_attr(node: ir.Node) -> Optional[List[int]]:
    """
    Return the Transpose 'perm' attribute as a list of ints, or None.
    """
    if not isinstance(node, ir.Node):
        return None

    attr = node.attributes.get("perm")
    if isinstance(attr, ir.Attr):
        # Check type is INTS before calling as_ints()
        if attr.type == IRAttrType.INTS:
            return [int(x) for x in attr.as_ints()]
        return None

    # Fallback: attribute not found or not an Attr object
    return None


def _value_identity(
    value_or_name: Union[ir.Value, str, None],
) -> Tuple[Optional[ir.Value], Optional[str]]:
    if value_or_name is None:
        return None, None
    if isinstance(value_or_name, ir.Value):
        return value_or_name, _v_name(value_or_name)
    if isinstance(value_or_name, str):
        return None, value_or_name or None
    return None, None


def _has_input_name_or_obj(
    node: ir.Node, name: Optional[str], obj: Optional[ir.Value]
) -> bool:
    """
    Return True if 'node' has an input that matches either the given name
    (by .name on Value or string equality) or the given object identity.
    """
    if not isinstance(node, ir.Node):
        return False

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
    return False


def _consumer_nodes(
    nodes: Sequence[ir.Node], value_or_name: Union[ir.Value, str, None]
) -> List[ir.Node]:
    """
    Return consumer nodes for a value, preferring IR APIs with name-based fallback.
    """
    if value_or_name is None:
        return []
    if isinstance(value_or_name, ir.Value):
        consumers = value_or_name.consumers()
        if consumers:
            return list(consumers)

    ref_value, ref_name = _value_identity(value_or_name)
    target_name = ref_name
    if target_name is None and isinstance(value_or_name, ir.Value):
        target_name = _v_name(value_or_name)

    found: List[ir.Node] = []
    for node in nodes:
        if _has_input_name_or_obj(
            node,
            target_name,
            (
                ref_value
                if ref_value is not None
                else (value_or_name if isinstance(value_or_name, ir.Value) else None)
            ),
        ):
            # Wait, value_or_name could be str.
            # _has_input_name_or_obj takes (node, name, obj: Optional[ir.Value])
            obj_arg = ref_value
            if obj_arg is None and isinstance(value_or_name, ir.Value):
                obj_arg = value_or_name

            if _has_input_name_or_obj(node, target_name, obj_arg):
                found.append(node)
    return found


def _producer_node(
    nodes: Sequence[ir.Node], value_or_name: Union[ir.Value, str, None]
) -> Optional[ir.Node]:
    """
    Return the producer node for a value/name, preferring IR APIs with fallback scans.
    """
    if value_or_name is None:
        return None
    if isinstance(value_or_name, ir.Value):
        prod = value_or_name.producer()
        if prod is not None:
            return prod

    ref_value, ref_name = _value_identity(value_or_name)
    target_name = ref_name
    if target_name is None and isinstance(value_or_name, ir.Value):
        target_name = _v_name(value_or_name)

    for node in nodes:
        for ov in _node_outputs(node):
            if ref_value is not None and ov is ref_value:
                return node
            if target_name and _v_name(ov) == target_name:
                return node
    return None


# ---------------- IR helpers ----------------


def _v_name(v: Union[ir.Value, str, None]) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    if isinstance(v, ir.Value):
        name = v.name
        return name or None
    return None


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


def _node_output(n: ir.Node) -> Optional[ir.Value]:
    outs = _node_outputs(n)
    return outs[0] if outs else None


def _node_inputs(n: ir.Node) -> ValueList:
    return list(cast(ValueSeq, n.inputs))


def _set_node_inputs(n: ir.Node, new_ins: Sequence[ir.Value]) -> None:
    for idx, val in enumerate(new_ins):
        n.replace_input_with(idx, val)


def _shape_dims_seq(
    shape: Union[ir.Shape, Sequence[Any], None],
) -> Optional[Tuple[Any, ...]]:
    if shape is None:
        return None
    if isinstance(shape, ir.Shape):
        return tuple(shape.dims)
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        return tuple(shape)
    return None


def _shape_tuple(v: Optional[ir.Value]) -> Optional[Tuple[int, ...]]:
    if v is None:
        return None
    dims = _shape_dims_seq(v.shape)
    if dims is None:
        return None
    tuple_dims: List[int] = []
    for d in dims:
        if isinstance(d, int):
            tuple_dims.append(d)
        else:
            tuple_dims.append(-1)
    return tuple(tuple_dims)


def _shape_dims_key(
    shape: Union[ir.Shape, Sequence[Any], None],
) -> Optional[Tuple[str, ...]]:
    """Return a hashable key representing the shape's dimensions."""
    dims = _shape_dims_seq(shape)
    if dims is None:
        return None
    key: List[str] = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            key.append(f"int:{int(d)}")
        else:
            key.append(f"repr:{repr(d)}")
    return tuple(key)


def _value_const_ints(val: Optional[ir.Value]) -> Optional[Tuple[int, ...]]:
    if not isinstance(val, ir.Value):
        return None
    arr = _to_numpy_from_any(val.const_value)
    if arr is None:
        return None
    np_arr = np.asarray(arr)
    if np_arr.dtype is None or np_arr.dtype.kind not in {"i"}:
        return None
    return tuple(int(x) for x in np_arr.reshape(-1).tolist())


def _shapes_compatible(a: Optional[ir.Value], b: Optional[ir.Value]) -> bool:
    try:
        da = _shape_dims_seq(a.shape) if a else None
        db = _shape_dims_seq(b.shape) if b else None
        if da is None or db is None:
            return False
        if len(da) != len(db):
            return False
        for x, y in zip(da, db):
            if isinstance(x, int) and isinstance(y, int) and x != y:
                return False
        return True
    except Exception:
        return False


def _replace_everywhere(
    nodes: List[ir.Node],
    old_v: Optional[ir.Value],
    old_name: Optional[str],
    new_v: ir.Value,
) -> None:
    """
    Rewire a value across a specific node subset.

    Several optimizations (reshape/transposed pair folding, dropout rewrites,
    etc.) need to redirect an intermediate edge only for the linear chain of
    nodes they touch, while other consumers of the same value must remain
    unchanged.  The global helper ``ir.convenience.replace_all_uses_with`` would
    rewrite every consumer in the graph, so we keep this scoped helper:

    * When ``old_v`` is present we rely on the IR helper (safe because the pass
      already knows all consumers that should be updated share that object).
    * When the pass only tracked the value name (older ONNX exports may expose
      strings or name copies), we manually scan the provided ``nodes`` and
      update inputs that match either the cached object or the cached name.
    * String-name rewrites also handle the case where ONNX IR stored a raw
      string in the input list.
    """
    if old_v is not None:
        ir.convenience.replace_all_uses_with(
            old_v,
            new_v,
        )
        if old_name is None:
            old_name = _v_name(old_v)
    new_name = _v_name(new_v)
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
                ins[i] = new_name if new_name is not None else ""
                changed = True
        if changed:
            _set_node_inputs(m, ins)
        if old_name and new_name and old_name != new_name:
            _propagate_value_name_to_subgraphs(m, old_name, new_name)


def _propagate_value_name_to_subgraphs(
    node: ir.Node, old_name: str, new_name: str
) -> None:
    """
    Ensure nested Graph/Graphs attributes no longer reference ``old_name``.

    When upstream rewrites replace a producer with a different value, we need
    to mirror that rename inside control-flow/function subgraphs.  Those
    subgraphs hold independent ``ir.Value`` clones that only stay connected to
    the parent graph via shared symbol names.
    """

    def _maybe_rename(val: Optional[ir.Value]) -> None:
        if val is None:
            return
        if not isinstance(val, ir.Value):
            return
        if _v_name(val) == old_name:
            try:
                val.name = new_name
            except Exception:
                pass

    def _walk_graph(graph: Optional[ir.Graph], seen: Set[int]) -> None:
        if graph is None:
            return
        gid = id(graph)
        if gid in seen:
            return
        seen.add(gid)

        for g_in in graph.inputs:
            _maybe_rename(g_in)
        for g_out in graph.outputs:
            _maybe_rename(g_out)

        init_container = graph.initializers
        init_values: Iterable[ir.Value]
        if isinstance(init_container, Mapping):
            init_values = init_container.values()
        elif isinstance(init_container, SequenceABC):
            init_values = init_container
        else:
            init_values = ()
        for init in init_values:
            if isinstance(init, ir.Value):
                _maybe_rename(init)

        for sub_node in list(graph):
            for iv in _node_inputs(sub_node):
                _maybe_rename(iv)
            for ov in _node_outputs(sub_node):
                _maybe_rename(ov)
            for attr in _iter_node_attrs(sub_node):
                kind = _attr_kind(attr)
                if kind == "GRAPH":
                    try:
                        sub_graph = attr.as_graph()
                    except Exception:
                        sub_graph = None
                    _walk_graph(sub_graph, seen)
                elif kind == "GRAPHS":
                    try:
                        sub_graphs = tuple(attr.as_graphs())
                    except Exception:
                        sub_graphs = ()
                    for sub_graph in sub_graphs:
                        _walk_graph(sub_graph, seen)

    seen_graphs: Set[int] = set()
    for attr in _iter_node_attrs(node):
        kind = _attr_kind(attr)
        if kind == "GRAPH":
            try:
                sub = attr.as_graph()
            except Exception:
                sub = None
            _walk_graph(sub, seen_graphs)
        elif kind == "GRAPHS":
            try:
                subs = tuple(attr.as_graphs())
            except Exception:
                subs = ()
            for sub in subs:
                _walk_graph(sub, seen_graphs)

    # Simplified graph walk that doesn't rely on _get_node_seq_and_setter
    def _walk_graph_simple(graph: Optional[ir.Graph], seen: Set[int]) -> None:
        if graph is None:
            return
        gid = id(graph)
        if gid in seen:
            return
        seen.add(gid)

        for g_in in graph.inputs:
            _maybe_rename(g_in)
        for g_out in graph.outputs:
            _maybe_rename(g_out)

        # Initializers
        init_container = graph.initializers
        if isinstance(init_container, Mapping):
            for init in init_container.values():
                if isinstance(init, ir.Value):
                    _maybe_rename(init)
        elif isinstance(init_container, SequenceABC):
            for init in init_container:
                if isinstance(init, ir.Value):
                    _maybe_rename(init)

        # Nodes
        for sub_node in graph:
            for iv in _node_inputs(sub_node):
                _maybe_rename(iv)
            for ov in _node_outputs(sub_node):
                _maybe_rename(ov)
            for attr in _iter_node_attrs(sub_node):
                kind = _attr_kind(attr)
                if kind == "GRAPH":
                    try:
                        sub_graph = attr.as_graph()
                    except Exception:
                        sub_graph = None
                    _walk_graph_simple(sub_graph, seen)
                elif kind == "GRAPHS":
                    try:
                        sub_graphs = tuple(attr.as_graphs())
                    except Exception:
                        sub_graphs = ()
                    for sub_graph in sub_graphs:
                        _walk_graph_simple(sub_graph, seen)


def _replace_in_graph_outputs(
    graph: ir.Graph,
    old_v: Optional[ir.Value],
    old_name: Optional[str],
    new_v: ir.Value,
) -> None:
    """
    Swap a graph output from ``old_v`` to ``new_v`` while keeping string-name fallbacks.

    Optimizer passes sometimes redirect the final node in a chain (e.g.
    collapsing Reshape→Reshape). If the old value fed a graph output we must
    update the graph outputs list; otherwise ONNX still sees the stale symbol.

    Parameters
    ----------
    graph:
        The graph whose outputs need updating.
    old_v:
        The value being replaced. When present we let the IR helper rewrite all
        consumers (graph outputs included).
    old_name:
        Fallback when only the value name is known. Some onnx_ir builds expose
        string inputs, so we still scan for matching names.
    new_v:
        The replacement value that should now feed the graph outputs.
    """
    graph_outputs = graph.outputs
    # Attempt IR-level replacement first
    graph_outputs = graph.outputs
    # Fallback: ensure graph outputs list is actually updated.
    # We DO NOT call replace_all_uses_with here because it is handled by _replace_everywhere
    # (or the global replacement pass), and calling it twice can cause ValueErrors if the
    # value is already gone from inputs.
    if old_v is not None:
        for idx, ov in enumerate(graph_outputs):
            if ov is old_v:
                graph_outputs[idx] = new_v

    # Name-based fallback (or if replace_all_uses_with missed the output list)
    if old_name:
        for idx, ov in enumerate(graph_outputs):
            if _v_name(ov) == old_name:
                graph_outputs[idx] = new_v


# ---------------- Attr access ----------------


def _get_attr(node: ir.Node, name: str) -> Optional[ir.Attr]:
    if not isinstance(node, ir.Node):
        return None
    # node.attributes is MutableMapping[str, Attr]
    attr = node.attributes.get(name)
    if isinstance(attr, ir.Attr):
        return attr
    return None


def _attr_to_int(attr: Optional[ir.Attr]) -> Optional[int]:
    if attr is None:
        return None
    if not isinstance(attr, ir.Attr):
        return None

    # Strict type checks
    if attr.type == IRAttrType.INT:
        return int(attr.as_int())

    if attr.type == IRAttrType.INTS:
        ints = tuple(attr.as_ints())
        if ints:
            return int(ints[0])

    # Fallbacks for non-strict legacy logic (e.g. TENSOR holding int scalar?)
    # "Prefer strong types" implies we rely on INT/INTS attribute types for integer intent.
    # However, value() might be used if type is not set?
    # But ir.Attr must have type.

    # Check value manually if type check failed or for robustness?
    # Existing code checked .value directly.
    # I will stick to type-based extraction.
    return None


def _collect_value_dtypes(graph: ir.Graph, nodes: Sequence[ir.Node]) -> Dict[str, int]:
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


def remove_redundant_casts_ir(graph: ir.Graph) -> None:
    nodes = list(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        dtype_map = _collect_value_dtypes(graph, nodes)
        for n in nodes:
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
                consumers = _consumer_nodes(nodes, out_val)
                if len(consumers) == 1:
                    next_node = consumers[0]
                    if next_node.op_type == "Cast":
                        next_outs = _node_outputs(next_node)
                        next_ins = _node_inputs(next_node)
                        if next_outs and next_ins:
                            next_target = _attr_to_int(_get_attr(next_node, "to"))
                            if next_target is not None and next_target == src_dtype:
                                final_out = next_outs[0]
                                src_val = ins[0]
                                _replace_in_graph_outputs(
                                    graph, final_out, _v_name(final_out), src_val
                                )
                                _replace_everywhere(
                                    nodes, final_out, _v_name(final_out), src_val
                                )
                                graph.remove([n, next_node])
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
            _replace_in_graph_outputs(graph, out_val, _v_name(out_val), src_val)
            _replace_everywhere(nodes, out_val, _v_name(out_val), src_val)
            graph.remove(n)
            changed = True
            break
    return


# ---------------- Transpose folding ----------------


def _transpose_perm(node: ir.Node) -> Optional[List[int]]:
    return _get_perm_attr(node)


def remove_redundant_transpose_pairs_ir(graph: ir.Graph) -> None:
    nodes = list(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if n.op_type != "Transpose":
                i += 1
                continue
            T1 = n
            T1_out = _node_output(T1)
            consumers = _consumer_nodes(nodes, T1_out)
            if len(consumers) != 1:
                i += 1
                continue
            chain_nodes: List[ir.Node] = [T1]
            allowed_nodes: List[ir.Node] = []
            cur = consumers[0]
            T2: Optional[ir.Node] = None
            steps = 0
            while steps < 8:
                steps += 1
                m = cur
                if m.op_type in ALLOWED_ELEMWISE:
                    chain_nodes.append(m)
                    allowed_nodes.append(m)
                    cur_val = _node_output(m)
                    next_nodes = _consumer_nodes(nodes, cur_val)
                    if len(next_nodes) != 1:
                        break
                    cur = next_nodes[0]
                    continue
                if m.op_type == "Transpose":
                    chain_nodes.append(m)
                    T2 = m
                break
            if T2 is None:
                i += 1
                continue
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
                    [node.op_type for node in chain_nodes],
                    "perm1",
                    perm1,
                    "perm2",
                    perm2,
                )
            t1_in = (_node_inputs(T1) or [None])[0]
            if t1_in is None:
                i += 1
                continue
            if allowed_nodes:
                first_allowed = allowed_nodes[0]
                last_allowed = allowed_nodes[-1]
                _replace_everywhere(
                    [first_allowed], _node_output(T1), _v_name(_node_output(T1)), t1_in
                )
                new_src = _node_output(last_allowed) or t1_in
            else:
                new_src = t1_in
            old_out = _node_output(T2)
            assert old_out is not None
            _replace_in_graph_outputs(graph, old_out, _v_name(old_out), new_src)
            _replace_everywhere(nodes, old_out, _v_name(old_out), new_src)
            graph.remove([T1, T2])
            changed = True
            break
    return


# ---------------- Reshape folding ----------------


def remove_redundant_reshape_pairs_ir(graph: ir.Graph) -> None:
    nodes = list(graph)
    if not nodes:
        return

    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        i = 0
        while i < len(nodes):
            T2 = nodes[i]
            if T2.op_type != "Reshape":
                i += 1
                continue
            v = (_node_inputs(T2) or [None])[0]
            allowed_nodes: List[ir.Node] = []
            T1: Optional[ir.Node] = None
            steps = 0
            while v is not None and steps < 8:
                steps += 1
                prod_node = _producer_node(nodes, v)
                if prod_node is None:
                    break
                if prod_node.op_type in ALLOWED_ELEMWISE:
                    allowed_nodes.append(prod_node)
                    v = (_node_inputs(prod_node) or [None])[0]
                    continue
                if prod_node.op_type == "Reshape":
                    T1 = prod_node
                break
            if T1 is None:
                i += 1
                continue
            src = (_node_inputs(T1) or [None])[0]
            dst = _node_output(T2)
            if not _shapes_compatible(src, dst):
                i += 1
                continue
            allowed_fwd = list(reversed(allowed_nodes))
            if RSH_DEBUG:
                print(
                    "[reshapefold/up]",
                    [n.op_type for n in ([T1] + allowed_fwd + [T2])],
                    "src",
                    _shape_tuple(src),
                    "dst",
                    _shape_tuple(dst),
                )
            if allowed_fwd:
                first_allowed = allowed_fwd[0]
                last_allowed = allowed_fwd[-1]
                _replace_everywhere(
                    [first_allowed], _node_output(T1), _v_name(_node_output(T1)), src
                )
                new_src = _node_output(last_allowed) or src
            else:
                new_src = src
            old_out = _node_output(T2)
            _replace_in_graph_outputs(graph, old_out, _v_name(old_out), new_src)
            _replace_everywhere(nodes, old_out, _v_name(old_out), new_src)
            graph.remove([T1, T2])
            changed = True
            break
    return


def _shapes_match_exact(
    src_dims: Optional[Tuple[object, ...]], target_dims: Tuple[int, ...]
) -> bool:
    if src_dims is None or len(src_dims) != len(target_dims):
        return False
    for src_dim, tgt_dim in zip(src_dims, target_dims):
        val = src_dim
        if hasattr(val, "value"):
            val = val.value
        if not isinstance(val, (int, np.integer)):
            return False
        if int(val) != int(tgt_dim):
            return False
    return True


def remove_identity_reshapes_ir(graph: ir.Graph) -> None:
    nodes = list(graph)
    if not nodes:
        return

    def _value_dims(val: Optional[ir.Value]) -> Optional[Tuple[object, ...]]:
        if val is None:
            return None
        return _shape_dims_seq(val.shape)

    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        for node in list(nodes):
            if node.op_type != "Reshape":
                continue
            ins = _node_inputs(node)
            outs = _node_outputs(node)
            if len(ins) < 2 or not outs:
                continue
            data_val = ins[0]
            shape_val = ins[1]
            target_dims = _value_const_ints(shape_val)
            if target_dims is None or not target_dims:
                continue
            if any(int(dim) in (-1, 0) for dim in target_dims):
                continue
            src_dims = _value_dims(data_val if isinstance(data_val, ir.Value) else None)
            if not _shapes_match_exact(src_dims, target_dims):
                continue
            dst_val = outs[0]
            dst_dims = _value_dims(dst_val)
            if dst_dims is not None and not _shapes_match_exact(dst_dims, target_dims):
                continue
            _replace_in_graph_outputs(graph, dst_val, _v_name(dst_val), data_val)
            _replace_everywhere(nodes, dst_val, _v_name(dst_val), data_val)
            graph.remove(node)
            changed = True
            break
    return


# ---------------- Shape propagation helpers ----------------


def _copy_shape_only(dst: Optional[ir.Value], src: Optional[ir.Value]) -> bool:
    """Copy shape metadata from src → dst when dst is missing/unknown."""
    if dst is None or src is None:
        return False

    if not isinstance(src, ir.Value) or not isinstance(dst, ir.Value):
        return False

    s_shp = src.shape
    if s_shp is None:
        return False

    d_shp = dst.shape

    s_key = _shape_dims_key(s_shp)
    d_key = _shape_dims_key(d_shp) if d_shp is not None else None

    if s_key is None:
        return False
    if d_key == s_key:
        return False

    cloned = ir.Shape(s_shp)
    if cloned is None:
        return False

    # We can assign shape directly
    dst.shape = cloned
    return True


def _copy_shape_dtype(dst: Optional[ir.Value], src: Optional[ir.Value]) -> bool:
    """
    Copy shape & dtype from src -> dst if present; return True if anything changed.
    """
    if dst is None or src is None:
        return False

    if not isinstance(src, ir.Value) or not isinstance(dst, ir.Value):
        return False

    changed = False
    s_shp = src.shape
    d_shp = dst.shape

    if s_shp is not None:
        s_key = _shape_dims_key(s_shp)
        d_key = _shape_dims_key(d_shp) if d_shp is not None else None
        if s_key is not None and s_key != d_key:
            cloned = ir.Shape(s_shp)
            if cloned is not None:
                dst.shape = cloned
                changed = True

    s_ty = src.type
    d_ty = dst.type

    if s_ty is not None and s_ty is not d_ty:
        dst.type = s_ty
        changed = True

    return changed


def propagate_unary_shapes_ir(graph: ir.Graph) -> None:
    """
    For known unary dataflow ops, set the first output's shape & dtype = first input's,
    when output metadata is missing/unknown. This helps preserve batch symbols across
    elementwise ops (e.g., BxN through Dropout/Gelu/etc.).
    """
    nodes = list(graph)
    if not nodes:
        return
    for n in nodes:
        op = n.op_type
        if op not in UNARY_DATAFLOW_OPS:
            continue
        ins = _node_inputs(n)
        outs = _node_outputs(n)
        if not ins or not outs:
            continue
        if op in {"Cast", "CastLike"}:
            _copy_shape_only(outs[0], ins[0])
            continue
        _copy_shape_dtype(outs[0], ins[0])
    return


# ---------------- Dropout.training_mode constant inlining ----------------


def _literal_bool_array(value: str) -> Optional[ArrayND]:
    normalized = value.strip().lower()
    if normalized == "true":
        return _as_ndarray(True, dtype=np.dtype(np.bool_))
    if normalized == "false":
        return _as_ndarray(False, dtype=np.dtype(np.bool_))
    return None


def _to_numpy_from_attr(attr: ir.Attr) -> Optional[ArrayND]:
    attr_type = attr.type

    if attr_type == IRAttrType.FLOAT:
        return _as_ndarray(attr.as_float())

    if attr_type == IRAttrType.FLOATS:
        return _as_ndarray(tuple(attr.as_floats()))

    if attr_type == IRAttrType.INT:
        return _as_ndarray(attr.as_int())

    if attr_type == IRAttrType.INTS:
        return _as_ndarray(tuple(attr.as_ints()))

    if attr_type == IRAttrType.STRING:
        string_value = attr.as_string()
        bool_arr = _literal_bool_array(string_value)
        if bool_arr is not None:
            return bool_arr
        return _as_ndarray(string_value)

    if attr_type == IRAttrType.STRINGS:
        strings = tuple(attr.as_strings())
        if len(strings) == 1:
            bool_arr = _literal_bool_array(strings[0])
            if bool_arr is not None:
                return bool_arr
        return _as_ndarray(strings)

    if attr_type == IRAttrType.TENSOR:
        return _to_numpy_from_any(attr.as_tensor())

    if attr_type == IRAttrType.TENSORS:
        tensors = tuple(attr.as_tensors())
        if not tensors:
            return None
        if len(tensors) == 1:
            return _to_numpy_from_any(tensors[0])

        arrays: list[ArrayND] = []
        for tensor in tensors:
            arr = _to_numpy_from_any(tensor)
            if arr is None:
                return None
            arrays.append(arr)
        try:
            return cast(ArrayND, np.stack(arrays))
        except Exception:
            return None

    value = attr.value
    if value is not None:
        return _to_numpy_from_any(value)
    return None


def _to_numpy_from_any(x: object) -> Optional[ArrayND]:
    """
    Convert common IR payload carriers (Values, Tensors, Attrs, numpy scalars) into
    numpy arrays.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return cast(ArrayND, x)
    if isinstance(x, np.generic):
        return _as_ndarray(x)

    # Handle primitive scalars strictly
    if isinstance(x, (bool, int, float, complex)):
        return _as_ndarray(x)

    if isinstance(x, str):
        # Allow string "true"/"false" parsing as special case for legacy boolean payloads
        bool_arr = _literal_bool_array(x)
        if bool_arr is not None:
            return bool_arr
        return _as_ndarray(x)

    if isinstance(x, ir.Value):
        return _to_numpy_from_any(x.const_value)

    if isinstance(x, ir.TensorProtocol):
        # Trust .numpy() from TensorProtocol
        try:
            return _as_ndarray(x.numpy())
        except Exception:
            return None

    if isinstance(x, ir.Attr):
        return _to_numpy_from_attr(x)

    if isinstance(x, SequenceABC) and not isinstance(x, (bytes, bytearray)):
        # Convert Sequences (list/tuple) to array
        try:
            # Tuple conversion is safer for numpy
            return _as_ndarray(tuple(x))
        except Exception:
            return None

    # Fallback for unexpected types?
    # Original code had a catch-all try/except block for _as_ndarray(x).
    # If we want strict typing, we should arguably RETURN NONE if it's not a known type.
    # But for robustness in converter, maybe keep fallback?
    # User said "prefer strong typing".
    # Strong typing means: if it's not one of the expected types, we don't know how to handle it.
    # Returning None is safer than crashing or guessing.

    # Check for legacy object-array-with-scalar case?
    # This was specific workaround.
    return None


def rewrite_mul_rsqrt_as_div_ir(graph: ir.Graph) -> None:
    """
    Rewrite Mul/Div(rsqrt) patterns into a single Div for cleaner graphs.
    """

    nodes = list(graph)
    if not nodes:
        return

    _dbg("rewrite_mul_rsqrt_as_div_ir start")

    # Disabled by default: parity drift was observed on some eqx_dino variants.
    # Re-enable via env flag when you explicitly want this rewrite.
    if os.getenv("JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT", "0") in ("", "0"):
        return

    def _is_scalar_one(val: Optional[ir.Value]) -> bool:
        arr = _to_numpy_from_any(val)
        if arr is None or arr.size != 1:
            return False
        try:
            return bool(np.isclose(arr.reshape(()), 1.0))
        except Exception:
            return False

    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        for node in nodes:
            if node.op_type != "Mul":
                continue
            mul_inputs = _node_inputs(node)
            if len(mul_inputs) != 2:
                continue
            for inv_pos in (0, 1):
                inv_val = mul_inputs[inv_pos]
                other_val = mul_inputs[1 - inv_pos]
                div_node = _producer_node(nodes, inv_val)
                if div_node is None:
                    if DEBUG:
                        _dbg("rewrite_mul_rsqrt skip: no producer", _v_name(inv_val))
                    continue
                if div_node.op_type != "Div":
                    if DEBUG:
                        _dbg(
                            "rewrite_mul_rsqrt skip: producer not Div", div_node.op_type
                        )
                    continue
                consumers = _consumer_nodes(nodes, inv_val)
                if consumers and any(consumer is not node for consumer in consumers):
                    if DEBUG:
                        _dbg("rewrite_mul_rsqrt skip: shared consumers", consumers)
                    continue
                div_inputs = _node_inputs(div_node)
                if len(div_inputs) != 2:
                    if DEBUG:
                        _dbg("rewrite_mul_rsqrt skip: div arity", len(div_inputs))
                    continue
                numerator, denominator = div_inputs
                if not _is_scalar_one(numerator):
                    if DEBUG:
                        _dbg("rewrite_mul_rsqrt skip: numerator not one")
                    continue
                denom_producer = _producer_node(nodes, denominator)
                if denom_producer is None:
                    if DEBUG:
                        _dbg("rewrite_mul_rsqrt skip: denominator producer missing")
                    continue
                if denom_producer.op_type != "Sqrt":
                    if DEBUG:
                        _dbg(
                            "rewrite_mul_rsqrt skip: denominator producer not Sqrt",
                            denom_producer.op_type,
                        )
                    continue
                if other_val is None or denominator is None:
                    if DEBUG:
                        _dbg("rewrite_mul_rsqrt skip: missing inputs")
                    continue
                node.op_type = "Div"
                _set_node_inputs(node, [other_val, denominator])
                if DEBUG:
                    _dbg("rewrite_mul_rsqrt_as_div", _v_name(_node_output(node)))
                changed = True
                break
            if changed:
                break
    return


def _as_scalar_bool(payload: object) -> Optional[bool]:
    if isinstance(payload, (bool, np.bool_)):
        return bool(payload)
    arr = _to_numpy_from_any(payload)
    if arr is None:
        return None
    try:
        return bool(arr.reshape(()).astype(np.bool_).item())
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

    producer = _producer_node(nodes, v_or_name)
    if producer is None:
        return None
    node = producer
    if node.op_type != "Constant":
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


def _constant_false_value() -> "ir.Value":
    return ir.Value(
        name="false_const",
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(()),
        const_value=_as_ndarray(False, dtype=np.dtype(np.bool_)),
    )


def inline_dropout_training_mode_constants_ir(graph: ir.Graph) -> None:
    """
    Constant-only inlining for Dropout.training_mode:
      - If training_mode is a constant False → drop it (make input #2 missing)
      - If training_mode is Not(True)        → drop it (make input #2 missing)
    This preserves dynamic (graph-input) cases: they are NOT inlined.
    """
    nodes = list(graph)
    if not nodes:
        return
    changed = False
    del_not_names: Set[str] = set()
    del_not_nodes: Set[ir.Node] = set()

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
        producer = _producer_node(nodes, tm)
        if producer is not None:
            # Find index for deletion tracking
            try:
                pidx = nodes.index(producer)
            except ValueError:
                pidx = -1

            if pidx != -1 and producer.op_type == "Not":
                _dbg_tm("tm producer is Not")
                not_in = (_node_inputs(producer) or [None])[0]
                if isinstance(not_in, ir.Value) and not_in.is_graph_input():
                    _dbg_tm("Not input is dynamic graph input; skipping")
                    _dbg_tm("Not input could not be proven True; nv=", None)
                    continue
                nv = _read_scalar_bool_from_value_or_constant(nodes, not_in)
                if nv is not None and bool(nv) is True:
                    rep_val = _constant_false_value()
                    ins_new = list(ins)
                    ins_new[2] = rep_val
                    old_not_out = _node_output(producer)
                    _replace_in_graph_outputs(
                        graph, old_not_out, _v_name(old_not_out), rep_val
                    )
                    _replace_everywhere(
                        nodes, old_not_out, _v_name(old_not_out), rep_val
                    )
                    _set_node_inputs(n, ins_new)
                    changed = True
                    out_v = _node_output(producer)
                    out_name = _v_name(out_v)
                    if out_name:
                        del_not_names.add(out_name)
                    del_not_nodes.add(producer)
                    # We need to mark producer index as deleted if we track by index
                    # But the current code uses del_not_nodes (set of nodes) and del_not_names.
                    # The original code used del_not_idx.
                    # We should probably stick to del_not_nodes which is safer.
                    continue
                _dbg_tm("Not input could not be proven True; nv=", nv)

        # Case A removed: we only inline Not(True) patterns to preserve
        # call-param wiring in inference graphs.
    if changed:
        _dbg_tm("changed detected; del_not_nodes=", len(del_not_nodes))
        # Remove any Not nodes whose outputs have no remaining consumers
        if del_not_nodes:
            final_del_nodes = []
            for not_node in del_not_nodes:
                has_uses = False
                for ov in _node_outputs(not_node):
                    if ov.uses() or ov.is_graph_output():
                        has_uses = True
                        break
                if not has_uses:
                    final_del_nodes.append(not_node)

            if final_del_nodes:
                if TRN_DEBUG or os.getenv("JAX2ONNX_TM_DEBUG"):
                    print(
                        f"[tm-inline] removed {len(final_del_nodes)} orphan Not nodes"
                    )
                graph.remove(final_del_nodes)


# ---------------- DCE ----------------


def remove_dead_nodes_ir(graph: ir.Graph) -> None:
    debug_metadata_flag = os.getenv("JAX2ONNX_ENABLE_STACKTRACE_METADATA", "")
    if debug_metadata_flag and debug_metadata_flag.strip().lower() not in (
        "0",
        "false",
        "off",
    ):
        # Keep the graph intact when stacktrace metadata is requested so downstream
        # tooling (e.g. sandbox repros) can inspect unused nodes.
        return

    # Iterative dead code elimination using IR APIs
    # Logic matched from ir-py/src/onnx_ir/passes/common/unused_removal.py
    graph_outputs = frozenset(graph.outputs)

    changed = True
    while changed:
        changed = False
        to_remove = []

        # Iterate reversed to handle chains of dead nodes in one pass (mostly)
        # We start with the full list snapshot
        nodes = list(graph)
        for node in reversed(nodes):
            is_used = False
            for output in node.outputs:
                if output in graph_outputs or output.uses():
                    is_used = True
                    break

            # If side-effecting nodes exist (e.g. Print), we might keep them.
            # But standard ONNX DCE assumes pure nodes.

            if not is_used:
                to_remove.append(node)

        if to_remove:
            if DCE_DEBUG:
                print(f"[dce] removing {len(to_remove)} nodes")
            graph.remove(to_remove)
            changed = True

    # Prune unused initializers
    # Refetch outputs/inputs as they might have changed or we need strict check
    used_values = set()
    for node in graph:
        for iv in node.inputs:
            used_values.add(iv)

    graph_outputs_set = set(graph.outputs)
    graph_inputs_set = set(graph.inputs)

    init_container = graph.initializers
    # graph.initializers is usually a Name -> Value mapping or similar container.
    # ir-py uses `list(initializers.values())`.

    if isinstance(init_container, MutableMapping):
        init_values = list(init_container.values())
        for init in init_values:
            if isinstance(init, ir.Value):
                if not (
                    init in used_values
                    or init in graph_outputs_set
                    or init in graph_inputs_set
                    or init.uses()
                ):
                    if init.name and init.name in init_container:
                        del init_container[init.name]


def ensure_unique_value_names_ir(graph: ir.Graph) -> None:
    """
    Ensure every ir.Value inside `graph` owns a unique non-empty name.
    ONNX requires value names to be globally unique per graph, and exporting
    invalid graphs causes ORT loads to fail. We fix collisions deterministically
    so downstream tooling always receives legal models.
    """

    seen: dict[str, int] = {}
    visited: set[int] = set()

    # Use direct iteration
    # graph is iterable (yields nodes)

    def _next_unique(name: str) -> str:
        counter = seen.get(name)
        if counter is None:
            seen[name] = 1
            return name
        candidate = f"{name}__{counter}"
        counter += 1
        while candidate in seen:
            candidate = f"{name}__{counter}"
            counter += 1
        seen[name] = counter
        seen[candidate] = 1
        return candidate

    def _rename_value(value: Optional[ir.Value]) -> None:
        if value is None or not isinstance(value, ir.Value):
            return
        vid = id(value)
        if vid in visited:
            return
        visited.add(vid)
        current = value.name
        if not current:
            return
        fresh = _next_unique(current)
        if fresh != current:
            value.name = fresh

    init_container = graph.initializers
    if isinstance(init_container, Mapping):
        init_values = list(init_container.values())
    else:
        init_values = list(init_container)
    for initializer in init_values:
        if isinstance(initializer, ir.Value):
            _rename_value(initializer)
    for g_input in graph.inputs:
        _rename_value(g_input)
    for node in graph:

        if node is None:
            continue
        for out_val in _node_outputs(node):
            _rename_value(out_val)
        for attr in _iter_node_attrs(node):
            kind = _attr_kind(attr)
            if kind == "GRAPH":
                sub_graph = None
                if isinstance(attr, ir.Attr):
                    try:
                        sub_graph = attr.as_graph()
                    except Exception:
                        sub_graph = None
                if sub_graph is not None:
                    ensure_unique_value_names_ir(sub_graph)
            elif kind == "GRAPHS":
                graphs: tuple[ir.Graph, ...] = ()
                if isinstance(attr, ir.Attr):
                    try:
                        graphs = tuple(attr.as_graphs())
                    except Exception:
                        graphs = ()
                for sub in graphs:
                    if sub is not None:
                        ensure_unique_value_names_ir(sub)
    for g_output in graph.outputs:

        _rename_value(g_output)


# ---------------- Prune unused graph inputs (top graph only) ----------------


def _attr_kind(attr: Optional[ir.Attr]) -> Optional[str]:
    if attr is None:
        return None
    if not isinstance(attr, ir.Attr):
        return None

    atype = attr.type
    if isinstance(atype, IRAttrType):
        return str(atype.name)
    if isinstance(atype, str):
        return atype.upper()
    return None


def _iter_node_attrs(node: ir.Node) -> Iterable[ir.Attr]:
    attrs = node.attributes
    if isinstance(attrs, Mapping):
        return [attr for attr in attrs.values() if isinstance(attr, ir.Attr)]
    if isinstance(attrs, SequenceABC):
        return [attr for attr in attrs if isinstance(attr, ir.Attr)]
    return []


def prune_unused_graph_inputs_ir(graph: ir.Graph) -> None:
    """
    Remove graph inputs that are not consumed by any node and are not graph outputs.
    (We do NOT run this on function bodies to avoid changing function signatures.)
    """

    def _should_always_keep(name: Optional[str]) -> bool:
        if not name:
            return True
        # Preserve positional graph inputs that correspond to original JAX
        # function arguments (named ``in_<index>`` by IRContext.add_input_for_invar).
        if name.startswith("in_"):
            suffix = name[3:]
            if suffix.isdigit():
                return True
        return False

    original_inputs = list(graph.inputs)
    keep: List[ir.Value] = []
    removed: List[str] = []
    for value in original_inputs:
        if _should_always_keep(value.name):
            keep.append(value)
        elif value.uses():
            keep.append(value)
        elif value.is_graph_output():
            # A graph input cannot be a direct graph output. But for the sake of completeness
            keep.append(value)
        else:
            removed.append(value.name)

    if removed and DEBUG:
        _dbg(f"prune_unused_graph_inputs_ir removed: {removed}")

    if keep != original_inputs:
        graph.inputs.clear()
        graph.inputs.extend(keep)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# ---------------- CSE ----------------


# ---------------- Constant Lifting ----------------


def lift_constants_to_initializers_ir(graph: ir.Graph) -> None:
    """
    Lift Constant nodes into graph Initializers.
    This simplifies the graph structure by removing nodes that just produce static data.
    """
    nodes = list(graph)
    if not nodes:
        return

    for node in nodes:
        if node.op_type != "Constant":
            continue

        if not isinstance(node, ir.Node):
            continue

        out_val = _node_output(node)
        if out_val is None:
            continue

        if out_val.is_graph_output():
            continue

        tensor: Optional[ir.TensorProtocol] = None

        # Check standard attributes in order of precedence/likelihood
        attrs = node.attributes

        # 1. TENSOR value
        if "value" in attrs:
            attr = attrs["value"]
            if isinstance(attr, ir.Attr) and attr.type == IRAttrType.TENSOR:
                if isinstance(attr.value, ir.TensorProtocol):
                    tensor = attr.value

        # 2. Scalar Float
        elif "value_float" in attrs:
            attr = attrs["value_float"]
            if isinstance(attr, ir.Attr) and attr.type == IRAttrType.FLOAT:
                # attr.value is standard float
                if isinstance(attr.value, float):
                    tensor = ir.tensor(attr.value, dtype=ir.DataType.FLOAT)

        # 3. Scalar Int
        elif "value_int" in attrs:
            attr = attrs["value_int"]
            if isinstance(attr, ir.Attr) and attr.type == IRAttrType.INT:
                # attr.value is standard int
                if isinstance(attr.value, int):
                    tensor = ir.tensor(attr.value, dtype=ir.DataType.INT64)

        # 4. Floats (1D)
        elif "value_floats" in attrs:
            attr = attrs["value_floats"]
            if isinstance(attr, ir.Attr) and attr.type == IRAttrType.FLOATS:
                if isinstance(
                    attr.value, tuple
                ):  # verified as tuple of floats in Attr.__init__
                    tensor = ir.tensor(attr.value, dtype=ir.DataType.FLOAT)

        # 5. Ints (1D)
        elif "value_ints" in attrs:
            attr = attrs["value_ints"]
            if isinstance(attr, ir.Attr) and attr.type == IRAttrType.INTS:
                if isinstance(attr.value, tuple):  # tuple of ints
                    tensor = ir.tensor(attr.value, dtype=ir.DataType.INT64)

        if tensor is None:
            continue

        # Create a new Initializer Value
        # We need to preserve the name for the downstream users.
        name = out_val.name

        # New initializer value
        # Note: In onnx-ir, we can have a value that is just an initializer.
        init_val = ir.Value(
            name=name,
            type=ir.TensorType(tensor.dtype),
            shape=ir.Shape(tensor.shape),
            const_value=tensor,
        )

        # Register it
        graph.register_initializer(init_val)

        # Replace uses
        # However, out_val MUST NOT be a graph output for this pass (checked above).
        ir.convenience.replace_all_uses_with(out_val, init_val)

        # Remove the node
        graph.remove(node)


def optimize_graph(ir_model: ir.Model) -> ir.Model:
    _dbg("optimize_graph invoked")
    # Top graph
    try:
        gr = ir_model.graph
        ensure_unique_value_names_ir(gr)
        remove_redundant_casts_ir(gr)
        remove_redundant_transpose_pairs_ir(gr)
        remove_redundant_reshape_pairs_ir(gr)
        remove_identity_reshapes_ir(gr)
        common_passes.CommonSubexpressionEliminationPass()(ir_model)
        lift_constants_to_initializers_ir(gr)
        rewrite_mul_rsqrt_as_div_ir(gr)
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
                ensure_unique_value_names_ir(fgr)
                remove_redundant_casts_ir(fgr)
                remove_redundant_transpose_pairs_ir(fgr)
                remove_redundant_reshape_pairs_ir(fgr)
                remove_identity_reshapes_ir(fgr)
                rewrite_mul_rsqrt_as_div_ir(fgr)
                inline_dropout_training_mode_constants_ir(fgr)
                propagate_unary_shapes_ir(fgr)
                remove_dead_nodes_ir(fgr)
                remove_redundant_casts_ir(fgr)
            except Exception as _fe:
                _dbg("optimize_graph: function pass skipped:", _fe)
    except Exception as _e:
        _dbg("optimize_graph: functions traversal skipped:", _e)

    return ir_model
