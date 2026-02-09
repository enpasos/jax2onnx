# jax2onnx/converter/ir_optimizations.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Set,
    Any,
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
    "max",
    "min",
    "clip",
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
    "Max",
    "Min",
    "Clip",
}

# Elementwise ops that are layout-invariant (used for transpose folding)
ELEMENTWISE_UNARY_OPS: Set[str] = {
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
    "Abs",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
}

ELEMENTWISE_BINARY_OPS: Set[str] = {
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Max",
    "Min",
    "Clip",
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


# ---------------- Replacement helpers ----------------


def _replace_all_uses_with(
    values: Union[ir.Value, Sequence[ir.Value]],
    replacements: Union[ir.Value, Sequence[ir.Value]],
    *,
    replace_graph_outputs: bool = False,
) -> None:
    try:
        ir.convenience.replace_all_uses_with(
            values,
            replacements,
            replace_graph_outputs=replace_graph_outputs,
        )
        return
    except TypeError:
        # Compatibility with older onnx_ir that lacks replace_graph_outputs.
        ir.convenience.replace_all_uses_with(values, replacements)


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
    current_node_ids: Set[int] = {id(node) for node in nodes}
    if isinstance(value_or_name, ir.Value):
        consumers = value_or_name.consumers()
        if consumers:
            try:
                if all(isinstance(c, ir.Node) for c in consumers):
                    filtered = [c for c in consumers if id(c) in current_node_ids]
                    if filtered:
                        return list(filtered)
            except Exception:
                # Fall back to name-based scan below.
                pass

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
    current_node_ids: Set[int] = {id(node) for node in nodes}
    if isinstance(value_or_name, ir.Value):
        prod = value_or_name.producer()
        if isinstance(prod, ir.Node) and id(prod) in current_node_ids:
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
    ta, tb = _shape_tuple(a), _shape_tuple(b)
    if ta is None or tb is None or len(ta) != len(tb):
        return False
    for da, db in zip(ta, tb):
        if da == -1 or db == -1:
            continue
        if da != db:
            return False
    return True


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
                                _replace_all_uses_with(
                                    final_out, src_val, replace_graph_outputs=True
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
            _replace_all_uses_with(out_val, src_val, replace_graph_outputs=True)
            graph.remove(n)
            changed = True
            break
    return


# ---------------- Transpose folding ----------------


def _transpose_perm(node: ir.Node) -> Optional[List[int]]:
    return _get_perm_attr(node)


def _is_inverse_perm(perm1: Sequence[int], perm2: Sequence[int]) -> bool:
    if len(perm1) != len(perm2):
        return False
    composed = [perm1[p] for p in perm2]
    return composed == list(range(len(composed)))


def _is_scalar_const_value(val: Optional[ir.Value]) -> bool:
    if not isinstance(val, ir.Value):
        return False
    if val.const_value is None and not val.is_initializer():
        return False
    arr = _to_numpy_from_any(val.const_value)
    if arr is not None:
        return bool(arr.size == 1)
    dims = _shape_dims_seq(val.shape)
    if dims is None:
        return False
    for d in dims:
        if not isinstance(d, (int, np.integer)):
            return False
        if int(d) != 1:
            return False
    return True


def _is_elementwise_node(node: ir.Node) -> bool:
    return (
        node.op_type in ELEMENTWISE_UNARY_OPS or node.op_type in ELEMENTWISE_BINARY_OPS
    )


def _elementwise_shape_source(inputs: Sequence[ir.Value]) -> Optional[ir.Value]:
    for iv in inputs:
        if not _is_scalar_const_value(iv):
            return iv
    return inputs[0] if inputs else None


def _dim_token(dim: Any) -> Tuple[str, Any]:
    if isinstance(dim, (int, np.integer)):
        return ("int", int(dim))
    if hasattr(dim, "value"):
        try:
            return ("value", getattr(dim, "value"))
        except Exception:
            pass
    return ("repr", repr(dim))


def _broadcast_shape_dims(
    shapes: Sequence[Tuple[Any, ...]],
) -> Optional[Tuple[Any, ...]]:
    if not shapes:
        return None

    max_rank = max(len(shape) for shape in shapes)
    if max_rank == 0:
        return ()

    padded_shapes: List[Tuple[Any, ...]] = []
    for shape in shapes:
        if len(shape) < max_rank:
            shape = (1,) * (max_rank - len(shape)) + shape
        padded_shapes.append(shape)

    result: List[Any] = []
    for axis in range(max_rank):
        resolved: Any = 1
        for shape in padded_shapes:
            dim = shape[axis]
            if isinstance(dim, (int, np.integer)):
                dim_int = int(dim)
                if dim_int == 1:
                    continue
                if isinstance(resolved, (int, np.integer)):
                    resolved_int = int(resolved)
                    if resolved_int == 1:
                        resolved = dim_int
                        continue
                    if resolved_int != dim_int:
                        return None
                    continue
                # Keep concrete dimensions over symbolic ones when available.
                resolved = dim_int
                continue

            # Symbolic/unknown dimension.
            if isinstance(resolved, (int, np.integer)):
                if int(resolved) == 1:
                    resolved = dim
                # If resolved is a concrete non-1 dim, keep it.
                continue
            if _dim_token(resolved) != _dim_token(dim):
                return None
        result.append(int(resolved) if isinstance(resolved, np.integer) else resolved)

    return tuple(result)


def _refresh_elementwise_output_shape(node: ir.Node) -> None:
    outs = _node_outputs(node)
    if not outs:
        return
    ins = _node_inputs(node)
    src = _elementwise_shape_source(ins)
    if src is None:
        return
    if node.op_type in {"Cast", "CastLike", "Not"}:
        # These ops can change dtype; keep existing dtype metadata untouched.
        _copy_shape_only(outs[0], src)
    else:
        _copy_shape_dtype(outs[0], src)
    candidate_shapes: List[Tuple[Any, ...]] = []
    for iv in ins:
        if _is_scalar_const_value(iv):
            continue
        dims = _shape_dims_seq(iv.shape)
        if dims is None:
            continue
        candidate_shapes.append(dims)
    merged = _broadcast_shape_dims(candidate_shapes)
    if merged is None:
        return
    out = outs[0]
    if _shape_dims_key(out.shape) == _shape_dims_key(merged):
        return
    out.shape = ir.Shape(merged)


def _collect_transpose_elementwise_chain(
    nodes: Sequence[ir.Node],
    start_value: ir.Value,
) -> Optional[Tuple[ir.Node, Set[ir.Node]]]:
    """
    Trace elementwise-only producers from start_value back to a single Transpose.
    Returns (source_transpose, elementwise_nodes) if successful.
    """
    allowed_nodes: Set[ir.Node] = set()
    visited_values: Set[ir.Value] = set()
    worklist: List[ir.Value] = [start_value]
    source_transpose: Optional[ir.Node] = None

    while worklist:
        val = worklist.pop()
        if val in visited_values:
            continue
        visited_values.add(val)
        if _is_scalar_const_value(val):
            continue
        producer = _producer_node(nodes, val)
        if producer is None:
            return None
        if producer.op_type == "Transpose":
            if source_transpose is None:
                source_transpose = producer
            elif source_transpose is not producer:
                return None
            continue
        if not _is_elementwise_node(producer):
            return None
        if producer not in allowed_nodes:
            allowed_nodes.add(producer)
            for iv in _node_inputs(producer):
                if _is_scalar_const_value(iv):
                    continue
                worklist.append(iv)

    if source_transpose is None:
        return None
    return source_transpose, allowed_nodes


def _collect_transpose_elementwise_forest(
    nodes: Sequence[ir.Node],
    start_value: ir.Value,
) -> Optional[Tuple[Set[ir.Node], Set[ir.Node]]]:
    """
    Trace elementwise-only producers from start_value back to one or more Transpose nodes.
    Returns (transpose_nodes, elementwise_nodes) if successful.
    """
    elementwise_nodes: Set[ir.Node] = set()
    transpose_nodes: Set[ir.Node] = set()
    visited_values: Set[ir.Value] = set()
    worklist: List[ir.Value] = [start_value]

    while worklist:
        val = worklist.pop()
        if val in visited_values:
            continue
        visited_values.add(val)
        if _is_scalar_const_value(val):
            continue
        producer = _producer_node(nodes, val)
        if producer is None:
            return None
        if producer.op_type == "Transpose":
            transpose_nodes.add(producer)
            continue
        if not _is_elementwise_node(producer):
            return None
        if producer not in elementwise_nodes:
            elementwise_nodes.add(producer)
            for iv in _node_inputs(producer):
                if _is_scalar_const_value(iv):
                    continue
                worklist.append(iv)

    if not transpose_nodes:
        return None
    return transpose_nodes, elementwise_nodes


def remove_redundant_transpose_reduce_ir(graph: ir.Graph) -> None:
    nodes = list(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        for node in nodes:
            if node.op_type != "Transpose":
                continue

            # Pattern: T1 -> Reduce -> T2 (node)
            ins = _node_inputs(node)
            if len(ins) != 1:
                continue

            reducer = _producer_node(nodes, ins[0])
            if reducer is None or reducer.op_type != "ReduceMean":
                continue

            reducer_ins = _node_inputs(reducer)
            if not reducer_ins:
                continue

            t1 = _producer_node(nodes, reducer_ins[0])
            if t1 is None or t1.op_type != "Transpose":
                continue

            perm2 = _transpose_perm(node)
            perm1 = _transpose_perm(t1)

            if perm1 is None or perm2 is None:
                continue

            if not _is_inverse_perm(perm1, perm2):
                continue

            keepdims = _attr_to_int(_get_attr(reducer, "keepdims"))
            if keepdims != 1:
                continue

            # Handle axes
            axes_val = None
            axes_input_idx = -1
            axes = None

            # Check inputs for axes (opset 13+)
            if len(reducer_ins) > 1:
                axes_val = reducer_ins[1]
                axes = _value_const_ints(axes_val)
                if axes is None:
                    # Fallback: check if produced by Constant node
                    prod = axes_val.producer()
                    if prod is not None and prod.op_type == "Constant":
                        attr = prod.attributes.get("value")
                        if attr is not None and attr.type == IRAttrType.TENSOR:
                            try:
                                arr = attr.as_tensor().numpy()
                                if arr is not None and np.issubdtype(
                                    arr.dtype, np.integer
                                ):
                                    axes = tuple(arr.flatten().tolist())
                            except Exception:
                                pass

                if axes is None:
                    # Dynamic axes or unable to resolve constant - Abort
                    if DEBUG:
                        _dbg("  Skip: dynamic/unresolvable axes")
                    continue
                axes_input_idx = 1

            # Check attribute if not in input
            if axes is None:
                axes_attr = _get_attr(reducer, "axes")
                if axes_attr is not None and axes_attr.type == IRAttrType.INTS:
                    axes = tuple(axes_attr.as_ints())

            # Normalize and map axes
            rank = len(perm1)
            if axes is not None:
                new_axes_list: List[int] = []
                valid_axes = True
                for a in axes:
                    if a < 0:
                        a += rank
                    if a < 0 or a >= rank:
                        # Invalid axis for rank? Skip optimization to be safe
                        valid_axes = False
                        break
                    new_axes_list.append(perm1[a])

                if not valid_axes:
                    continue
                new_axes = tuple(sorted(new_axes_list))
            else:
                # Reduce all
                new_axes = None

                # PERFORM REWRITE
                _dbg(
                    "Remove redundant Transpose-Reduce-Transpose:",
                    _v_name(node),
                    "around",
                    _v_name(reducer),
                )

            # 0. Safety: Reducer output must NOT be used by anything else
            # (Because we are about to change its semantic output from NHWC to NCHW)
            reducer_out_val = _node_output(reducer)
            reducer_consumers = _consumer_nodes(nodes, reducer_out_val)
            # We expect exactly one consumer: 'node' (T2)
            if len(reducer_consumers) != 1:
                if DEBUG:
                    _dbg(
                        "  Skip: reducer has multiple consumers",
                        [c.name for c in reducer_consumers],
                    )
                continue
            if reducer_consumers[0] is not node:
                # Should be covered by consumers scan logic, but double check
                continue

            # 1. Update Reducer inputs
            # Input 0 becomes T1 input 0
            t1_input = _node_inputs(t1)[0]
            reducer.replace_input_with(0, t1_input)

            # 2. Update Axes
            if new_axes is not None:
                if axes_input_idx != -1:
                    new_axes_arr = np.array(new_axes, dtype=np.int64)
                    new_axes_val = ir.Value(
                        name=f"{reducer.name or 'reduce'}_axes_optimized",
                        shape=ir.Shape((len(new_axes),)),
                        type=ir.TensorType(ir.DataType.INT64),
                    )
                    new_axes_val.const_value = ir.tensor(new_axes_arr)

                    # Register as initializer
                    graph.initializers.add(new_axes_val)

                    reducer.replace_input_with(axes_input_idx, new_axes_val)
                else:
                    reducer.attributes["axes"] = ir.Attr(
                        "axes", IRAttrType.INTS, list(new_axes)
                    )

            # 3. Bypass T2
            t2_out = _node_output(node)
            reducer_out = _node_output(reducer)
            _copy_shape_dtype(reducer_out, t2_out)
            _replace_all_uses_with(t2_out, reducer_out, replace_graph_outputs=True)

            # 4. Cleanup
            # node.inputs might be tuple? Just remove from graph.
            graph.remove(node)

            changed = True
            break


def _collect_add_transpose_forest(
    nodes: Sequence[ir.Node], start: ir.Node
) -> Optional[Tuple[List[ir.Node], List[int], List[int], Set[ir.Node], Set[ir.Node]]]:
    """
    Collect a forward Add forest where external inputs are wrapped by one
    Transpose(perm_fwd) and external outputs are wrapped by one
    Transpose(perm_inv). Returns None when the pattern does not match.
    """
    if start.op_type != "Add":
        return None

    perm_fwd: Optional[List[int]] = None
    perm_inv: Optional[List[int]] = None
    add_nodes: List[ir.Node] = []
    add_set: Set[ir.Node] = set()
    input_transposes: Set[ir.Node] = set()
    output_transposes: Set[ir.Node] = set()
    queue: List[ir.Node] = [start]

    while queue:
        node = queue.pop(0)
        if node in add_set:
            continue
        if node.op_type != "Add":
            return None

        ins = _node_inputs(node)
        if len(ins) < 2:
            return None

        add_input_count = 0
        transpose_input_count = 0
        for iv in ins:
            prod = _producer_node(nodes, iv)
            if prod is not None and prod.op_type == "Add":
                add_input_count += 1
                if prod not in add_set:
                    # Keep traversal strictly forward from the selected root.
                    return None
                continue
            if prod is None or prod.op_type != "Transpose":
                return None
            perm = _transpose_perm(prod)
            if perm is None:
                return None
            if perm_fwd is None:
                perm_fwd = perm
            elif perm_fwd != perm:
                return None
            input_transposes.add(prod)
            transpose_input_count += 1

        if node is start:
            if add_input_count != 0 or transpose_input_count == 0:
                return None
        else:
            if add_input_count == 0:
                return None

        out = _node_output(node)
        if out is None:
            return None
        consumers = _consumer_nodes(nodes, out)
        for consumer in consumers:
            if consumer.op_type == "Add":
                if consumer not in add_set:
                    queue.append(consumer)
                continue
            if consumer.op_type != "Transpose":
                return None
            perm = _transpose_perm(consumer)
            if perm is None:
                return None
            if perm_inv is None:
                perm_inv = perm
            elif perm_inv != perm:
                return None
            output_transposes.add(consumer)

        add_set.add(node)
        add_nodes.append(node)

    if (
        not add_nodes
        or perm_fwd is None
        or perm_inv is None
        or not output_transposes
        or not _is_inverse_perm(perm_fwd, perm_inv)
    ):
        return None
    return add_nodes, perm_fwd, perm_inv, input_transposes, output_transposes


def remove_redundant_transpose_add_forests_ir(graph: ir.Graph) -> None:
    """
    Lift Add forests to pre-transpose layout when they are enclosed by inverse
    transpose boundaries.
    """
    nodes = list(cast(NodeSeq, graph))
    if not nodes:
        return

    changed = True
    while changed:
        changed = False
        nodes = list(cast(NodeSeq, graph))
        for start in nodes:
            if start.op_type != "Add":
                continue

            match = _collect_add_transpose_forest(nodes, start)
            if match is None:
                continue
            add_nodes, perm_fwd, _perm_inv, input_transposes, output_transposes = match

            # Rewrite Add inputs from Transpose(perm_fwd)(x) to x.
            for add_node in add_nodes:
                ins = _node_inputs(add_node)
                for idx, iv in enumerate(ins):
                    prod = _producer_node(nodes, iv)
                    if prod is None or prod.op_type != "Transpose":
                        continue
                    perm = _transpose_perm(prod)
                    if perm is None or perm != perm_fwd:
                        continue
                    src = (_node_inputs(prod) or [None])[0]
                    if isinstance(src, ir.Value):
                        add_node.replace_input_with(idx, src)
                _refresh_elementwise_output_shape(add_node)

            # Remove output Transpose(perm_inv) wrappers.
            for out_transpose in output_transposes:
                t_out = _node_output(out_transpose)
                t_in = (_node_inputs(out_transpose) or [None])[0]
                if t_out is None or not isinstance(t_in, ir.Value):
                    continue
                _replace_all_uses_with(t_out, t_in, replace_graph_outputs=True)
            graph.remove(list(output_transposes))

            # Remove now-unused input Transpose(perm_fwd) nodes.
            live_nodes = list(cast(NodeSeq, graph))
            removable_inputs: List[ir.Node] = []
            for in_transpose in input_transposes:
                t_out = _node_output(in_transpose)
                if t_out is None:
                    removable_inputs.append(in_transpose)
                    continue
                if _consumer_nodes(live_nodes, t_out):
                    continue
                if t_out.is_graph_output():
                    continue
                removable_inputs.append(in_transpose)
            if removable_inputs:
                graph.remove(removable_inputs)

            if TRN_DEBUG:
                _dbg(
                    "remove_redundant_transpose_add_forests_ir:",
                    "start=",
                    start.name,
                    "adds=",
                    len(add_nodes),
                    "in_t=",
                    len(input_transposes),
                    "out_t=",
                    len(output_transposes),
                )

            changed = True
            break


def remove_redundant_transpose_pairs_ir(graph: ir.Graph) -> None:
    nodes = list(graph)
    if not nodes:
        return
    changed = True
    while changed:
        changed = False
        nodes = list(graph)
        # Pass -1: collapse Add chains surrounded by transposes (NHWC <-> NCHW)
        visited_adds: Set[ir.Node] = set()
        for start in nodes:
            if start.op_type != "Add":
                continue
            if start in visited_adds:
                continue

            add_chain: List[ir.Node] = []
            perm_fwd: Optional[List[int]] = None
            perm_inv: Optional[List[int]] = None
            ok = True

            prev: Optional[ir.Node] = None
            cur: Optional[ir.Node] = start
            while cur is not None:
                if cur.op_type != "Add":
                    ok = False
                    break
                ins = _node_inputs(cur)
                if len(ins) < 2:
                    ok = False
                    break

                transpose_inputs: List[Tuple[ir.Node, ir.Value]] = []
                has_prev_input = False
                for iv in ins:
                    prod = _producer_node(nodes, iv)
                    if prod is not None and prod is prev:
                        has_prev_input = True
                        continue
                    if prod is None or prod.op_type != "Transpose":
                        ok = False
                        break
                    perm = _transpose_perm(prod)
                    if perm is None:
                        ok = False
                        break
                    if perm_fwd is None:
                        perm_fwd = perm
                    elif perm_fwd != perm:
                        ok = False
                        break
                    transpose_inputs.append((prod, iv))
                if not ok:
                    break

                if prev is None:
                    if len(transpose_inputs) < 1:
                        ok = False
                        break
                else:
                    if not has_prev_input or len(transpose_inputs) != 1:
                        ok = False
                        break

                out = _node_output(cur)
                if out is None:
                    ok = False
                    break
                consumers = _consumer_nodes(nodes, out)
                add_consumers = [c for c in consumers if c.op_type == "Add"]
                other_consumers = [c for c in consumers if c.op_type != "Add"]
                if len(add_consumers) > 1:
                    ok = False
                    break
                for consumer in other_consumers:
                    if consumer.op_type != "Transpose":
                        ok = False
                        break
                    perm = _transpose_perm(consumer)
                    if perm is None:
                        ok = False
                        break
                    if perm_inv is None:
                        perm_inv = perm
                    elif perm_inv != perm:
                        ok = False
                        break
                if not ok:
                    break

                add_chain.append(cur)
                if add_consumers:
                    prev = cur
                    cur = add_consumers[0]
                    continue
                break

            if (
                not ok
                or not add_chain
                or perm_fwd is None
                or perm_inv is None
                or not _is_inverse_perm(perm_fwd, perm_inv)
            ):
                continue

            # Rewrite: move Add chain to pre-transpose layout (NCHW).
            for node in add_chain:
                ins = _node_inputs(node)
                for idx, iv in enumerate(ins):
                    prod = _producer_node(nodes, iv)
                    if prod is None or prod.op_type != "Transpose":
                        continue
                    perm = _transpose_perm(prod)
                    if perm is None or perm != perm_fwd:
                        continue
                    src = (_node_inputs(prod) or [None])[0]
                    if src is None:
                        continue
                    node.replace_input_with(idx, src)
                _refresh_elementwise_output_shape(node)

            # Remove inverse transposes on outputs of the chain.
            to_remove: Set[ir.Node] = set()
            for node in add_chain:
                out = _node_output(node)
                if out is None:
                    continue
                for consumer in _consumer_nodes(nodes, out):
                    if consumer.op_type != "Transpose":
                        continue
                    perm = _transpose_perm(consumer)
                    if perm is None or perm != perm_inv:
                        continue
                    consumer_out = _node_output(consumer)
                    if consumer_out is None:
                        continue
                    _replace_all_uses_with(
                        consumer_out, out, replace_graph_outputs=True
                    )
                    to_remove.add(consumer)

            if to_remove:
                graph.remove(list(to_remove))
                changed = True
                visited_adds.update(add_chain)
                break
        if changed:
            continue
        # Pass -0.5: fold inverse transpose around elementwise DAG with
        # multiple transpose inputs (e.g., scale + residual add).
        for t2_node in nodes:
            if t2_node.op_type != "Transpose":
                continue
            t2_in = (_node_inputs(t2_node) or [None])[0]
            if not isinstance(t2_in, ir.Value):
                continue
            perm2 = _transpose_perm(t2_node)
            if perm2 is None:
                continue
            forest_match = _collect_transpose_elementwise_forest(nodes, t2_in)
            if forest_match is None:
                continue
            transpose_nodes, elem_nodes = forest_match
            perm1: Optional[List[int]] = None
            ok = True
            for t_node in transpose_nodes:
                perm = _transpose_perm(t_node)
                if perm is None:
                    ok = False
                    break
                if perm1 is None:
                    perm1 = perm
                elif perm1 != perm:
                    ok = False
                    break
            if not ok or perm1 is None or not _is_inverse_perm(perm1, perm2):
                continue

            # Ensure elementwise outputs only feed the subgraph or inverse transposes.
            output_transposes: Set[ir.Node] = set()
            for node in elem_nodes:
                out = _node_output(node)
                if out is None:
                    continue
                for consumer in _consumer_nodes(nodes, out):
                    if consumer in elem_nodes:
                        continue
                    if consumer.op_type != "Transpose":
                        ok = False
                        break
                    perm = _transpose_perm(consumer)
                    if perm is None or perm != perm2:
                        ok = False
                        break
                    output_transposes.add(consumer)
                if not ok:
                    break
            if not ok:
                continue
            if t2_node not in output_transposes:
                continue

            # Rewrite: replace transpose outputs feeding elementwise nodes with
            # their pre-transpose sources.
            trans_in_map: Dict[ir.Value, ir.Value] = {}
            for t_node in transpose_nodes:
                t_out = _node_output(t_node)
                t_src = (_node_inputs(t_node) or [None])[0]
                if isinstance(t_out, ir.Value) and isinstance(t_src, ir.Value):
                    trans_in_map[t_out] = t_src

            for node in elem_nodes:
                ins = _node_inputs(node)
                for idx, iv in enumerate(ins):
                    if iv in trans_in_map:
                        node.replace_input_with(idx, trans_in_map[iv])
                _refresh_elementwise_output_shape(node)

            # Remove inverse transposes on outputs of the DAG.
            for t_out_node in output_transposes:
                t_out = _node_output(t_out_node)
                if t_out is None:
                    continue
                t_in = (_node_inputs(t_out_node) or [None])[0]
                if not isinstance(t_in, ir.Value):
                    continue
                _replace_all_uses_with(t_out, t_in, replace_graph_outputs=True)
            if output_transposes:
                graph.remove(list(output_transposes))

            # Remove now-unused transpose inputs.
            live_nodes = list(graph)
            for t_node in transpose_nodes:
                t_out = _node_output(t_node)
                if t_out is None:
                    continue
                if not _consumer_nodes(live_nodes, t_out):
                    graph.remove(t_node)

            changed = True
            break
        if changed:
            continue
        # Pass 0: fold inverse transpose pairs around elementwise-only chains
        for t2_node in nodes:
            if t2_node.op_type != "Transpose":
                continue
            t2_in = (_node_inputs(t2_node) or [None])[0]
            if not isinstance(t2_in, ir.Value):
                continue
            perm2 = _transpose_perm(t2_node)
            if perm2 is None:
                continue
            chain_match = _collect_transpose_elementwise_chain(nodes, t2_in)
            if chain_match is None:
                continue
            T1, elem_nodes = chain_match
            if T1 is t2_node:
                continue
            perm1 = _transpose_perm(T1)
            if perm1 is None or not _is_inverse_perm(perm1, perm2):
                continue
            t1_out = _node_output(T1)
            if t1_out is None:
                continue
            ok = True
            for consumer in _consumer_nodes(nodes, t1_out):
                if consumer is t2_node:
                    continue
                if consumer not in elem_nodes:
                    ok = False
                    break
            if not ok:
                continue
            for node in elem_nodes:
                out = _node_output(node)
                if out is None:
                    continue
                for consumer in _consumer_nodes(nodes, out):
                    if consumer is t2_node:
                        continue
                    if consumer not in elem_nodes:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            t1_in = (_node_inputs(T1) or [None])[0]
            if t1_in is None:
                continue
            for node in elem_nodes:
                ins = _node_inputs(node)
                for idx, iv in enumerate(ins):
                    if iv is t1_out:
                        node.replace_input_with(idx, t1_in)
            for node in nodes:
                if node in elem_nodes:
                    _refresh_elementwise_output_shape(node)
            t2_out = _node_output(t2_node)
            if t2_out is None:
                continue
            new_src = t1_in if not elem_nodes else t2_in
            _replace_all_uses_with(t2_out, new_src, replace_graph_outputs=True)
            graph.remove([T1, t2_node])
            changed = True
            break
        if changed:
            continue
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if n.op_type != "Transpose":
                i += 1
                continue
            T1 = n
            T1_out = _node_output(T1)
            consumers = _consumer_nodes(nodes, T1_out)
            if len(consumers) == 0:
                i += 1
                continue

            # Case 1: Single consumer - use existing chain-following logic
            if len(consumers) == 1:
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
                if perm1 is None or perm2 is None or not _is_inverse_perm(perm1, perm2):
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
                    last_allowed = allowed_nodes[-1]
                    _replace_all_uses_with(
                        _node_output(T1), t1_in, replace_graph_outputs=True
                    )
                    new_src = _node_output(last_allowed) or t1_in
                else:
                    new_src = t1_in
                old_out = _node_output(T2)
                assert old_out is not None
                _replace_all_uses_with(old_out, new_src, replace_graph_outputs=True)
                graph.remove([T1, T2])
                changed = True
                break

            # Case 2: Multiple consumers - check for direct Transpose consumers
            # that cancel with T1 (no intermediate ops allowed in this case)
            else:
                t1_in = (_node_inputs(T1) or [None])[0]
                if t1_in is None:
                    i += 1
                    continue
                perm1 = _transpose_perm(T1)
                if perm1 is None:
                    i += 1
                    continue

                # Find direct Transpose consumers that cancel with T1
                removed_any = False
                for consumer in consumers:
                    if consumer.op_type != "Transpose":
                        continue
                    T2 = consumer
                    perm2 = _transpose_perm(T2)
                    if perm2 is None or not _is_inverse_perm(perm1, perm2):
                        continue
                    # Found a direct T2 that cancels with T1
                    if TRN_DEBUG:
                        print(
                            "[transposefold/multi]",
                            "T1 ->",
                            T1.op_type,
                            "T2 ->",
                            T2.op_type,
                            "perm1",
                            perm1,
                            "perm2",
                            perm2,
                        )
                    # Rewire T2's consumers to use T1's input directly
                    old_out = _node_output(T2)
                    assert old_out is not None
                    _replace_all_uses_with(old_out, t1_in, replace_graph_outputs=True)
                    # Remove only T2, not T1 (T1 still has other consumers)
                    graph.remove(T2)
                    removed_any = True
                    break  # Restart the outer loop after modification

                if removed_any:
                    changed = True
                    break
                i += 1
                continue
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
            chain_nodes: Set[ir.Node] = set(allowed_fwd)

            # Safety gate: only fold reshape pairs when the chain is isolated.
            # If any intermediate elementwise output also feeds non-chain
            # consumers (notably Shape/Slice/Concat shape-reconstruction paths),
            # removing the pair can change rank assumptions downstream.
            safe_chain = True

            t1_out = _node_output(T1)
            if t1_out is not None:
                for consumer in _consumer_nodes(nodes, t1_out):
                    if consumer in chain_nodes or consumer is T2:
                        continue
                    safe_chain = False
                    break

            if safe_chain:
                for node in allowed_fwd:
                    out = _node_output(node)
                    if out is None:
                        continue
                    for consumer in _consumer_nodes(nodes, out):
                        if consumer in chain_nodes or consumer is T2:
                            continue
                        safe_chain = False
                        break
                    if not safe_chain:
                        break

            if not safe_chain:
                i += 1
                continue

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
                last_allowed = allowed_fwd[-1]
                _replace_all_uses_with(
                    _node_output(T1), src, replace_graph_outputs=True
                )
                new_src = _node_output(last_allowed) or src
            else:
                new_src = src
            old_out = _node_output(T2)
            _replace_all_uses_with(old_out, new_src, replace_graph_outputs=True)
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
            _replace_all_uses_with(dst_val, data_val, replace_graph_outputs=True)
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


def propagate_elementwise_shapes_ir(graph: ir.Graph) -> None:
    """Refresh output shapes for binary elementwise operators from current inputs."""
    nodes = list(graph)
    if not nodes:
        return
    for node in nodes:
        if node.op_type in ELEMENTWISE_BINARY_OPS:
            _refresh_elementwise_output_shape(node)


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

    if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray)):
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
                    _replace_all_uses_with(
                        old_not_out, rep_val, replace_graph_outputs=True
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


def remove_dead_nodes_ir(model: ir.Model) -> None:
    debug_metadata_flag = os.getenv("JAX2ONNX_ENABLE_STACKTRACE_METADATA", "")
    if debug_metadata_flag and debug_metadata_flag.strip().lower() not in (
        "0",
        "false",
        "off",
    ):
        # Keep the graph intact when stacktrace metadata is requested so downstream
        # tooling (e.g. sandbox repros) can inspect unused nodes.
        return
    common_passes.RemoveUnusedNodesPass()(model)


# ---------------- Targeted cleanup: orphan Transpose nodes ----------------


def _has_named_consumer(
    nodes: Sequence[ir.Node], *, producer: ir.Node, output_name: str
) -> bool:
    for node in nodes:
        if node is producer:
            continue
        for inp in _node_inputs(node):
            if _v_name(inp) == output_name:
                return True
    return False


def remove_orphan_transposes_ir(graph: ir.Graph) -> None:
    """
    Remove Transpose nodes whose outputs are not consumed and are not graph outputs.

    This pass uses name-based consumer discovery to stay robust even when IR-level
    consumer links are stale.
    """
    while True:
        nodes = list(cast(NodeSeq, graph))
        graph_output_names: Set[str] = {
            name for out in graph.outputs if (name := _v_name(out)) is not None
        }
        to_remove: List[ir.Node] = []

        for node in nodes:
            if node.op_type != "Transpose":
                continue

            outputs = _node_outputs(node)
            if not outputs:
                to_remove.append(node)
                continue

            is_live = False
            for out in outputs:
                out_name = _v_name(out)
                if out_name is None:
                    continue
                if out_name in graph_output_names:
                    is_live = True
                    break
                if _has_named_consumer(nodes, producer=node, output_name=out_name):
                    is_live = True
                    break

            if not is_live:
                to_remove.append(node)

        if not to_remove:
            break

        if DCE_DEBUG:
            _dbg(
                "remove_orphan_transposes_ir removed:",
                [n.name or "<unnamed>" for n in to_remove],
            )
        graph.remove(to_remove)


# ---------------- Prune unused graph inputs (top graph only) ----------------


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


def optimize_graph(ir_model: ir.Model) -> ir.Model:
    _dbg("optimize_graph invoked")
    # Top graph
    gr = ir_model.graph
    common_passes.NameFixPass()(ir_model)
    remove_redundant_casts_ir(gr)
    remove_redundant_transpose_reduce_ir(gr)
    remove_redundant_transpose_add_forests_ir(gr)
    remove_redundant_transpose_pairs_ir(gr)
    remove_redundant_reshape_pairs_ir(gr)
    remove_identity_reshapes_ir(gr)
    common_passes.CommonSubexpressionEliminationPass()(ir_model)
    common_passes.LiftConstantsToInitializersPass(size_limit=0)(ir_model)
    rewrite_mul_rsqrt_as_div_ir(gr)
    inline_dropout_training_mode_constants_ir(gr)
    propagate_elementwise_shapes_ir(gr)
    propagate_unary_shapes_ir(gr)
    remove_redundant_casts_ir(gr)
    remove_dead_nodes_ir(ir_model)
    remove_orphan_transposes_ir(gr)
    prune_unused_graph_inputs_ir(gr)

    # The passes are destructive; might as well raise exceptions if they occur.

    # Function bodies – do NOT prune function inputs (signature!)
    for fn in ir_model.functions.values():
        fgr = fn.graph
        remove_redundant_casts_ir(fgr)
        remove_redundant_transpose_reduce_ir(fgr)
        remove_redundant_transpose_add_forests_ir(fgr)
        remove_redundant_transpose_pairs_ir(fgr)
        remove_redundant_reshape_pairs_ir(fgr)
        remove_identity_reshapes_ir(fgr)
        rewrite_mul_rsqrt_as_div_ir(fgr)
        inline_dropout_training_mode_constants_ir(fgr)
        propagate_elementwise_shapes_ir(fgr)
        propagate_unary_shapes_ir(fgr)
        remove_redundant_casts_ir(fgr)
        remove_orphan_transposes_ir(fgr)

    return ir_model
