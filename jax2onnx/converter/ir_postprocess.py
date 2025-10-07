# jax2onnx/converter/ir_postprocess.py

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from typing import Iterable, Sequence, Any

import numpy as np
import onnx_ir as ir
from onnx_ir import AttributeType as IRAttrType


def _list_from_maybe_iterable(obj: object) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, (str, bytes)):
        return []
    if isinstance(obj, IterableABC):
        try:
            return list(obj)
        except Exception:
            return []
    return []


def _value_name(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    return getattr(value, "name", None)


def _shape_dims(shape_obj: object) -> Sequence[object] | None:
    if shape_obj is None:
        return None
    dims = getattr(shape_obj, "dims", None)
    if dims is not None:
        return list(dims)
    if isinstance(shape_obj, (list, tuple)):
        return list(shape_obj)
    return None


def _dim_is_known(dim: object) -> bool:
    if dim is None:
        return False
    if isinstance(dim, (int, np.integer)):
        return True
    if isinstance(dim, str):
        return bool(dim)
    for attr in ("param", "name", "symbol", "label"):
        val = getattr(dim, attr, None)
        if val:
            return True
    value_attr = getattr(dim, "value", None)
    if isinstance(value_attr, (int, np.integer)):
        return True
    try:
        text = str(dim)
        if text and text.isidentifier():
            return True
        if "SymbolicDim" in text:
            return True
    except Exception:
        pass
    return False


def _make_unknown_shape_like(
    shape_obj: object, *, force_rank_only: bool = False
) -> ir.Shape | None:
    dims = _shape_dims(shape_obj)
    if dims is None:
        return None
    if not dims:
        return None
    if not force_rank_only and all(_dim_is_known(d) for d in dims):
        return None
    unknown_dims = tuple(None for _ in dims)
    return ir.Shape(unknown_dims)


def _node_iter(graph: ir.Graph) -> Iterable[ir.Node]:
    all_nodes_fn = getattr(graph, "all_nodes", None)
    if callable(all_nodes_fn):
        try:
            nodes = list(all_nodes_fn())
            if nodes:
                return nodes
        except Exception:
            pass

    nodes_attr = getattr(graph, "node", None)
    if isinstance(nodes_attr, list):
        return list(nodes_attr)
    nodes_from_attr = _list_from_maybe_iterable(nodes_attr)
    if nodes_from_attr:
        return nodes_from_attr

    nodes_private = _list_from_maybe_iterable(getattr(graph, "_nodes", None))
    if nodes_private:
        return nodes_private
    return []


def _attribute_iter(node: ir.Node) -> Iterable[object]:
    return node.attributes.values()


def _loosen_graph_value_infos(
    graph: ir.Graph, *, force_rank_only: bool = False
) -> None:
    io_values = _list_from_maybe_iterable(
        getattr(graph, "inputs", None)
    ) + _list_from_maybe_iterable(getattr(graph, "outputs", None))
    io_names = {name for name in (_value_name(v) for v in io_values) if name}

    seen: set[str] = set()
    for node in _node_iter(graph):
        for out in getattr(node, "outputs", []) or []:
            if not isinstance(out, ir.Value):
                continue
            name = _value_name(out)
            if name and name in io_names:
                continue
            new_shape = _make_unknown_shape_like(
                getattr(out, "shape", None), force_rank_only=force_rank_only
            )
            if new_shape is not None:
                out.shape = new_shape
                tensor_type = getattr(out, "type", None)
                if isinstance(tensor_type, ir.TensorType):
                    out.type = ir.TensorType(tensor_type.elem_type)
            if name:
                seen.add(name)

    # Clean up any existing value_info containers if present
    for attr_name in ("value_info", "_value_info"):
        vi_list = getattr(graph, attr_name, None)
        if isinstance(vi_list, list):
            filtered: list[ir.Value] = []
            for vi in vi_list:
                name = _value_name(vi)
                if name and name in io_names:
                    filtered.append(vi)
                    continue
                new_shape = _make_unknown_shape_like(
                    getattr(vi, "shape", None), force_rank_only=force_rank_only
                )
                if new_shape is not None:
                    vi.shape = new_shape
                    tt = getattr(vi, "type", None)
                    if isinstance(tt, ir.TensorType):
                        vi.type = ir.TensorType(tt.elem_type)
                filtered.append(vi)
            setattr(graph, attr_name, filtered)


def _tensor_to_numpy(tensor_obj: object) -> np.ndarray | None:
    if tensor_obj is None:
        return None
    if hasattr(tensor_obj, "to_numpy"):
        try:
            return tensor_obj.to_numpy()
        except Exception:
            pass
    try:
        return np.asarray(tensor_obj)
    except Exception:
        return None


def _maybe_promote_value_to_double(val: ir.Value) -> None:
    const_val = getattr(val, "const_value", None)
    arr = _tensor_to_numpy(const_val)
    if arr is None or arr.dtype != np.float32:
        return
    promoted = ir.tensor(arr.astype(np.float64))
    val.const_value = promoted
    if isinstance(getattr(val, "type", None), ir.TensorType):
        val.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        val.type = ir.TensorType(ir.DataType.DOUBLE)


def _promote_constant_attributes(node: ir.Node) -> None:
    attrs = _attribute_iter(node)
    for attr in attrs:
        attr_name = getattr(attr, "name", None)
        attr_type = getattr(attr, "type", None)
        if attr_name != "value" or attr_type != IRAttrType.TENSOR:
            continue
        tensor_obj = getattr(attr, "value", None)
        arr = _tensor_to_numpy(tensor_obj)
        if arr is None or arr.dtype != np.float32:
            continue
        promoted_tensor = ir.tensor(arr.astype(np.float64))
        if hasattr(attr, "_value"):
            attr._value = promoted_tensor
        else:
            setattr(attr, "value", promoted_tensor)


def _process_graph(
    graph: ir.Graph,
    *,
    loosen: bool,
    promote: bool,
    force_rank_only: bool = False,
) -> None:
    if loosen:
        _loosen_graph_value_infos(graph, force_rank_only=force_rank_only)

    if promote:
        for init_val in getattr(graph, "initializers", []) or []:
            if isinstance(init_val, ir.Value):
                _maybe_promote_value_to_double(init_val)

    for node in _node_iter(graph):
        if promote and getattr(node, "op_type", None) == "Constant":
            _promote_constant_attributes(node)

        if promote:
            for output in getattr(node, "outputs", []) or []:
                if isinstance(output, ir.Value):
                    _maybe_promote_value_to_double(output)

        child_force_rank_only = force_rank_only or getattr(node, "op_type", "") in {
            "Loop",
            "Scan",
        }
        for attr in _attribute_iter(node):
            attr_type = getattr(attr, "type", None)
            if attr_type == IRAttrType.GRAPH:
                sub_graph = getattr(attr, "value", None)
                if sub_graph is not None:
                    _process_graph(
                        sub_graph,
                        loosen=loosen,
                        promote=promote,
                        force_rank_only=child_force_rank_only,
                    )
            elif attr_type == IRAttrType.GRAPHS:
                sub_graphs = getattr(attr, "value", None)
                if sub_graphs is not None:
                    for sub in sub_graphs:
                        _process_graph(
                            sub,
                            loosen=loosen,
                            promote=promote,
                            force_rank_only=child_force_rank_only,
                        )


def _process_functions(model: ir.Model, *, loosen: bool, promote: bool) -> None:
    for fn in getattr(model, "functions", []) or []:
        if loosen:
            _loosen_graph_value_infos(fn, force_rank_only=False)
        _process_graph(fn, loosen=loosen, promote=promote, force_rank_only=False)


def postprocess_ir_model(model: ir.Model, *, promote_to_double: bool) -> None:
    _process_graph(model.graph, loosen=True, promote=False, force_rank_only=False)
    _process_functions(model, loosen=True, promote=False)
