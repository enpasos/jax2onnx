from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import onnx_ir as ir
from onnx_ir import AttributeType as IRAttrType


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


def _make_unknown_shape_like(shape_obj: object) -> ir.Shape | None:
    dims = _shape_dims(shape_obj)
    if dims is None:
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
    if hasattr(nodes_attr, "__iter__") and not callable(nodes_attr):
        return list(nodes_attr)

    nodes_private = getattr(graph, "_nodes", None)
    if nodes_private is not None:
        return list(nodes_private)
    return []


def _attribute_iter(node: ir.Node) -> Iterable[object]:
    attrs = getattr(node, "attributes", None)
    if attrs is None:
        return []
    if isinstance(attrs, dict):
        return list(attrs.values())
    if hasattr(attrs, "values"):
        try:
            return list(attrs.values())
        except Exception:
            pass
    if isinstance(attrs, Iterable):
        return list(attrs)
    return []


def _loosen_graph_value_infos(graph: ir.Graph) -> None:
    io_values = list(getattr(graph, "inputs", []) or []) + list(
        getattr(graph, "outputs", []) or []
    )
    io_names = {
        name
        for name in (_value_name(v) for v in io_values)
        if name
    }

    seen: set[str] = set()
    for node in _node_iter(graph):
        for out in getattr(node, "outputs", []) or []:
            if not isinstance(out, ir.Value):
                continue
            name = _value_name(out)
            if name and name in io_names:
                continue
            new_shape = _make_unknown_shape_like(getattr(out, "shape", None))
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
                new_shape = _make_unknown_shape_like(getattr(vi, "shape", None))
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


def _process_graph(graph: ir.Graph, *, loosen: bool, promote: bool) -> None:
    if loosen:
        _loosen_graph_value_infos(graph)

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

        for attr in _attribute_iter(node):
            attr_type = getattr(attr, "type", None)
            if attr_type == IRAttrType.GRAPH:
                sub_graph = getattr(attr, "value", None)
                if sub_graph is not None:
                    _process_graph(sub_graph, loosen=loosen, promote=promote)
            elif attr_type == IRAttrType.GRAPHS:
                sub_graphs = getattr(attr, "value", None)
                if sub_graphs is not None:
                    for sub in sub_graphs:
                        _process_graph(sub, loosen=loosen, promote=promote)


def _process_functions(model: ir.Model, *, loosen: bool, promote: bool) -> None:
    for fn in getattr(model, "functions", []) or []:
        if loosen:
            _loosen_graph_value_infos(fn)
        _process_graph(fn, loosen=loosen, promote=promote)


def postprocess_ir_model(model: ir.Model, *, promote_to_double: bool) -> None:
    _process_graph(model.graph, loosen=True, promote=False)
    _process_functions(model, loosen=True, promote=False)
