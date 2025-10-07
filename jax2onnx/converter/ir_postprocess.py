# jax2onnx/converter/ir_postprocess.py

from __future__ import annotations

from itertools import chain

import numpy as np
import onnx_ir as ir
from onnx_ir import AttributeType


def _value_name(value: ir.Value | None) -> str | None:
    if value is None:
        return None
    name = value.name
    return name or None


def _io_value_names(graph: ir.Graph) -> set[str]:
    return {
        name
        for name in (_value_name(value) for value in chain(graph.inputs, graph.outputs))
        if name
    }


def _shape_dims(shape_obj: object) -> list[object] | None:
    if shape_obj is None:
        return None
    if isinstance(shape_obj, ir.Shape):
        return list(shape_obj)
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
    for attr in ("param", "name", "symbol", "label", "value"):
        try:
            val = getattr(dim, attr)
        except Exception:
            continue
        if isinstance(val, (int, np.integer)):
            return True
        if val:
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


def _normalize_dim(dim: object) -> object:
    if isinstance(dim, np.integer):
        return int(dim)
    if isinstance(dim, str):
        return ir.SymbolicDim(dim)
    return dim


def _unknown_shape_like(value: ir.Value, *, force_rank_only: bool) -> ir.Shape | None:
    dims = _shape_dims(value.shape)
    if not dims:
        return None
    new_dims: list[object | None] = []
    changed = False
    for dim in dims:
        if force_rank_only:
            new_dims.append(None)
            changed = True
            continue
        if _dim_is_known(dim):
            new_dims.append(_normalize_dim(dim))
        else:
            new_dims.append(None)
            changed = True
    if not changed:
        return None
    return ir.Shape(tuple(new_dims))


def _reset_tensor_type(value: ir.Value) -> None:
    dtype = value.dtype
    if dtype is None:
        return
    value.type = ir.TensorType(dtype)


def _loosen_graph_value_infos(
    graph: ir.Graph, *, force_rank_only: bool = False
) -> None:
    io_names = _io_value_names(graph)

    for node in graph:
        for output in node.outputs:
            if output is None:
                continue
            name = _value_name(output)
            if name and name in io_names:
                continue
            unknown_shape = _unknown_shape_like(output, force_rank_only=force_rank_only)
            if unknown_shape is None:
                continue
            output.shape = unknown_shape
            _reset_tensor_type(output)


def _tensor_to_numpy(tensor: object) -> np.ndarray | None:
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, ir.Tensor):
        return tensor.numpy()
    if hasattr(tensor, "numpy"):
        result = tensor.numpy()
        return result if isinstance(result, np.ndarray) else np.asarray(result)
    if isinstance(tensor, (list, tuple)):
        return np.asarray(tensor)
    return None


def _maybe_promote_value_to_double(value: ir.Value) -> None:
    tensor = value.const_value
    array = _tensor_to_numpy(tensor)
    if array is None or array.dtype != np.float32:
        return
    promoted = ir.tensor(array.astype(np.float64))
    value.const_value = promoted
    value.type = ir.TensorType(ir.DataType.DOUBLE)


def _promote_constant_attributes(node: ir.Node) -> None:
    value_attr = node.attributes.get("value")
    if value_attr is None or value_attr.type is not AttributeType.TENSOR:
        return
    tensor = value_attr.as_tensor()
    array = tensor.numpy()
    if array.dtype != np.float32:
        return
    promoted = ir.tensor(array.astype(np.float64))
    node.attributes["value"] = ir.Attr(
        "value",
        AttributeType.TENSOR,
        promoted,
    )


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
        for initializer in graph.initializers.values():
            _maybe_promote_value_to_double(initializer)

    for node in graph:
        if promote and node.op_type == "Constant":
            _promote_constant_attributes(node)

        if promote:
            for output in node.outputs:
                if output is not None:
                    _maybe_promote_value_to_double(output)

        child_force_rank_only = force_rank_only or node.op_type in {"Loop", "Scan"}
        for attr in list(node.attributes.values()):
            if attr.type is AttributeType.GRAPH:
                sub_graph = attr.as_graph()
                if sub_graph is not None:
                    _process_graph(
                        sub_graph,
                        loosen=loosen,
                        promote=promote,
                        force_rank_only=child_force_rank_only,
                    )
            elif attr.type is AttributeType.GRAPHS:
                sub_graphs = attr.as_graphs()
                for sub_graph in sub_graphs:
                    _process_graph(
                        sub_graph,
                        loosen=loosen,
                        promote=promote,
                        force_rank_only=child_force_rank_only,
                    )


def _process_functions(model: ir.Model, *, loosen: bool, promote: bool) -> None:
    for function in model.functions.values():
        if loosen:
            _loosen_graph_value_infos(function, force_rank_only=False)
        _process_graph(function, loosen=loosen, promote=promote, force_rank_only=False)


def postprocess_ir_model(model: ir.Model, *, promote_to_double: bool) -> None:
    _process_graph(
        model.graph,
        loosen=True,
        promote=False,
        force_rank_only=False,
    )
    _process_functions(model, loosen=True, promote=False)
