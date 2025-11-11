# jax2onnx/converter/ir_postprocess.py

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping, Sequence as SequenceABC
from itertools import chain
from typing import Any, Iterable, TypeAlias, Union, cast

import numpy as np
import onnx_ir as ir
from onnx_ir import AttributeType

DimValue: TypeAlias = Union[int, ir.SymbolicDim, None]


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


def _shape_dims(shape_obj: object) -> list[DimValue] | None:
    if shape_obj is None:
        return None
    if isinstance(shape_obj, ir.Shape):
        return [dim for dim in shape_obj.dims]
    if isinstance(shape_obj, SequenceABC) and not isinstance(shape_obj, (str, bytes)):
        dims: list[DimValue] = []
        dims_src = cast(Iterable[object], shape_obj)
        for dim in dims_src:
            if dim is None:
                dims.append(None)
            elif isinstance(dim, (int, ir.SymbolicDim)):
                dims.append(dim)
            else:
                dims.append(ir.SymbolicDim(str(dim)))
        return dims
    return None


def _dim_is_known(dim: object) -> bool:
    if dim is None:
        return False
    if isinstance(dim, (int, np.integer)):
        return True
    if isinstance(dim, ir.SymbolicDim):
        if dim.value is not None:
            return True
        text = str(dim)
        return bool(text and text != "?")
    if isinstance(dim, str):
        return bool(dim)
    text = str(dim)
    return bool(text and text != "?")


def _normalize_dim(dim: object) -> DimValue:
    if isinstance(dim, np.integer):
        return int(dim)
    if isinstance(dim, int):
        return dim
    if isinstance(dim, ir.SymbolicDim):
        return dim
    if isinstance(dim, str):
        return ir.SymbolicDim(dim)
    return None


def _unknown_shape_like(value: ir.Value, *, force_rank_only: bool) -> ir.Shape | None:
    dims = _shape_dims(value.shape)
    if not dims:
        return None
    new_dims: list[DimValue] = []
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


def _iter_initializers(graph: ir.Graph) -> IterableABC[ir.Value]:
    initializers_obj = graph.initializers
    if hasattr(initializers_obj, "values"):
        return cast(IterableABC[ir.Value], initializers_obj.values())
    return cast(IterableABC[ir.Value], initializers_obj)


def _loosen_graph_value_shapes(
    graph: ir.Graph, *, force_rank_only: bool = False
) -> None:
    io_names = _io_value_names(graph)

    for node in graph:
        for output in node.outputs:
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
    if hasattr(tensor, "numpy"):
        try:
            result = tensor.numpy()
        except Exception:
            return None
        if isinstance(result, np.ndarray):
            return result
        return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(result))
    if isinstance(tensor, (list, tuple)):
        return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(tensor))
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
        _loosen_graph_value_shapes(graph, force_rank_only=force_rank_only)

    if promote:
        for initializer in _iter_initializers(graph):
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
                for sub_graph in attr.as_graphs():
                    _process_graph(
                        sub_graph,
                        loosen=loosen,
                        promote=promote,
                        force_rank_only=child_force_rank_only,
                    )


def _process_functions(model: ir.Model, *, loosen: bool, promote: bool) -> None:
    funcs_obj: object | None = model.functions
    if funcs_obj is None:
        return
    if isinstance(funcs_obj, Mapping):
        iterable = cast(
            Iterable[ir.Function | ir.Graph],
            funcs_obj.values(),
        )
    else:
        iterable = cast(Iterable[ir.Function | ir.Graph], funcs_obj)
    for fn_obj in iterable:
        if isinstance(fn_obj, ir.Function):
            graph_obj: ir.Graph = fn_obj.graph
        else:
            graph_obj = fn_obj
        if loosen:
            _loosen_graph_value_shapes(graph_obj, force_rank_only=False)
        _process_graph(graph_obj, loosen=loosen, promote=promote, force_rank_only=False)


def postprocess_ir_model(model: ir.Model, *, promote_to_double: bool) -> None:
    _process_graph(
        model.graph,
        loosen=True,
        promote=False,
        force_rank_only=False,
    )
    _process_functions(model, loosen=True, promote=False)
