# jax2onnx/converter/conv_transpose_bias_folding.py

from __future__ import annotations

from typing import Any, Final

import numpy as np
import onnx_ir as ir
from onnx_ir.tape import Tape

from jax2onnx.ir_utils import const_value_to_numpy

from .optimizer_graph_utils import _node_inputs, _node_outputs, _producer_node

_BIAS_DTYPES: Final = frozenset({ir.DataType.FLOAT, ir.DataType.DOUBLE})
_SHAPE_ONLY_OPS: Final = frozenset(
    {"Identity", "Reshape", "Unsqueeze", "Squeeze", "Transpose"}
)


def _standard_node(node: ir.Node, op_type: str) -> bool:
    return node.op_type == op_type and node.domain == "" and not node.overload


def _static_shape(value: ir.Value) -> tuple[int, ...] | None:
    if value.shape is None:
        return None
    dimensions: list[int] = []
    for dimension in value.shape:
        if not isinstance(dimension, int):
            return None
        dimensions.append(dimension)
    return tuple(dimensions)


def _immutable_array(value: ir.Value) -> np.ndarray[Any, np.dtype[Any]] | None:
    """Read only actual initializers/Constants, never overridable inputs."""
    if value.is_graph_input() or value.graph is None:
        return None
    producer = value.producer()
    if not value.is_initializer() and (
        producer is None
        or not _standard_node(producer, "Constant")
        or producer.inputs
        or any(attribute.is_ref() for attribute in producer.attributes.values())
    ):
        return None
    return const_value_to_numpy(value)


def _channel_constant(
    value: ir.Value,
    graph: ir.Graph,
    *,
    channels: int,
    dtype: ir.DataType,
) -> tuple[ir.Value, np.ndarray[Any, np.dtype[Any]]] | None:
    """Unwrap shape-only constant views whose only non-unit axis is C.

    Moving unit dimensions does not reorder this vector. Requiring constant
    auxiliary inputs also prevents folding a runtime reshape or axes parameter.
    """
    nodes = list(graph)
    seen: set[ir.Value] = set()
    while value not in seen:
        seen.add(value)
        if value.is_graph_input() or value.dtype != dtype:
            return None
        shape = _static_shape(value)
        if shape is None or any(dim not in {1, channels} for dim in shape):
            return None
        if int(np.prod(shape, dtype=np.int64)) != channels:
            return None
        array = _immutable_array(value)
        if array is not None:
            expected_dtype = np.dtype(
                np.float32 if dtype == ir.DataType.FLOAT else np.float64
            )
            if array.dtype != expected_dtype or tuple(array.shape) != shape:
                return None
            return value, array

        producer = _producer_node(nodes, value)
        if (
            producer is None
            or producer.graph is not graph
            or producer.domain != ""
            or producer.overload
            or producer.op_type not in _SHAPE_ONLY_OPS
            or len(_node_outputs(producer)) != 1
            or any(attr.is_ref() for attr in producer.attributes.values())
        ):
            return None
        inputs = _node_inputs(producer)
        if not inputs or inputs[0] is None:
            return None
        if producer.op_type in {"Identity", "Transpose"} and len(inputs) != 1:
            return None
        if any(
            auxiliary is None or _immutable_array(auxiliary) is None
            for auxiliary in inputs[1:]
        ):
            return None
        value = inputs[0]
    return None


def _bias_vector(
    graph: ir.Graph,
    before: ir.Node,
    source: ir.Value,
    array: np.ndarray[Any, np.dtype[Any]],
) -> ir.Value:
    if array.ndim == 1:
        return source
    # A local Constant works in graph bodies and ONNX Functions alike. It also
    # keeps a shared shaped initializer unchanged for its other consumers.
    tensor = ir.tensor(array.reshape(-1))
    tape = Tape()
    vector = tape.op("Constant", [], attributes={"value": tensor})
    vector.const_value = tensor
    vector.dtype = tensor.dtype
    vector.shape = ir.Shape((array.size,))
    graph.insert_before(before, tape.nodes)
    return vector


def _replace_with_biased_conv(
    graph: ir.Graph,
    conv: ir.Node,
    replacement: ir.Node,
    vector: ir.Value,
    *,
    removed: tuple[ir.Node, ...],
) -> None:
    inputs = _node_inputs(conv)
    output = replacement.outputs[0]
    conv_output = conv.outputs[0]
    # Keep the public result Value and the convolution's name for late overrides.
    replacement.resize_inputs(3)
    replacement.replace_input_with(0, inputs[0])
    replacement.replace_input_with(1, inputs[1])
    replacement.replace_input_with(2, vector)
    replacement.op_type = "ConvTranspose"
    replacement.version = conv.version
    replacement.attributes.clear()
    replacement.attributes.update(conv.attributes)
    replacement.name = conv.name
    replacement.doc_string = conv.doc_string
    replacement.metadata_props.update(conv.metadata_props)
    if output.dtype is None and conv_output.dtype is not None:
        output.dtype = conv_output.dtype
    if output.shape is None and conv_output.shape is not None:
        output.shape = ir.Shape(conv_output.shape)
    graph.remove(removed, safe=True)


def _fold_local_graph(graph: ir.Graph) -> None:
    nodes = list(graph)
    for add in nodes:
        if not _standard_node(add, "Add") or add.attributes:
            continue
        add_opset = add.version or graph.opset_imports.get("", 0)
        if add_opset < 7:
            continue
        add_inputs = _node_inputs(add)
        add_outputs = _node_outputs(add)
        if len(add_inputs) != 2 or len(add_outputs) != 1:
            continue
        for index, conv_output in enumerate(add_inputs):
            if conv_output is None:
                continue
            conv = _producer_node(nodes, conv_output)
            if (
                conv is None
                or conv.graph is not graph
                or not _standard_node(conv, "ConvTranspose")
                or len(_node_outputs(conv)) != 1
                or conv_output.is_graph_output()
            ):
                continue
            uses = tuple(conv_output.uses())
            if len(uses) != 1 or uses[0].node is not add:
                continue
            conv_inputs = _node_inputs(conv)
            if len(conv_inputs) not in {2, 3} or (
                len(conv_inputs) == 3 and conv_inputs[2] is not None
            ):
                continue
            dtype = conv_output.dtype
            if (
                dtype is None
                or dtype not in _BIAS_DTYPES
                or any(
                    operand is None or operand.dtype != dtype
                    for operand in conv_inputs[:2]
                )
            ):
                continue
            shape = conv_output.shape
            if shape is None or len(shape) < 3 or not isinstance(shape[1], int):
                continue
            channels = int(shape[1])
            bias = add_inputs[1 - index]
            if channels <= 0 or bias is None or bias.dtype != dtype:
                continue
            bias_shape = _static_shape(bias)
            rank = len(shape)
            if bias_shape is None or len(bias_shape) not in {rank - 1, rank}:
                continue
            aligned_shape = (1,) * (rank - len(bias_shape)) + bias_shape
            if aligned_shape != (1, channels) + (1,) * (rank - 2):
                continue
            result = add_outputs[0]
            if result.dtype is not None and result.dtype != dtype:
                continue
            if result.shape is not None and tuple(result.shape) != tuple(shape):
                continue
            matched_bias = _channel_constant(
                bias, graph, channels=channels, dtype=dtype
            )
            if matched_bias is None:
                continue
            source, array = matched_bias
            vector = _bias_vector(graph, add, source, array)

            _replace_with_biased_conv(graph, conv, add, vector, removed=(conv,))
            break


def _permutation(node: ir.Node, rank: int) -> tuple[int, ...] | None:
    if not _standard_node(node, "Transpose") or set(node.attributes) != {"perm"}:
        return None
    attribute = node.attributes["perm"]
    if attribute.is_ref() or attribute.type != ir.AttributeType.INTS:
        return None
    permutation = tuple(attribute.as_ints())
    if sorted(permutation) != list(range(rank)):
        return None
    return permutation


def _single_use(value: ir.Value, consumer: ir.Node) -> bool:
    uses = tuple(value.uses())
    return not value.is_graph_output() and len(uses) == 1 and uses[0].node is consumer


def _fold_layout_pairs(graph: ir.Graph) -> None:
    """Commute a channel Add through an otherwise empty inverse layout pair."""
    nodes = list(graph)
    for add in nodes:
        if (
            not _standard_node(add, "Add")
            or add.attributes
            or (add.version or graph.opset_imports.get("", 0)) < 7
            or len(add.inputs) != 2
            or len(add.outputs) != 1
        ):
            continue
        added = add.outputs[0]
        added_uses = tuple(added.uses())
        if added.is_graph_output() or len(added_uses) != 1:
            continue
        after = added_uses[0].node
        if (
            after.graph is not graph
            or len(after.inputs) != 1
            or len(after.outputs) != 1
        ):
            continue
        for index, transposed in enumerate(add.inputs):
            if transposed is None:
                continue
            before = _producer_node(nodes, transposed)
            if (
                before is None
                or before.graph is not graph
                or len(before.inputs) != 1
                or len(before.outputs) != 1
                or not _single_use(transposed, add)
            ):
                continue
            raw = before.inputs[0]
            if raw is None or not _single_use(raw, before):
                continue
            conv = _producer_node(nodes, raw)
            if (
                conv is None
                or conv.graph is not graph
                or not _standard_node(conv, "ConvTranspose")
                or len(conv.outputs) != 1
                or len(conv.inputs) not in {2, 3}
                or (len(conv.inputs) == 3 and conv.inputs[2] is not None)
            ):
                continue
            dtype = raw.dtype
            shape = raw.shape
            if (
                dtype is None
                or dtype not in _BIAS_DTYPES
                or shape is None
                or len(shape) < 3
                or not isinstance(shape[1], int)
                or shape[1] <= 0
                or any(
                    value is None or value.dtype != dtype for value in conv.inputs[:2]
                )
            ):
                continue
            rank = len(shape)
            perm_before = _permutation(before, rank)
            perm_after = _permutation(after, rank)
            if (
                perm_before is None
                or perm_after is None
                or tuple(perm_before[axis] for axis in perm_after) != tuple(range(rank))
            ):
                continue
            view_shape = tuple(shape[axis] for axis in perm_before)
            output = after.outputs[0]
            if (
                transposed.dtype != dtype
                or transposed.shape is None
                or tuple(transposed.shape) != view_shape
                or added.dtype != dtype
                or added.shape is None
                or tuple(added.shape) != view_shape
                or output.dtype != dtype
                or output.shape is None
                or tuple(output.shape) != tuple(shape)
            ):
                continue
            bias = add.inputs[1 - index]
            if bias is None or bias.dtype != dtype:
                continue
            bias_shape = _static_shape(bias)
            if bias_shape is None or not bias_shape or len(bias_shape) > rank:
                continue
            aligned = (1,) * (rank - len(bias_shape)) + bias_shape
            nchw_bias = tuple(aligned[axis] for axis in perm_after)
            channels = int(shape[1])
            if nchw_bias != (1, channels) + (1,) * (rank - 2):
                continue
            matched = _channel_constant(bias, graph, channels=channels, dtype=dtype)
            if matched is None:
                continue
            source, array = matched
            vector = _bias_vector(graph, after, source, array)
            _replace_with_biased_conv(
                graph, conv, after, vector, removed=(add, before, conv)
            )
            break


def fold_conv_transpose_constant_bias(graph: ir.Graph) -> None:
    """Attach a direct immutable channel Add as an absent ConvTranspose bias.

    Do not merge existing biases, cross Pad, or change raw
    convolution values that remain observable. Low precision is excluded because
    fusing its bias can change intermediate rounding. Only constant shape views
    and a single inverse Transpose pair around Add are evaluated; there is no
    parameter arithmetic or normalization folding.
    """
    for local_graph in (graph, *graph.subgraphs()):
        _fold_local_graph(local_graph)
        _fold_layout_pairs(local_graph)
