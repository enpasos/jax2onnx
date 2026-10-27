# jax2onnx/plugins/_source_control_flow_structure.py

"""Original dynamic If contracts; neither source admission nor graph rewriting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, Literal
from jax2onnx.converter.source_structure import (
    EquationHandle,
    RegionKey,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
    _ORIGIN,
    _OWNER,
)


@dataclass(frozen=True)
class BooleanIndexRule:
    """Exact scalar BOOL -> strong INT32, including a proved composed use."""

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        if len(inputs) != 1 or len(outputs) != 1 or not isinstance(inputs[0], ir.Value):
            raise StructureUnavailable("boolean-index-ports-unavailable")
        node = outputs[0].producer()
        if (
            node is None
            or node.graph is not graph
            or node not in set(graph)
            or node.domain
            or node.overload
            or node.op_type != "Cast"
            or len(node.inputs) != 1
            or node.inputs[0] is not inputs[0]
            or len(node.outputs) != 1
            or node.outputs[0] is not outputs[0]
            or set(node.attributes) != {"to"}
            or node.attributes["to"].type is not ir.AttributeType.INT
            or node.attributes["to"].as_int() != int(ir.DataType.INT32.value)
        ):
            raise StructureMismatch("boolean-index-cast-differs")
        return (node,)


@dataclass(frozen=True)
class IfRule:
    """Child body content lives once in the frozen equation's native roles."""

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("dynamic-if-attachment-unavailable")


@dataclass(frozen=True)
class IfAttachment:
    origin: object
    owner: object
    true: RegionKey
    false: RegionKey
    true_owner: object
    false_owner: object


def freeze_control_flow_rule(
    equation: Any,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> BooleanIndexRule | IfRule | None:
    if equation.effects:
        return None
    if equation.primitive is jax.lax.convert_element_type_p:
        params = equation.params
        if (
            len(inputs) == len(outputs) == 1
            and inputs[0].logical_dtype == "bool"
            and inputs[0].dtype is ir.DataType.BOOL
            and outputs[0].logical_dtype == "int32"
            and outputs[0].dtype is ir.DataType.INT32
            and inputs[0].shape == outputs[0].shape == ()
            and inputs[0].logical_shape == outputs[0].logical_shape == ()
            and inputs[0].encoding == outputs[0].encoding == "identity"
            and outputs[0].weak_type is False
            and type(params) is dict
            and {"new_dtype", "weak_type"}
            <= set(params)
            <= {"new_dtype", "weak_type", "sharding"}
            and type(params["new_dtype"]) is type(np.dtype(np.int32))
            and params["new_dtype"] == np.dtype(np.int32)
            and params["weak_type"] is False
            and params.get("sharding") is None
        ):
            return BooleanIndexRule()
        return None
    if equation.primitive is not jax.lax.cond_p:
        return None
    if (
        not inputs
        or not outputs
        or isinstance(equation.invars[0], Literal)
        or type(equation.params) is not dict
        or set(equation.params) != {"branches"}
        or inputs[0].logical_dtype != "int32"
        or inputs[0].dtype is not ir.DataType.INT32
        or inputs[0].weak_type is not False
        or inputs[0].shape != ()
        or inputs[0].encoding != "identity"
        or any(value.symbol_token is not None for value in (*inputs, *outputs))
    ):
        return None
    branches = equation.params["branches"]
    if type(branches) is not tuple or len(branches) != 2:
        return None
    from jax2onnx.converter.source_structure import _ShapeFreezer

    for closed in branches:
        if type(closed) is not ClosedJaxpr and type(closed) is not Jaxpr:
            return None
        body = closed.jaxpr
        if body.effects or len(body.constvars) != len(closed.consts):
            return None
        shapes = _ShapeFreezer()
        if (
            tuple(TensorContract.from_aval(var.aval, shapes) for var in body.invars)
            != inputs[1:]
            or tuple(TensorContract.from_aval(var.aval, shapes) for var in body.outvars)
            != outputs
        ):
            return None
    return IfRule()


def record_if_attachment(
    ctx: Any,
    equation: Any,
    true: RegionKey,
    false: RegionKey,
    true_graph: ir.Graph,
    false_graph: ir.Graph,
) -> None:
    ledger = ctx._source_structure
    if not ledger.handles or type(ledger.handles[-1]) is not EquationHandle:
        raise StructureUnavailable("original-if-handle-unavailable")
    handle = ledger.handles[-1]
    if handle in ledger.if_calls or true.parent != handle or false.parent != handle:
        raise StructureUnavailable("if-attachment-identity-unavailable")
    outputs = tuple(ctx.lookup_value_for_var(var) for var in equation.outvars)
    if not outputs or outputs[0] is None or outputs[0].producer() is None:
        raise StructureUnavailable("if-attachment-output-unavailable")
    node = outputs[0].producer()
    ledger.if_calls[handle] = IfAttachment(
        node.meta.setdefault(_ORIGIN, object()),
        ctx.builder.graph.meta[_OWNER],
        true,
        false,
        true_graph.meta[_OWNER],
        false_graph.meta[_OWNER],
    )
