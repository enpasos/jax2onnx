# jax2onnx/plugins/_source_function_structure.py

"""Plugin-owned source Function identity and ordered graph-edge contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import onnx_ir as ir

from jax2onnx.converter.source_structure import (
    LiteralOperand,
    StructureMismatch,
    StructureUnavailable,
    _ORIGIN,
    _OWNER,
)


def _original_function_trace(equation: Any) -> Any:
    from jax2onnx.plugins.plugin_system import (
        FunctionPlugin,
        PLUGIN_REGISTRY,
        _SOURCE_TRACE,
        _SourceFunctionTrace,
    )

    owner = PLUGIN_REGISTRY.get(equation.primitive.name)
    if (
        not isinstance(owner, FunctionPlugin)
        or equation.primitive is not owner._source_primitive
    ):
        return None
    trace = equation.params.get(_SOURCE_TRACE)
    if type(trace) is not _SourceFunctionTrace:
        raise StructureUnavailable("original-function-trace-unavailable")
    return trace


def original_function_body(equation: Any) -> Any:
    trace = _original_function_trace(equation)
    return trace.body if trace is not None else None


def declared_function_body(equation: Any) -> Any:
    """Read existing region metadata independently of semantic-rule discovery."""
    from jax2onnx.converter._source_regions import FrozenBody

    trace = _original_function_trace(equation)
    if trace is None:
        return None
    if type(trace.body) is not FrozenBody:
        raise StructureUnavailable("original-function-declaration-unavailable")
    return trace.body


@dataclass(frozen=True)
class FunctionCall:
    owner: object
    origin: object
    order: tuple[int, ...]
    body: Any
    target: tuple[str, str, str]


@dataclass(frozen=True)
class FunctionRule:
    body: Any

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("function-call-attachment-unavailable")

    def check_call(
        self,
        graph: ir.Graph,
        inputs: Any,
        outputs: Any,
        match_literal: Any,
        call: FunctionCall,
        definition: ir.Function,
    ) -> tuple[ir.Node, ...]:
        if self.body.gaps:
            raise StructureUnavailable("original-function-body-contract-unavailable")
        if call.body != self.body or call.owner is not definition.graph.meta.get(
            _OWNER
        ):
            raise StructureMismatch("function-body-binding-differs")
        if not outputs:
            raise StructureUnavailable("zero-result-function-call-unavailable")
        node = outputs[0].producer()
        if (
            node is None
            or node.graph is not graph
            or node not in set(graph)
            or node.meta.get(_ORIGIN) is not call.origin
            or (node.domain, node.op_type, node.overload) != call.target
            or call.target
            != (definition.domain, definition.name, getattr(definition, "overload", ""))
            or node.attributes
            or len(node.outputs) != len(outputs)
            or any(
                actual is not expected
                for actual, expected in zip(node.outputs, outputs, strict=True)
            )
        ):
            raise StructureMismatch("function-call-target-or-result-differs")
        if (
            type(call.order) is not tuple
            or any(type(port) is not int for port in call.order)
            or sorted(call.order) != list(range(len(inputs)))
            or len(node.inputs) != len(inputs)
        ):
            raise StructureMismatch("function-call-port-permutation-differs")
        for actual, port in zip(node.inputs, call.order, strict=True):
            expected = inputs[port]
            if isinstance(expected, LiteralOperand):
                match_literal(actual, expected)
            elif actual is not expected:
                raise StructureMismatch("function-call-input-edge-differs")
        return (node,)


def record_function_call(
    ctx: Any,
    equation: Any,
    definition: ir.Function,
    body: Any,
    order: tuple[int, ...],
) -> None:
    from jax2onnx.converter._source_regions import FrozenBody

    ledger = ctx._source_structure
    if not ledger.handles or ledger.handles[-1] is None:
        raise StructureUnavailable("original-equation-handle-unavailable")
    handle = ledger.handles[-1]
    if handle in ledger.function_calls:
        raise StructureUnavailable("duplicate-function-call-receipt")
    scope = next(
        (
            value
            for value in ledger.active
            if value is not None and value.invocation is handle.invocation
        ),
        None,
    )
    if scope is None or type(scope.equations[handle.index].rule) is not FunctionRule:
        raise StructureUnavailable("original-function-equation-unavailable")
    expected = scope.equations[handle.index].rule.body
    if type(body) is not FrozenBody or body != expected or body.gaps:
        raise StructureUnavailable("original-function-body-contract-unavailable")
    owner = definition.graph.meta.get(_OWNER)
    if ledger.function_bodies.get(owner) != body.permuted_inputs(order):
        raise StructureMismatch("reused-function-definition-body-differs")
    outputs = tuple(ctx.lookup_value_for_var(var) for var in equation.outvars)
    if not outputs or outputs[0] is None or outputs[0].producer() is None:
        raise StructureUnavailable("function-call-output-unavailable")
    node = outputs[0].producer()
    origin = node.meta.setdefault(_ORIGIN, object())
    ledger.function_calls[handle] = FunctionCall(
        owner,
        origin,
        order,
        body,
        (definition.domain, definition.name, getattr(definition, "overload", "")),
    )
