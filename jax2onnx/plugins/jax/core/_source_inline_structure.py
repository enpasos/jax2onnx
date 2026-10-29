# jax2onnx/plugins/jax/core/_source_inline_structure.py

"""Original native JIT invocations in their existing physical parent graph.

The child body is the semantic authority. These observations add neither an
ONNX wrapper nor a numerical selection, and never produce a native trace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import onnx_ir as ir

from jax2onnx._compat.jax import (
    ClosedJaxpr,
    Jaxpr,
    is_empty_jit_mesh,
    is_unspecified_jit_sharding,
    native_jit_boolean_inline_hints,
    native_jit_primitive,
)
from jax2onnx.converter.source_structure import (
    ConstantContract,
    EquationHandle,
    RegionKey,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
    _OWNER,
    _ShapeFreezer,
    _ValueWitness,
    _resolve_bindings,
    _unused_inputs,
    _witness,
)

_JIT_FIELDS: frozenset[str] = frozenset(
    {
        "jaxpr",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "donated_invars",
        "ctx_mesh",
        "name",
        "keep_unused",
        "inline",
        "compiler_options_kvs",
    }
)


def native_inline_body(equation: Any) -> Any:
    """Read only the concrete body of the current finite plain-JIT schema.

    Admission still belongs to source_profile. Unknown schemas are unavailable
    observations; in particular a lazy body is never called to discover a tree.
    """
    if equation.primitive is not native_jit_primitive():
        return None
    params = equation.params
    if type(params) is not dict or set(params) != _JIT_FIELDS or equation.effects:
        return None
    closed = params["jaxpr"]
    if type(closed) is not ClosedJaxpr:
        return None
    body = closed.jaxpr  # On JAX 0.11+ this is self, not another unwrap step.
    if (
        type(body) is not Jaxpr
        or body.effects
        or getattr(body, "is_high", False)
        or type(closed.consts) not in (list, tuple)
        or len(body.constvars) != len(closed.consts)
        or len(body.invars) != len(equation.invars)
        or len(body.outvars) != len(equation.outvars)
    ):
        return None
    for prefix, count in (("in", len(equation.invars)), ("out", len(equation.outvars))):
        shardings, layouts = params[prefix + "_shardings"], params[prefix + "_layouts"]
        if (
            type(shardings) is not tuple
            or len(shardings) != count
            or not all(is_unspecified_jit_sharding(value) for value in shardings)
            or type(layouts) is not tuple
            or len(layouts) != count
            or not all(value is None for value in layouts)
        ):
            return None
    donated = params["donated_invars"]
    if (
        type(donated) is not tuple
        or len(donated) != len(equation.invars)
        or not all(type(value) is bool and not value for value in donated)
        or not isinstance(params["name"], str)
        or type(params["keep_unused"]) is not bool
        or not any(
            params["inline"] is hint for hint in native_jit_boolean_inline_hints()
        )
        or type(params["compiler_options_kvs"]) is not tuple
        or params["compiler_options_kvs"]
        or not is_empty_jit_mesh(params["ctx_mesh"])
    ):
        return None
    return closed


@dataclass(frozen=True)
class InlineRule:
    inline: bool
    keep_unused: bool

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("inline-attachment-unavailable")


def freeze_inline_rule(
    equation: Any,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> InlineRule | None:
    closed = native_inline_body(equation)
    if closed is None or any(
        value.symbol_token is not None for value in (*inputs, *outputs)
    ):
        return None
    shapes = _ShapeFreezer()
    body = closed.jaxpr
    if (
        tuple(TensorContract.from_aval(var.aval, shapes) for var in body.invars)
        != inputs
        or tuple(TensorContract.from_aval(var.aval, shapes) for var in body.outvars)
        != outputs
    ):
        return None
    hints = native_jit_boolean_inline_hints()
    return InlineRule(
        equation.params["inline"] is hints[1], equation.params["keep_unused"]
    )


@dataclass(frozen=True)
class InlineAttachment:
    owner: object
    region: RegionKey
    inputs: tuple[_ValueWitness, ...]
    outputs: tuple[_ValueWitness, ...]
    constants: tuple[_ValueWitness, ...]


_INLINE_EVIDENCE: WeakKeyDictionary[Any, dict[EquationHandle, InlineAttachment]] = (
    WeakKeyDictionary()
)


def authentic_inline_attachment(
    ledger: Any, handle: EquationHandle, attachment: Any
) -> bool:
    return (
        type(attachment) is InlineAttachment
        and _INLINE_EVIDENCE.get(ledger, {}).get(handle) is attachment
    )


def record_inline_attachment(
    ctx: Any,
    equation: Any,
    key: RegionKey,
    body: Any,
    operands: tuple[ir.Value, ...],
) -> None:
    """Capture only bindings already made by the generic JIT lowerer."""
    ledger = ctx._source_structure
    handle = ledger.handles[-1] if ledger.handles else None
    if (
        type(handle) is not EquationHandle
        or handle in ledger.inline_calls
        or type(key) is not RegionKey
        or key != RegionKey(handle, "inline-body")
        or key not in ledger.regions
    ):
        raise StructureUnavailable("inline-original-handle-unavailable")
    if (
        len(operands) != len(equation.invars)
        or len(operands) != len(body.invars)
        or any(not isinstance(value, ir.Value) for value in operands)
    ):
        raise StructureUnavailable("inline-original-input-binding-unavailable")
    attachment = InlineAttachment(
        ctx.builder.graph.meta[_OWNER],
        key,
        tuple(_witness(value) for value in operands),
        tuple(_witness(ctx.lookup_value_for_var(var)) for var in equation.outvars),
        tuple(_witness(ctx.lookup_value_for_var(var)) for var in body.constvars),
    )
    _INLINE_EVIDENCE.setdefault(ledger, {})[handle] = attachment
    ledger.inline_calls[handle] = attachment


@dataclass(frozen=True)
class InlineAudit:
    regions: tuple[Any, ...]
    inputs: tuple[ir.Value, ...]
    outputs: tuple[ir.Value, ...]
    constants: tuple[ir.Value, ...]
    adapters: tuple[ir.Node, ...]


def _constant_storage(value: ir.Value, graph: ir.Graph) -> tuple[Any, ir.Node | None]:
    producer = value.producer()
    if producer is None:
        if value.graph is not graph or not any(
            item is value for item in graph.initializers.values()
        ):
            raise StructureUnavailable("inline-constant-definition-unavailable")
        return value.const_value, None
    if (
        producer.graph is not graph
        or not any(node is producer for node in graph)
        or producer.domain
        or producer.overload
        or producer.op_type != "Constant"
        or producer.inputs
        or len(producer.outputs) != 1
        or producer.outputs[0] is not value
        or set(producer.attributes) != {"value"}
        or producer.attributes["value"].type is not ir.AttributeType.TENSOR
    ):
        raise StructureUnavailable("inline-constant-definition-unavailable")
    return producer.attributes["value"].as_tensor(), producer


def inline_region(
    ledger: Any,
    scopes: dict[object, Any],
    handle: EquationHandle,
    attachment: InlineAttachment,
) -> Any:
    """Validate an original ordered inline receipt without consulting final IR."""
    scope = scopes[handle.invocation]
    equation = scope.equations[handle.index]
    key = RegionKey(handle, "inline-body")
    if (
        type(attachment) is not InlineAttachment
        or type(attachment.region) is not RegionKey
        or type(attachment.region.parent) is not EquationHandle
        or any(
            type(bindings) is not tuple
            or any(type(witness) is not _ValueWitness for witness in bindings)
            for bindings in (
                attachment.inputs,
                attachment.outputs,
                attachment.constants,
            )
        )
        or attachment.region != key
        or attachment.owner is not scope.owner
        or key not in ledger.regions
    ):
        raise StructureUnavailable("inline-attachment-owner-or-role-unavailable")
    child = scopes.get(ledger.regions[key])
    if (
        child is None
        or child.owner is not scope.owner
        or child.region is None
        or child.region.key != key
        or child.body is None
    ):
        raise StructureUnavailable("inline-region-owner-unavailable")
    frozen_children = scope.body.equations[handle.index].children
    if (
        tuple(role for role, _ in frozen_children) != ("inline-body",)
        or child.body != frozen_children[0][1]
        or child.region.expected != child.body
        or child.body.gaps
    ):
        raise StructureMismatch("inline-original-child-body-differs")
    receipt = scope.receipts.get(handle.index)
    if receipt is None or child.result_values is None:
        raise StructureUnavailable("inline-original-receipt-unavailable")
    if (
        receipt.outputs != attachment.outputs
        or len(attachment.inputs) != len(receipt.inputs)
        or len(child.region.captures) != len(receipt.inputs)
        or len(attachment.outputs) != len(child.body.outputs)
        or len(attachment.constants) != len(child.body.constants)
        or attachment.constants != child.closed_constant_values
        or tuple(port.binding for port in child.region.captures) != receipt.inputs
    ):
        raise StructureMismatch("inline-original-binding-order-differs")
    if (
        type(equation.rule) is not InlineRule
        or type(equation.rule.inline) is not bool
        or type(equation.rule.keep_unused) is not bool
        or scope.body.equations[handle.index].semantics
        != ("inline-jit", equation.rule.inline, equation.rule.keep_unused)
    ):
        raise StructureMismatch("inline-original-settings-differ")
    unused = _unused_inputs(child.body)
    for index, (port, witness, use) in enumerate(
        zip(child.region.captures, attachment.inputs, equation.inputs, strict=True)
    ):
        if (
            port.outer != witness
            or port.local != witness
            or child.input_values[index] != witness
            or port.tensor != use.tensor
            or port.tensor != child.body.inputs[index]
            or type(port.unused) is not bool
            or port.unused != (index in unused)
        ):
            raise StructureMismatch("inline-region-input-order-or-type-differs")
    return child


def plan_inline_calls(
    ledger: Any,
    scopes: dict[object, Any],
    graphs: dict[object, ir.Graph],
    matches: Callable[[TensorContract, ir.Value], bool],
    *,
    projected_calls: dict[EquationHandle, InlineAttachment] | None = None,
    inactive_calls: frozenset[EquationHandle] = frozenset(),
) -> dict[EquationHandle, InlineAudit]:
    """Resolve active original bindings; literal unreachability is classified first."""
    wanted = {
        EquationHandle(scope.invocation, index)
        for scope in scopes.values()
        for index, equation in enumerate(scope.equations)
        if type(equation.rule) is InlineRule
    }
    if wanted != set(ledger.inline_calls):
        raise StructureUnavailable("inline-call-coverage-incomplete")
    projections = {} if projected_calls is None else projected_calls
    if (
        not set(projections) <= wanted
        or not inactive_calls <= wanted
        or set(projections) & inactive_calls
    ):
        raise StructureUnavailable("inline-literal-classification-unavailable")
    plans = {}
    for handle in wanted:
        scope = scopes[handle.invocation]
        equation = scope.equations[handle.index]
        attachment = projections.get(handle, ledger.inline_calls[handle])
        child = inline_region(ledger, scopes, handle, attachment)
        if handle in inactive_calls:
            # Complete registered source evidence was discharged by the exact
            # literal selector; there is deliberately no final graph lookup.
            continue
        graph = graphs[scope.owner]
        unused = _unused_inputs(child.body)
        optional = frozenset(
            witness
            for index, witness in enumerate(attachment.inputs)
            if index in unused
        )
        # The caller's receipt may still require an unused actual operand. If it
        # survives, it must be the very same final definition as the child port.
        ports = _resolve_bindings(graph, set(attachment.inputs), optional)
        resolved = _resolve_bindings(graph, set(attachment.outputs))
        actual_inputs = []
        for index, (port, witness, use) in enumerate(
            zip(child.region.captures, attachment.inputs, equation.inputs, strict=True)
        ):
            if (
                port.outer != witness
                or port.local != witness
                or child.input_values[index] != witness
                or port.tensor != use.tensor
                or port.tensor != child.body.inputs[index]
                or type(port.unused) is not bool
                or port.unused != (index in unused)
            ):
                raise StructureMismatch("inline-region-input-order-or-type-differs")
            actual = ports.get(witness)
            if actual is None:
                raise StructureUnavailable("inline-caller-input-witness-lost")
            if not matches(port.tensor, actual):
                raise StructureMismatch("inline-region-input-type-differs")
            actual_inputs.append(actual)
        used_constants = {
            use.index
            for use in (
                *child.body.outputs,
                *(use for eqn in child.body.equations for use in eqn.inputs),
            )
            if use.kind == "constant"
        }
        constant_values = _resolve_bindings(
            graph,
            set(attachment.constants),
            frozenset(
                witness
                for index, witness in enumerate(attachment.constants)
                if index not in used_constants
            ),
        )
        retained = []
        nodes = []
        for witness, (tensor, storage) in zip(
            attachment.constants, child.body.constants, strict=True
        ):
            if (
                ledger.constants.get(witness) != storage
                or tensor is None
                or storage is None
            ):
                raise StructureMismatch("inline-original-constant-order-differs")
            if witness not in child.variables.values():
                raise StructureMismatch("inline-original-constant-binding-differs")
            value = constant_values.get(witness)
            if value is None:
                continue
            actual_storage, node = _constant_storage(value, graph)
            if not matches(tensor, value):
                raise StructureMismatch("inline-constant-type-differs")
            if actual_storage is None:
                raise StructureUnavailable("inline-constant-storage-unavailable")
            if (
                actual_storage.dtype != tensor.dtype
                or tuple(actual_storage.shape) != tensor.shape
            ):
                raise StructureMismatch("inline-constant-declared-type-differs")
            if ConstantContract.from_array(actual_storage.numpy()) != storage:
                raise StructureMismatch("inline-constant-storage-differs")
            retained.append(value)
            if node is not None:
                nodes.append(node)
        plans[handle] = InlineAudit(
            (child,),
            tuple(actual_inputs),
            tuple(resolved[witness] for witness in attachment.outputs),
            tuple(retained),
            tuple(nodes),
        )
    return plans
