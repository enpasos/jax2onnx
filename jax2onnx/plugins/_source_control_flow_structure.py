# jax2onnx/plugins/_source_control_flow_structure.py

"""Original dynamic If contracts; neither source admission nor graph rewriting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, Literal
from jax2onnx.converter.source_structure import (
    ConstantContract,
    EquationHandle,
    LiteralOperand,
    RegionKey,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
    _ORIGIN,
    _OWNER,
    _ValueWitness,
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
class LiteralIfRule:
    """Original strong scalar selector; no final selector tensor is implied."""

    index: int
    selector: LiteralOperand

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("literal-if-attachment-unavailable")


@dataclass(frozen=True)
class LiteralIfBody:
    invocation: object
    owner: object
    body: Any
    equations: tuple[Any, ...]
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    input_values: tuple[Any, ...]
    closed_constant_values: tuple[_ValueWitness, ...]
    variables: tuple[tuple[int, Any], ...]
    receipts: tuple[tuple[int, Any], ...]
    result_values: tuple[Any, ...] | None
    region: Any


@dataclass(frozen=True)
class LiteralIfTree:
    """Complete ordered original scopes and inline calls in one staging owner."""

    scopes: tuple[LiteralIfBody, ...]
    inline_calls: tuple[tuple[EquationHandle, Any], ...]


@dataclass(frozen=True)
class LiteralIfDischarge:
    """Disjoint original obligations; unreachable bodies claim no physical IR."""

    selector: LiteralOperand
    inactive_role: str
    inactive_invocation: object
    inactive_body: Any
    graph_matched_equations: int
    semantically_discharged_equations: int
    inactive_nodes: tuple[()] = ()
    inactive_storage: tuple[()] = ()


@dataclass(frozen=True)
class LiteralIfAttachment:
    handle: EquationHandle
    owner: object
    rule: LiteralIfRule
    false: LiteralIfBody
    true: LiteralIfBody
    values: tuple[tuple[Any, Any], ...]
    node_origins: tuple[object, ...]
    node_ports: tuple[int, ...]
    original_returned: tuple[Any, ...]
    returned: tuple[Any, ...]
    results: tuple[Any, ...]
    discharge: LiteralIfDischarge
    trees: tuple[LiteralIfTree, ...] = ()


_LITERAL_IF_EVIDENCE: WeakKeyDictionary[
    Any, dict[EquationHandle, LiteralIfAttachment]
] = WeakKeyDictionary()


def authentic_literal_if_attachment(ledger: Any, attachment: Any) -> bool:
    """Only the original producer's exact immutable receipt has authority."""
    return (
        type(attachment) is LiteralIfAttachment
        and _LITERAL_IF_EVIDENCE.get(ledger, {}).get(attachment.handle) is attachment
    )


def literal_if_body(scope: Any) -> LiteralIfBody:
    return LiteralIfBody(
        scope.invocation,
        scope.owner,
        scope.body,
        scope.equations,
        scope.inputs,
        scope.outputs,
        scope.input_values,
        scope.closed_constant_values,
        tuple(sorted(scope.variables.items())),
        tuple(sorted(scope.receipts.items())),
        scope.result_values,
        scope.region,
    )


def _literal_leaf_semantics(rule: Any, semantics: tuple[object, ...] | None) -> bool:
    from jax2onnx.plugins._source_structure import _DirectRule, _PRIMITIVES
    from jax2onnx.plugins.jax.lax._source_dot_structure import _DotRule
    from jax2onnx.plugins.jax.lax._source_reduce_structure import _ReductionRule
    from jax2onnx.plugins.jax.lax._source_shape_structure import _ShapeRule

    if rule is None:
        # Preserve the existing pure source conversion key for unreachable
        # bodies. This adds no selected graph rule or native kernel authority.
        return semantics == (
            "native-f64-f16-convert",
            (("new_dtype", "float16"), ("sharding", None), ("weak_type", False)),
        )
    if type(rule) is BooleanIndexRule:
        return semantics == ("boolean-index",)
    if type(rule) is _DirectRule:
        # Compare an already frozen plugin key. Native selection remains the
        # original freezer's exact primitive identity, never a name lookup here.
        return (
            type(semantics) is tuple
            and len(semantics) == 5
            and semantics[0] == "direct"
            and semantics[1] in _PRIMITIVES
            and semantics[2:] == (rule.op_type, rule.attributes, rule.identity_alias)
        )
    if type(rule) is _DotRule:
        return semantics == ("dot_general", rule.axes, rule.precision, rule.preferred)
    if type(rule) is _ReductionRule:
        return semantics == ("reduction", rule.operation, rule.axes)
    if type(rule) is _ShapeRule:
        return semantics == (
            "shape",
            rule.operation,
            rule.permutation,
            rule.broadcast_dimensions,
        )
    return False


def _literal_scope_bindings(ledger: Any, scope: Any) -> None:
    """Check original frozen SSA and exact staged receipts, without final IR."""
    from jax2onnx.converter._source_regions import FrozenBody
    from jax2onnx.converter.source_structure import CapturePort, RegionReceipt, _Receipt

    body = scope.body
    if (
        type(body) is not FrozenBody
        or body.gaps
        or body.expected_equations is None
        or len(scope.equations) != len(body.equations)
        or tuple(use.tensor for use in scope.inputs) != body.inputs
        or tuple(use.tensor for use in scope.outputs)
        != tuple(use.tensor for use in body.outputs)
        or len(scope.input_values) != len(scope.inputs)
        or len(scope.closed_constant_values) != len(body.constants)
        or set(scope.receipts) != set(range(len(scope.equations)))
        or scope.result_values is None
        or len(scope.result_values) != len(body.outputs)
        or scope.pending
        or scope.cast_plan is not None
        or scope.cast_selection is not None
        or type(scope.region) is not RegionReceipt
        or scope.region.expected != body
        or len(scope.region.captures) != len(scope.inputs)
    ):
        raise StructureUnavailable("literal-if-childless-contract-unavailable")
    variables = scope.variables
    known_ids = {use.variable_id for use in scope.inputs}
    for use, witness in zip(scope.inputs, scope.input_values, strict=True):
        if variables.get(use.variable_id) != witness:
            raise StructureMismatch("literal-if-original-formal-binding-differs")
    for witness, (tensor, storage) in zip(
        scope.closed_constant_values, body.constants, strict=True
    ):
        if (
            tensor is None
            or storage is None
            or ledger.constants.get(witness) != storage
        ):
            raise StructureMismatch("literal-if-original-constant-binding-differs")
    for port, use, witness in zip(
        scope.region.captures, scope.inputs, scope.input_values, strict=True
    ):
        if (
            type(port) is not CapturePort
            or port.local != witness
            or port.tensor != use.tensor
        ):
            raise StructureMismatch("literal-if-original-capture-binding-differs")

    def frozen_binding(use: Any) -> Any:
        if (
            type(use.index) is not int
            or use.index < 0
            or type(use.port) is not int
            or use.port < 0
            or type(use.tensor) is not TensorContract
        ):
            raise StructureMismatch("literal-if-original-use-coordinate-differs")
        if use.kind == "literal":
            if use.index != 0 or use.port != 0:
                raise StructureMismatch("literal-if-original-use-coordinate-differs")
            if use.tensor is None or use.literal is None:
                raise StructureUnavailable("literal-if-original-literal-unavailable")
            return LiteralOperand(use.tensor, use.literal)
        if use.kind == "input":
            if use.port != 0 or use.index >= len(scope.input_values):
                raise StructureMismatch("literal-if-original-use-coordinate-differs")
            return scope.input_values[use.index]
        if use.kind == "constant":
            if use.port != 0 or use.index >= len(scope.closed_constant_values):
                raise StructureMismatch("literal-if-original-use-coordinate-differs")
            return scope.closed_constant_values[use.index]
        if use.kind == "equation":
            if use.index >= len(scope.equations) or use.port >= len(
                scope.receipts[use.index].outputs
            ):
                raise StructureMismatch("literal-if-original-use-coordinate-differs")
            return scope.receipts[use.index].outputs[use.port]
        raise StructureUnavailable("literal-if-original-use-unavailable")

    for index, (equation, frozen) in enumerate(
        zip(scope.equations, body.equations, strict=True)
    ):
        receipt = scope.receipts[index]
        if (
            type(receipt) is not _Receipt
            or tuple(use.tensor for use in equation.inputs)
            != tuple(use.tensor for use in frozen.inputs)
            or tuple(use.tensor for use in equation.outputs) != frozen.outputs
            or len(receipt.outputs) != len(frozen.outputs)
            or receipt.inputs != tuple(frozen_binding(use) for use in frozen.inputs)
        ):
            raise StructureMismatch("literal-if-original-equation-ports-differ")
        for use, frozen_use, binding in zip(
            equation.inputs, frozen.inputs, receipt.inputs, strict=True
        ):
            if use.literal != frozen_use.literal or (
                use.literal is None and variables.get(use.variable_id) != binding
            ):
                raise StructureMismatch("literal-if-original-use-binding-differs")
            if frozen_use.kind == "equation" and frozen_use.index >= index:
                raise StructureMismatch("literal-if-original-producer-order-differs")
        for use, witness in zip(equation.outputs, receipt.outputs, strict=True):
            if (
                variables.get(use.variable_id) != witness
                or use.variable_id in known_ids
            ):
                raise StructureMismatch("literal-if-original-result-binding-differs")
            known_ids.add(use.variable_id)
    if Counter(
        value for key, value in variables.items() if key not in known_ids
    ) != Counter(scope.closed_constant_values):
        raise StructureMismatch("literal-if-original-constvar-coverage-differs")
    if scope.result_values != tuple(frozen_binding(use) for use in body.outputs):
        raise StructureMismatch("literal-if-original-return-binding-differs")
    for use, binding in zip(scope.outputs, scope.result_values, strict=True):
        if (
            use.literal is not None
            and binding != LiteralOperand(use.tensor, use.literal)
        ) or (use.literal is None and variables.get(use.variable_id) != binding):
            raise StructureMismatch("literal-if-original-return-use-differs")
    witnesses = (
        *scope.input_values,
        *scope.closed_constant_values,
        *variables.values(),
        *(
            witness
            for receipt in scope.receipts.values()
            for witness in receipt.outputs
        ),
    )
    if any(
        type(witness) is not _ValueWitness or witness.owner is not scope.owner
        for witness in witnesses
    ):
        raise StructureMismatch("literal-if-original-witness-owner-differs")


def literal_if_tree(ledger: Any, root: Any) -> LiteralIfTree:
    """Enumerate precisely the pure, same-owner, native inline invocation tree."""
    from jax2onnx.plugins.jax.core._source_inline_structure import (
        InlineRule,
        authentic_inline_attachment,
        inline_region,
    )

    scopes = {scope.invocation: scope for scope in ledger.scopes}
    if len(scopes) != len(ledger.scopes):
        raise StructureUnavailable("literal-if-original-invocation-duplicated")
    ordered = []
    calls = []
    seen: set[object] = set()

    def visit(scope: Any) -> int:
        if scope.invocation in seen or scope.owner is not root.owner:
            raise StructureUnavailable("literal-if-original-inline-tree-unavailable")
        seen.add(scope.invocation)
        _literal_scope_bindings(ledger, scope)
        ordered.append(literal_if_body(scope))
        total = len(scope.equations)
        for index, (equation, frozen) in enumerate(
            zip(scope.equations, scope.body.equations, strict=True)
        ):
            handle = EquationHandle(scope.invocation, index)
            if type(equation.rule) is InlineRule:
                if handle not in ledger.inline_calls:
                    raise StructureUnavailable(
                        "literal-if-original-inline-call-unavailable"
                    )
                attachment = ledger.inline_calls[handle]
                if not authentic_inline_attachment(ledger, handle, attachment):
                    raise StructureUnavailable(
                        "literal-if-original-inline-receipt-unavailable"
                    )
                child = inline_region(ledger, scopes, handle, attachment)
                for use, witness in zip(
                    equation.inputs, attachment.inputs, strict=True
                ):
                    if (
                        use.literal is not None
                        and ledger.constants.get(witness) != use.literal
                    ):
                        raise StructureMismatch(
                            "literal-if-original-inline-literal-differs"
                        )
                calls.append((handle, attachment))
                total += visit(child)
            elif frozen.children or not _literal_leaf_semantics(
                equation.rule, frozen.semantics
            ):
                # Preserve the established explicit unfinished-Function reason.
                raise StructureUnavailable("literal-if-childless-contract-unavailable")
        if total != scope.body.expected_equations:
            raise StructureMismatch("literal-if-original-inline-count-differs")
        return total

    visit(root)
    wanted = {RegionKey(handle, "inline-body") for handle, _ in calls}
    observed = {key for key in ledger.regions if key.parent.invocation in seen}
    if observed != wanted or {
        handle for handle in ledger.inline_calls if handle.invocation in seen
    } != {handle for handle, _ in calls}:
        raise StructureUnavailable("literal-if-original-inline-coverage-incomplete")
    return LiteralIfTree(tuple(ordered), tuple(calls))


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
) -> BooleanIndexRule | IfRule | LiteralIfRule | None:
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
        or type(equation.params) is not dict
        or set(equation.params) != {"branches"}
        or inputs[0].logical_dtype != "int32"
        or inputs[0].dtype is not ir.DataType.INT32
        or inputs[0].weak_type is not False
        or inputs[0].shape != ()
        or inputs[0].logical_shape != ()
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
    if isinstance(equation.invars[0], Literal):
        selector = equation.invars[0]
        if type(selector.val) is int:
            index = selector.val
        else:
            payload = np.asarray(selector.val)
            if payload.dtype != np.dtype(np.int32) or payload.shape != ():
                return None
            index = int(payload.item())
        if index not in (0, 1):
            return None
        payload = np.asarray(index, dtype=np.int32)
        return LiteralIfRule(
            index, LiteralOperand(inputs[0], ConstantContract.from_array(payload))
        )
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


def prepare_literal_if_clone(ctx: Any, graph: ir.Graph) -> None:
    """Stamp actual staged definitions before the existing numerical clone."""
    from jax2onnx.converter.source_structure import _witness

    for node in graph:
        for value in node.outputs:
            _witness(value)


def record_literal_if_attachment(
    ctx: Any,
    equation: Any,
    graph: ir.Graph,
    nodes: tuple[ir.Node, ...],
    returned: tuple[ir.Value, ...],
) -> None:
    """Observe the producer's actual clone map after its existing attachment."""
    from jax2onnx.converter.source_structure import _witness

    ledger = ctx._source_structure
    handle = ledger.handles[-1] if ledger.handles else None
    if type(handle) is not EquationHandle or handle in ledger.literal_if_calls:
        raise StructureUnavailable("literal-if-handle-unavailable")
    parent = next(
        (
            item
            for item in ledger.active
            if item is not None and item.invocation is handle.invocation
        ),
        None,
    )
    if parent is None:
        raise StructureUnavailable("literal-if-parent-unavailable")
    rule = parent.equations[handle.index].rule
    if type(rule) is not LiteralIfRule:
        raise StructureUnavailable("literal-if-rule-unavailable")
    bodies = []
    trees = []
    for role in ("false", "true"):
        invocation = ledger.regions.get(RegionKey(handle, role))
        matches = [scope for scope in ledger.scopes if scope.invocation is invocation]
        if len(matches) != 1:
            raise StructureUnavailable("literal-if-original-body-unavailable")
        scope = matches[0]
        tree = literal_if_tree(ledger, scope)
        trees.append(tree)
        bodies.append(tree.scopes[0])
    if bodies[0].owner is bodies[1].owner or {
        scope.invocation for scope in trees[0].scopes
    } & {scope.invocation for scope in trees[1].scopes}:
        raise StructureUnavailable("literal-if-original-branches-overlap")
    selected, inactive = bodies[rule.index], bodies[1 - rule.index]
    if graph.meta.get(_OWNER) is not selected.owner:
        raise StructureMismatch("literal-if-selected-owner-differs")
    originals = tuple(graph)
    if len(originals) != len(nodes) or len(graph.outputs) != len(returned):
        raise StructureMismatch("literal-if-selected-clone-arity-differs")
    mapping = []
    for original, clone in zip(originals, nodes, strict=True):
        if (
            original.meta.get(_ORIGIN) is not clone.meta.get(_ORIGIN)
            or clone.graph is not ctx.builder.graph
            or len(original.outputs) != len(clone.outputs)
        ):
            raise StructureMismatch("literal-if-selected-clone-origin-differs")
        for before, after in zip(original.outputs, clone.outputs, strict=True):
            source, target = _witness(before), _witness(after)
            mapping.append((source, target))
            if source in ledger.constants:
                if target in ledger.constants:
                    raise StructureUnavailable("literal-if-constant-replay-unavailable")
                ledger.constants[target] = ledger.constants[source]
    values = dict(mapping)
    original_returns = tuple(_witness(value) for value in graph.outputs)
    actual_returns = tuple(_witness(value) for value in returned)
    if any(
        values.get(before) != after
        for before, after in zip(original_returns, actual_returns, strict=True)
    ):
        raise StructureMismatch("literal-if-selected-result-map-differs")
    results = tuple(_witness(ctx.lookup_value_for_var(var)) for var in equation.outvars)
    attachment = LiteralIfAttachment(
        handle,
        ctx.builder.graph.meta[_OWNER],
        rule,
        bodies[0],
        bodies[1],
        tuple(mapping),
        tuple(node.meta[_ORIGIN] for node in nodes),
        tuple(len(node.outputs) for node in originals),
        original_returns,
        actual_returns,
        results,
        LiteralIfDischarge(
            rule.selector,
            "true" if rule.index == 0 else "false",
            inactive.invocation,
            inactive.body,
            selected.body.expected_equations,
            1 + inactive.body.expected_equations,
        ),
        tuple(trees),
    )
    _LITERAL_IF_EVIDENCE.setdefault(ledger, {})[handle] = attachment
    ledger.literal_if_calls[handle] = attachment
