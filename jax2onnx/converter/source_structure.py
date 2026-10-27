# jax2onnx/converter/source_structure.py

"""Private compositional SOURCE structure observations, independent of numerics.

The dispatcher freezes the actual source equations before invoking plugins.
Rules owned by plugins check the final IR, after optimization and facade changes.
Original region contracts precede dispatch; definition reuse retains every call
obligation. Missing rules, attachments or clone witnesses keep the audit
unverified; observation never rejects an export.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Any, Final, Literal, Protocol, TYPE_CHECKING, TypeAlias, cast

import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal as JaxLiteral
from .source_representation import (
    native_shape_contract,
    normalized_shape_dimensions,
    representation_for_aval,
)

if TYPE_CHECKING:
    from jax2onnx.plugins.jax.lax._source_loop_structure import WhileAudit

    from .typing_support import ConverterContext

_OWNER: Final = "jax2onnx.source.structure.owner"
_ORIGIN: Final = "jax2onnx.source.structure.origin"
_VALUES: Final = "jax2onnx.source.structure.values"
_LEDGER: Final = "jax2onnx.source.structure.ledger"
RESULT_KEY: Final = "jax2onnx.source.structure.result"


class StructureUnavailable(ValueError):
    """A required observation or reviewed rule is unavailable."""


class StructureMismatch(ValueError):
    """Observed final structure contradicts a frozen source contract."""


class _ShapeFreezer:
    """Transient native authority; only its opaque token survives source freeze."""

    def __init__(self) -> None:
        self._native_scope: object | None = None
        self._label: str | None = None
        self._token: object | None = None

    def freeze(
        self, shape: Sequence[object]
    ) -> tuple[tuple[int | str, ...], object | None, int | None]:
        try:
            contract = native_shape_contract(shape)
        except ValueError as exc:
            raise StructureUnavailable("symbolic-shape-contract-unavailable") from exc
        if contract.symbol is None:
            return contract.shape, None, None
        label, native_scope = contract.symbol
        if self._native_scope is None:
            self._native_scope, self._label, self._token = native_scope, label, object()
        elif native_scope is not self._native_scope or label != self._label:
            raise StructureUnavailable("symbolic-scope-contract-unavailable")
        return contract.shape, self._token, contract.symbol_axis


@dataclass(frozen=True)
class TensorContract:
    logical_dtype: str
    logical_shape: tuple[int | str, ...]
    weak_type: bool
    dtype: ir.DataType
    shape: tuple[int | str, ...]
    encoding: str
    symbol_token: object | None
    logical_symbol_axis: int | None
    symbol_axis: int | None

    @classmethod
    def from_aval(cls, aval: Any, shapes: _ShapeFreezer) -> TensorContract:
        representation = representation_for_aval(aval)
        logical, logical_token, logical_axis = shapes.freeze(
            representation.logical_shape
        )
        physical, physical_token, physical_axis = shapes.freeze(
            representation.physical_shape
        )
        if logical_token is not physical_token:
            raise StructureUnavailable("symbolic-representation-contract-unavailable")
        return cls(
            str(representation.logical_dtype),
            logical,
            bool(aval.weak_type),
            representation.ir_dtype,
            physical,
            representation.encoding,
            physical_token,
            logical_axis,
            physical_axis,
        )

    def matches(self, value: ir.Value) -> bool:
        if value.shape is not None and any(
            not isinstance(dim, (int, np.integer, str, ir.SymbolicDim))
            or isinstance(dim, (bool, np.bool_))
            for dim in value.shape
        ):
            # Final IR labels describe a frozen native relationship. Do not
            # decode another native Scope/DimExpr through a string here.
            raise StructureUnavailable("ir-shape-representation-unavailable")
        return (
            value.dtype == self.dtype
            and value.shape is not None
            and normalized_shape_dimensions(tuple(value.shape)) == self.shape
        )

    def same_shape(self, other: TensorContract) -> bool:
        return (
            self.shape == other.shape
            and self.symbol_token is other.symbol_token
            and self.symbol_axis == other.symbol_axis
        )


@dataclass(frozen=True)
class ConstantContract:
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str

    @classmethod
    def from_array(cls, array: np.ndarray[Any, Any]) -> ConstantContract:
        # Exact original storage: signed zero and NaN payloads are not normalized.
        payload = (
            memoryview(array).cast("B")
            if array.flags.c_contiguous
            else memoryview(array.tobytes(order="C"))
        )
        return cls(
            array.dtype.str,
            tuple(array.shape),
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )


@dataclass(frozen=True)
class LiteralOperand:
    """Original typed literal storage, without an emitted-definition claim."""

    tensor: TensorContract
    storage: ConstantContract


RuleOperand: TypeAlias = ir.Value | LiteralOperand
LiteralMatcher: TypeAlias = Callable[[ir.Value | None, LiteralOperand], None]


@dataclass(frozen=True)
class _ValueWitness:
    owner: object
    origin: object
    role: Literal["input", "initializer", "producer"]
    port: int


_Binding: TypeAlias = _ValueWitness | LiteralOperand


def _location(
    value: ir.Value,
) -> tuple[
    ir.Graph, ir.Value | ir.Node, Literal["input", "initializer", "producer"], int
]:
    graph = value.graph
    if graph is None:
        raise StructureUnavailable("value-owner-unavailable")
    producer = value.producer()
    if producer is not None:
        port = value.index()
        if (
            producer.graph is not graph
            or not any(node is producer for node in graph)
            or type(port) is not int
            or port < 0
            or port >= len(producer.outputs)
            or producer.outputs[port] is not value
        ):
            raise StructureUnavailable("producer-port-unavailable")
        return graph, producer, "producer", port
    for port, candidate in enumerate(graph.inputs):
        if candidate is value:
            return graph, value, "input", port
    if any(candidate is value for candidate in graph.initializers.values()):
        return graph, value, "initializer", 0
    raise StructureUnavailable("definition-unavailable")


def _witness(value: ir.Value) -> _ValueWitness:
    graph, definition, role, port = _location(value)
    owner = graph.meta.setdefault(_OWNER, object())
    origin = definition.meta.setdefault(_ORIGIN, object())
    if type(owner) is not object or type(origin) is not object:
        raise StructureUnavailable("definition-marker-unavailable")
    witness = _ValueWitness(owner, origin, role, port)
    previous = value.meta.get(_VALUES, ())
    if type(previous) is not tuple or any(
        type(item) is not _ValueWitness for item in previous
    ):
        raise StructureUnavailable("value-marker-unavailable")
    if witness not in previous:
        value.meta[_VALUES] = (*previous, witness)
    return witness


class EquationRule(Protocol):
    """Plugin-owned semantics frozen before the plugin emits its graph."""

    def check(
        self,
        graph: ir.Graph,
        inputs: tuple[RuleOperand, ...],
        outputs: tuple[ir.Value, ...],
        match_literal: LiteralMatcher,
    ) -> tuple[ir.Node, ...]: ...


@dataclass(frozen=True)
class _Use:
    variable_id: int
    tensor: TensorContract
    literal: ConstantContract | None = None


def _use(var: Any, shapes: _ShapeFreezer) -> _Use:
    literal = None
    if isinstance(var, JaxLiteral):
        # These conversions observe already-traced storage. Production binding
        # retains ownership of any cast warning; observation must not duplicate it.
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(var.val, dtype=var.aval.dtype)
            literal = ConstantContract.from_array(
                representation_for_aval(var.aval).encode_array(array)
            )
    return _Use(id(var), TensorContract.from_aval(var.aval, shapes), literal)


@dataclass(frozen=True)
class _Equation:
    source_id: int
    inputs: tuple[_Use, ...]
    outputs: tuple[_Use, ...]
    rule: EquationRule | None


@dataclass(frozen=True)
class _Receipt:
    inputs: tuple[_Binding, ...]
    outputs: tuple[_ValueWitness, ...]


@dataclass(frozen=True)
class EquationHandle:
    """Original equation authority, independent of a plugin's local replace()."""

    invocation: object
    index: int


@dataclass(frozen=True)
class RegionKey:
    """An expected invocation role, independent of its physical graph owner."""

    parent: EquationHandle
    role: str


@dataclass(frozen=True)
class CapturePort:
    binding: _Binding
    outer: _ValueWitness
    local: _ValueWitness
    tensor: TensorContract
    unused: bool


@dataclass(frozen=True)
class RegionReceipt:
    key: RegionKey
    expected: Any
    captures: tuple[CapturePort, ...]


@dataclass
class _PendingRegion:
    key: RegionKey
    expected: Any
    bindings: tuple[_Binding, ...]
    outer: tuple[_ValueWitness, ...]
    claimed: bool = False


def _unused_inputs(body: Any) -> frozenset[int]:
    if body.gaps:
        raise StructureUnavailable("region-use-def-contract-unavailable")
    used = {
        use.index
        for equation in body.equations
        for use in equation.inputs
        if use.kind == "input"
    }
    used.update(use.index for use in body.outputs if use.kind == "input")
    return frozenset(set(range(len(body.inputs))) - used)


@dataclass
class _Scope:
    invocation: object
    owner: object
    equations: tuple[_Equation, ...]
    inputs: tuple[_Use, ...]
    outputs: tuple[_Use, ...]
    input_values: tuple[_ValueWitness, ...]
    variables: dict[int, _ValueWitness]
    body: Any = None
    pending: dict[int, tuple[_Binding, ...]] = field(default_factory=dict)
    receipts: dict[int, _Receipt] = field(default_factory=dict)
    result_values: tuple[_Binding, ...] | None = None
    region: RegionReceipt | None = None
    cast_plan: Any = None
    cast_selection: Any = None


@dataclass(frozen=True)
class StructureResult:
    status: Literal["validated", "failed", "unverified"]
    expected_equations: int | None
    checked_equations: int
    reasons: tuple[str, ...]
    runtime: Literal["unverified"] = "unverified"
    numerics: Literal["unverified"] = "unverified"


class SourceStructureLedger:
    """One export's real source-to-IR bindings, shared by its child contexts."""

    def __init__(self) -> None:
        self.scopes: list[_Scope] = []
        self.active: list[_Scope | None] = []
        self.constants: dict[_ValueWitness, ConstantContract] = {}
        self.gaps: set[str] = set()
        self.failures: set[str] = set()
        self.expected_equations: int | None = 0
        self.handles: list[EquationHandle | None] = []
        self.function_bodies: dict[object, Any] = {}
        self.function_calls: dict[EquationHandle, Any] = {}
        self.if_calls: dict[EquationHandle, Any] = {}
        self.while_calls: dict[EquationHandle, Any] = {}
        self.cast_plans: dict[EquationHandle, Any] = {}
        self.cast_selections: dict[EquationHandle, Any] = {}
        self.regions: dict[RegionKey, object] = {}
        self.pending_regions: list[_PendingRegion | None] = []
        self.root_body: Any = None

    @contextmanager
    def region(
        self, role: str, operands: Sequence[ir.Value]
    ) -> Iterator[RegionKey | None]:
        """Bracket only the already-planned lowering of one original child body."""

        def begin() -> _PendingRegion:
            from jax2onnx.plugins._source_control_flow_structure import IfRule
            from jax2onnx.plugins.jax.lax._source_loop_structure import WhileRule

            handle = self.handles[-1] if self.handles else None
            if handle is None:
                raise StructureUnavailable("region-parent-handle-unavailable")
            scope = next(
                (
                    item
                    for item in self.active
                    if item is not None and item.invocation is handle.invocation
                ),
                None,
            )
            if scope is None:
                raise StructureUnavailable("region-parent-contract-unavailable")
            rule = scope.equations[handle.index].rule
            if type(rule) not in (IfRule, WhileRule):
                raise StructureUnavailable("region-parent-contract-unavailable")
            expected = dict(scope.body.equations[handle.index].children).get(role)
            if expected is None or expected.gaps:
                raise StructureUnavailable("original-region-body-unavailable")
            key = RegionKey(handle, role)
            if key in self.regions or handle.index not in scope.pending:
                raise StructureUnavailable("region-identity-unavailable")
            original = scope.pending[handle.index]
            if type(rule) is IfRule:
                bindings = original[1:]
            elif type(rule) is WhileRule:
                bindings = tuple(
                    original[port] for port in rule.input_ports(role, len(original))
                )
                if role == "next-condition":
                    body_invocation = self.regions.get(RegionKey(handle, "body"))
                    body_scope = next(
                        (
                            item
                            for item in self.scopes
                            if item.invocation is body_invocation
                        ),
                        None,
                    )
                    if body_scope is None or body_scope.result_values is None:
                        raise StructureUnavailable(
                            "while-original-body-results-unavailable"
                        )
                    bindings = (
                        *original[: rule.condition_constants],
                        *body_scope.result_values,
                    )
            else:
                raise StructureUnavailable("region-parent-contract-unavailable")
            if len(bindings) != len(operands) or len(expected.inputs) != len(operands):
                raise StructureMismatch("region-capture-arity-differs")
            return _PendingRegion(
                key, expected, bindings, tuple(_witness(value) for value in operands)
            )

        pending = self._observe("region-capture", begin)
        self.pending_regions.append(pending)
        try:
            yield pending.key if pending is not None else None
        finally:
            if pending is not None and not pending.claimed:
                self.gaps.add("region-scope-unobserved")
            self.pending_regions.pop()

    @contextmanager
    def dispatch(self, eqn: JaxprEqn, index: int) -> Iterator[None]:
        def handle() -> EquationHandle:
            scope, _ = self._expected(eqn, index)
            original = EquationHandle(scope.invocation, index)
            if scope.cast_plan is not None:
                if (
                    index != scope.cast_plan.eligibility.root_ordinal
                    or original in self.cast_plans
                ):
                    raise StructureUnavailable("cast-plan-equation-handle-unavailable")
                self.cast_plans[original] = scope.cast_plan
            if scope.cast_selection is not None:
                if index != 0 or original in self.cast_selections:
                    raise StructureUnavailable(
                        "selected-cast-equation-handle-unavailable"
                    )
                self.cast_selections[original] = scope.cast_selection
            return original

        self.handles.append(self._observe("equation-handle", handle))
        try:
            yield
        finally:
            self.handles.pop()

    def function_body(
        self, graph: ir.Graph, original: Any, order: tuple[int, ...], actual: Any
    ) -> None:
        from ._source_regions import FrozenBody

        if type(original) is not FrozenBody or type(actual) is not FrozenBody:
            raise StructureUnavailable("original-function-body-unavailable")
        if original.gaps or actual.gaps:
            raise StructureUnavailable("original-function-body-contract-unavailable")
        expected = original.permuted_inputs(order)
        if expected != actual:
            raise StructureMismatch("original-function-body-differs")
        owner = graph.meta.setdefault(_OWNER, object())
        if owner in self.function_bodies:
            raise StructureUnavailable("function-definition-reused-owner")
        self.function_bodies[owner] = actual

    def _observe(self, stage: str, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except StructureMismatch as exc:
            self.failures.add(stage + ":" + str(exc))
            return None
        except StructureUnavailable as exc:
            self.gaps.add(stage + ":" + str(exc))
            return None
        except Exception as exc:
            # Bounded, value-free diagnostics; never turn verifier gaps into a
            # new source export rejection or echo a tensor payload in an error.
            self.gaps.add(stage + ":" + type(exc).__name__)
            return None

    def constant(self, value: ir.Value, original_storage: np.ndarray[Any, Any]) -> None:
        def record() -> None:
            witness = _witness(value)
            if witness not in self.constants:
                self.constants[witness] = ConstantContract.from_array(original_storage)

        self._observe("constant-capture", record)

    @contextmanager
    def scope(
        self,
        ctx: ConverterContext,
        jaxpr: Jaxpr,
        freeze_rule: Callable[
            [JaxprEqn, tuple[TensorContract, ...], tuple[TensorContract, ...]],
            EquationRule | None,
        ],
    ) -> Iterator[None]:
        if not self.scopes and not self.active:
            # Root capture establishes the denominator before any child plugin
            # dispatch. A reused definition cannot remove expected invocations.
            from ._source_regions import declare_body

            self.root_body = declare_body(jaxpr)
            self.expected_equations = self.root_body.expected_equations
            if self.expected_equations is None:
                self.gaps.add("expected-region-total-unknown")

        def begin() -> _Scope:
            graph = ctx.builder.graph
            owner = graph.meta.setdefault(_OWNER, object())
            if type(owner) is not object or jaxpr.effects:
                raise StructureUnavailable("scope-contract-unavailable")
            shapes = _ShapeFreezer()
            inputs = tuple(_use(var, shapes) for var in jaxpr.invars)
            # Closed constants also participate in native shape provenance,
            # including a source constant unused by the retained equations.
            for var in jaxpr.constvars:
                _use(var, shapes)

            def freeze_equation(eqn: JaxprEqn) -> _Equation:
                from ._source_cast_authority import _selected_cast_rule

                operands = tuple(_use(var, shapes) for var in eqn.invars)
                results = tuple(_use(var, shapes) for var in eqn.outvars)
                selected = _selected_cast_rule(ctx, eqn)
                return _Equation(
                    id(eqn),
                    operands,
                    results,
                    (
                        selected
                        if selected is not None
                        else (
                            freeze_rule(
                                eqn,
                                tuple(use.tensor for use in operands),
                                tuple(use.tensor for use in results),
                            )
                            if not eqn.effects
                            else None
                        )
                    ),
                )

            equations = tuple(freeze_equation(eqn) for eqn in jaxpr.eqns)
            outputs = tuple(_use(var, shapes) for var in jaxpr.outvars)
            values = tuple(_witness(_lookup(ctx, var)) for var in jaxpr.invars)
            variables = {
                id(var): _witness(_lookup(ctx, var))
                for var in (*jaxpr.constvars, *jaxpr.invars)
            }
            for var in jaxpr.constvars:
                if variables[id(var)] not in self.constants:
                    raise StructureUnavailable("source-constant-capture-unavailable")
            from ._source_regions import freeze_body

            body = freeze_body(
                jaxpr,
                constant_contracts=tuple(
                    self.constants[variables[id(var)]] for var in jaxpr.constvars
                ),
            )
            if not self.scopes and not self.active:
                if body.expected_equations != self.expected_equations:
                    raise StructureUnavailable("source-region-declaration-differs")
                if any(equation.children for equation in body.equations) and body.gaps:
                    self.gaps.update("original-regions:" + gap for gap in body.gaps)
            scope = _Scope(
                object(),
                owner,
                equations,
                inputs,
                outputs,
                values,
                variables,
                body,
            )
            from jax2onnx.plugins.jax._source_f64_f16_structure import (
                SelectedF64F16Rule,
            )

            selections = tuple(
                equation.rule
                for equation in equations
                if type(equation.rule) is SelectedF64F16Rule
            )
            if selections:
                if (
                    self.scopes
                    or self.active
                    or len(equations) != 1
                    or len(selections) != 1
                ):
                    raise StructureUnavailable("selected-cast-root-scope-differs")
                scope.cast_selection = selections[0]
            if (
                not self.scopes
                and not self.active
                and ctx._source_cast_plan is not None
            ):
                from ._source_cast_planning import plan_is_current

                plan = ctx._source_cast_plan
                if not plan_is_current(plan):
                    raise StructureUnavailable("cast-plan-contract-unavailable")
                if plan.eligibility is not None:
                    if plan.eligibility.body != body:
                        raise StructureMismatch("cast-plan-original-body-differs")
                    scope.cast_plan = plan
            pending = self.pending_regions[-1] if self.pending_regions else None
            if pending is not None and not pending.claimed:
                if body.gaps:
                    raise StructureUnavailable("original-region-contract-unavailable")
                if body != pending.expected:
                    raise StructureMismatch("original-region-body-differs")
                unused = _unused_inputs(pending.expected)
                scope.region = RegionReceipt(
                    pending.key,
                    pending.expected,
                    tuple(
                        CapturePort(binding, outer, local, use.tensor, index in unused)
                        for index, (binding, outer, local, use) in enumerate(
                            zip(
                                pending.bindings,
                                pending.outer,
                                values,
                                inputs,
                                strict=True,
                            )
                        )
                    ),
                )
                pending.claimed = True
                self.regions[pending.key] = scope.invocation
            self.scopes.append(scope)
            return scope

        scope = self._observe("scope-capture", begin)
        self.active.append(scope)
        try:
            yield
        finally:
            if scope is not None:

                def finish() -> None:
                    # Literal root results need not have an IR definition yet:
                    # ordinary output binding materializes them after this scope.
                    actual = tuple(
                        _capture_binding(ctx, use, var)
                        for use, var in zip(scope.outputs, jaxpr.outvars, strict=True)
                    )
                    for use, value in zip(scope.outputs, actual, strict=True):
                        self._check_source_use(scope, use, value)
                    scope.result_values = actual

                self._observe("scope-results", finish)
            self.active.pop()

    def _check_source_use(self, scope: _Scope, use: _Use, value: _Binding) -> None:
        if use.literal is not None:
            if value != LiteralOperand(use.tensor, use.literal):
                raise StructureMismatch("literal-source-binding-differs")
        elif scope.variables.get(use.variable_id) != value:
            raise StructureMismatch("source-ssa-binding-differs")

    def before_equation(self, ctx: ConverterContext, eqn: JaxprEqn, index: int) -> None:
        def record() -> None:
            scope, expected = self._expected(eqn, index)
            if index in scope.pending or index in scope.receipts:
                raise StructureUnavailable("equation-identity-unavailable")
            if len(eqn.invars) != len(expected.inputs):
                raise StructureMismatch("source-input-arity-differs")
            inputs = tuple(
                _capture_binding(ctx, use, var)
                for use, var in zip(expected.inputs, eqn.invars, strict=True)
            )
            for use, var, value in zip(
                expected.inputs, eqn.invars, inputs, strict=True
            ):
                if id(var) != use.variable_id:
                    raise StructureMismatch("source-input-identity-differs")
                self._check_source_use(scope, use, value)
            scope.pending[index] = inputs

        self._observe("equation-input-capture", record)

    def _expected(self, eqn: JaxprEqn, index: int) -> tuple[_Scope, _Equation]:
        if not self.active or self.active[-1] is None:
            raise StructureUnavailable("equation-scope-unavailable")
        scope = self.active[-1]
        assert scope is not None
        if (
            index < 0
            or index >= len(scope.equations)
            or scope.equations[index].source_id != id(eqn)
        ):
            raise StructureUnavailable("equation-identity-unavailable")
        return scope, scope.equations[index]

    def equation(self, ctx: ConverterContext, eqn: JaxprEqn, index: int) -> None:
        def record() -> None:
            scope, expected = self._expected(eqn, index)
            if index not in scope.pending or index in scope.receipts:
                raise StructureUnavailable("equation-identity-unavailable")
            inputs = scope.pending.pop(index)
            outputs = tuple(_witness(_lookup(ctx, var)) for var in eqn.outvars)
            if len(outputs) != len(expected.outputs):
                raise StructureMismatch("source-result-arity-differs")
            for use, value in zip(expected.inputs, inputs, strict=True):
                self._check_source_use(scope, use, value)
            for use, var, value in zip(
                expected.outputs, eqn.outvars, outputs, strict=True
            ):
                if id(var) != use.variable_id:
                    raise StructureMismatch("source-result-identity-differs")
                if use.variable_id in scope.variables:
                    raise StructureMismatch("source-result-redefinition")
                scope.variables[use.variable_id] = value
            scope.receipts[index] = _Receipt(inputs, outputs)

        self._observe("equation-capture", record)


def _capture_binding(ctx: ConverterContext, use: _Use, var: object) -> _Binding:
    if id(var) != use.variable_id:
        raise StructureMismatch("source-use-identity-differs")
    if use.literal is not None:
        # Observation must never call require_value_for_var for a literal: it
        # would create another initializer, unrelated to the actual consumer.
        return LiteralOperand(use.tensor, use.literal)
    return _witness(_lookup(ctx, var))


def _lookup(ctx: ConverterContext, var: object) -> ir.Value:
    if isinstance(var, JaxLiteral):
        raise StructureUnavailable("literal-use-contract-unavailable")
    value = ctx.lookup_value_for_var(var)
    if value is None:
        raise StructureUnavailable("source-binding-unavailable")
    return value


def attach_source_structure(
    model: ir.Model,
    ledger: SourceStructureLedger | None,
    *,
    boundary_layout: bool = False,
) -> None:
    if ledger is not None:
        if boundary_layout:
            ledger.gaps.add("boundary-layout-contract-unavailable")
        model.meta[_LEDGER] = ledger


def _resolve_bindings(
    graph: ir.Graph,
    wanted: set[_ValueWitness],
    optional: frozenset[_ValueWitness] = frozenset(),
) -> dict[_ValueWitness, ir.Value]:
    """Resolve actual retained definitions, without recovering identities by name."""
    found: dict[_ValueWitness, ir.Value] = {}
    values = [*graph.inputs, *graph.initializers.values()]
    for node in graph:
        values.extend(node.outputs)
    seen_values: set[int] = set()
    for value in values:
        if id(value) in seen_values:
            continue
        seen_values.add(id(value))
        markers = value.meta.get(_VALUES, ())
        if type(markers) is not tuple or any(
            type(item) is not _ValueWitness for item in markers
        ):
            raise StructureUnavailable("final-value-marker-unavailable")
        for witness in markers:
            if witness not in wanted:
                continue
            actual_graph, definition, role, port = _location(value)
            if (
                actual_graph is not graph
                or graph.meta.get(_OWNER) is not witness.owner
                or definition.meta.get(_ORIGIN) is not witness.origin
                or role != witness.role
                or port != witness.port
                or witness in found
            ):
                raise StructureUnavailable("final-value-witness-unavailable")
            found[witness] = value
    if wanted - set(found) - optional:
        raise StructureUnavailable("final-value-witness-lost")
    return found


def _resolve(
    graph: ir.Graph, scope: _Scope, optional: frozenset[_ValueWitness] = frozenset()
) -> dict[_ValueWitness, ir.Value]:
    if graph.meta.get(_OWNER) is not scope.owner:
        raise StructureUnavailable("final-graph-owner-unavailable")
    wanted = {
        binding
        for receipt in scope.receipts.values()
        for binding in (*receipt.inputs, *receipt.outputs)
        if isinstance(binding, _ValueWitness)
    }
    wanted.update(scope.input_values)
    wanted.update(
        binding
        for binding in scope.result_values or ()
        if isinstance(binding, _ValueWitness)
    )
    return _resolve_bindings(graph, wanted, optional)


@dataclass(frozen=True)
class _IfAudit:
    node: ir.Node
    selector: EquationHandle
    boolean: _ValueWitness
    index: _ValueWitness
    foldable: bool
    regions: tuple[_Scope, _Scope]
    captures: tuple[ir.Value, ...]
    adapters: tuple[ir.Node, ...]


@dataclass
class _Coverage:
    nodes: set[ir.Node] = field(default_factory=set)
    constants: set[ir.Value] = field(default_factory=set)


def _visible_before(
    value: ir.Value, node: ir.Node, parents: dict[object, tuple[ir.Graph, ir.Node]]
) -> bool:
    owner_graph, definition, _, _ = _location(value)
    current = node.graph
    boundary = node
    while current is not None:
        if current is owner_graph:
            if isinstance(definition, ir.Value):
                return True
            nodes = list(current)
            return nodes.index(definition) < nodes.index(boundary)
        link = parents.get(current.meta.get(_OWNER))
        if link is None:
            return False
        current, boundary = link
    return False


def _plan_if_calls(
    ledger: SourceStructureLedger,
    scopes: dict[object, _Scope],
    graphs: dict[object, ir.Graph],
    parents: dict[object, tuple[ir.Graph, ir.Node]],
    matches: Callable[[TensorContract, ir.Value], bool],
) -> dict[EquationHandle, _IfAudit]:
    from jax2onnx.plugins._source_control_flow_structure import (
        BooleanIndexRule,
        IfAttachment,
        IfRule,
    )

    wanted = {
        EquationHandle(scope.invocation, index)
        for scope in scopes.values()
        for index, equation in enumerate(scope.equations)
        if type(equation.rule) is IfRule
    }
    if wanted != set(ledger.if_calls):
        raise StructureUnavailable("if-call-coverage-incomplete")
    by_invocation = {scope.invocation: scope for scope in scopes.values()}
    plans: dict[EquationHandle, _IfAudit] = {}
    for handle, attachment in ledger.if_calls.items():
        if type(attachment) is not IfAttachment:
            raise StructureUnavailable("if-attachment-unavailable")
        scope = by_invocation[handle.invocation]
        expected = scope.equations[handle.index]
        receipt = scope.receipts.get(handle.index)
        if receipt is None:
            raise StructureUnavailable("if-source-receipt-unavailable")
        graph = graphs[scope.owner]
        outputs = _resolve_bindings(graph, set(receipt.outputs))
        node = outputs[receipt.outputs[0]].producer()
        if (
            node is None
            or node.graph is not graph
            or node not in set(graph)
            or node.meta.get(_ORIGIN) is not attachment.origin
            or attachment.owner is not scope.owner
            or node.domain
            or node.overload
            or node.op_type != "If"
            or len(node.inputs) != 1
            or len(node.outputs) != len(receipt.outputs)
            or any(
                actual is not outputs[witness]
                for actual, witness in zip(node.outputs, receipt.outputs, strict=True)
            )
            or set(node.attributes) != {"then_branch", "else_branch"}
        ):
            raise StructureMismatch("if-target-or-results-differ")
        children: list[_Scope] = []
        captures: tuple[ir.Value, ...] | None = None
        body_children = dict(scope.body.equations[handle.index].children)
        for role, key, owner, attr in (
            ("false", attachment.false, attachment.false_owner, "else_branch"),
            ("true", attachment.true, attachment.true_owner, "then_branch"),
        ):
            if key != RegionKey(handle, role) or key not in ledger.regions:
                raise StructureUnavailable("if-region-role-unavailable")
            child = by_invocation.get(ledger.regions[key])
            if (
                child is None
                or child.owner is not owner
                or child.region is None
                or child.region.key != key
            ):
                raise StructureUnavailable("if-region-owner-unavailable")
            attribute = node.attributes[attr]
            if (
                attribute.type is not ir.AttributeType.GRAPH
                or attribute.as_graph() is not graphs.get(owner)
            ):
                raise StructureMismatch("if-child-attachment-differs")
            link = parents.get(owner)
            if link is None or link[0] is not graph or link[1] is not node:
                raise StructureMismatch("if-child-ancestry-differs")
            if graphs[owner].inputs:
                raise StructureMismatch("if-child-formals-differ")
            if (
                child.region.expected != body_children.get(role)
                or child.body != child.region.expected
                or child.body.gaps
            ):
                raise StructureMismatch("if-original-child-body-differs")
            ports = child.region.captures
            if (
                len(ports) != len(expected.inputs) - 1
                or tuple(port.binding for port in ports) != receipt.inputs[1:]
            ):
                raise StructureMismatch("if-ordered-captures-differ")
            actual: dict[_ValueWitness, ir.Value] = {}
            for port in ports:
                owner_graph = graphs.get(port.outer.owner)
                if owner_graph is None:
                    raise StructureUnavailable("if-capture-owner-unavailable")
                actual.update(_resolve_bindings(owner_graph, {port.outer}))
            ordered = tuple(actual[port.outer] for port in ports)
            unused = _unused_inputs(body_children[role])
            for index, (port, value) in enumerate(zip(ports, ordered, strict=True)):
                if (
                    type(port) is not CapturePort
                    or port.local != child.input_values[index]
                    or port.tensor != child.body.inputs[index]
                    or type(port.unused) is not bool
                    or port.unused != (index in unused)
                    or port.tensor != expected.inputs[index + 1].tensor
                    or not matches(port.tensor, value)
                ):
                    raise StructureMismatch("if-capture-type-differs")
                if not _visible_before(value, node, parents):
                    raise StructureMismatch("if-capture-not-visible-before-call")
                if (
                    isinstance(port.binding, _ValueWitness)
                    and port.outer != port.binding
                ):
                    raise StructureMismatch("if-capture-source-edge-differs")
            if captures is not None and any(
                left is not right for left, right in zip(captures, ordered, strict=True)
            ):
                raise StructureMismatch("if-branch-capture-edges-differ")
            captures = ordered
            children.append(child)
        selector_variable = expected.inputs[0].variable_id
        producers = [
            (index, equation)
            for index, equation in enumerate(scope.equations)
            if type(equation.rule) is BooleanIndexRule
            and len(equation.outputs) == 1
            and equation.outputs[0].variable_id == selector_variable
        ]
        if len(producers) != 1:
            raise StructureUnavailable("if-boolean-index-source-unavailable")
        selector_index, conversion = producers[0]
        conversion_receipt = scope.receipts.get(selector_index)
        if (
            conversion_receipt is None
            or len(conversion_receipt.inputs) != 1
            or len(conversion_receipt.outputs) != 1
        ):
            raise StructureUnavailable("if-boolean-index-receipt-unavailable")
        boolean = conversion_receipt.inputs[0]
        if (
            type(boolean) is not _ValueWitness
            or receipt.inputs[0] != conversion_receipt.outputs[0]
        ):
            raise StructureMismatch("if-boolean-index-source-edge-differs")
        foldable = not any(
            use.variable_id == selector_variable for use in scope.outputs
        )
        for index, equation in enumerate(scope.equations):
            for source_port, use in enumerate(equation.inputs):
                if use.variable_id == selector_variable and not (
                    source_port == 0
                    and type(equation.rule) is IfRule
                    and EquationHandle(scope.invocation, index) in wanted
                ):
                    foldable = False
        index_witness = conversion_receipt.outputs[0]
        selector_values = _resolve_bindings(
            graph,
            {boolean, index_witness},
            frozenset({index_witness}) if foldable else frozenset(),
        )
        boolean_value = selector_values[boolean]
        if not matches(
            conversion.inputs[0].tensor, boolean_value
        ) or not _visible_before(boolean_value, node, parents):
            raise StructureMismatch("if-boolean-source-type-or-visibility-differs")
        condition = node.inputs[0]
        if (
            condition is None
            or condition.dtype is not ir.DataType.BOOL
            or condition.shape is None
            or tuple(condition.shape) != ()
        ):
            raise StructureMismatch("if-condition-type-differs")
        adapters: tuple[ir.Node, ...] = ()
        if condition is not boolean_value:
            index_value = selector_values.get(index_witness)
            cast = condition.producer()
            if (
                index_value is None
                or cast is None
                or cast.graph is not graph
                or cast not in set(graph)
                or cast.domain
                or cast.overload
                or cast.op_type != "Cast"
                or len(cast.inputs) != 1
                or cast.inputs[0] is not index_value
                or len(cast.outputs) != 1
                or cast.outputs[0] is not condition
                or set(cast.attributes) != {"to"}
                or cast.attributes["to"].type is not ir.AttributeType.INT
                or cast.attributes["to"].as_int() != int(ir.DataType.BOOL.value)
            ):
                raise StructureMismatch("if-selector-composition-differs")
            adapters = (cast,)
        plans[handle] = _IfAudit(
            node,
            EquationHandle(scope.invocation, selector_index),
            boolean,
            index_witness,
            foldable,
            (children[0], children[1]),
            captures or (),
            adapters,
        )
    return plans


def _audit_scope(
    graph: ir.Graph,
    scope: _Scope,
    ledger: SourceStructureLedger,
    definitions: dict[object, ir.Function],
    plans: dict[EquationHandle, _IfAudit],
    coverage: _Coverage,
    while_plans: dict[EquationHandle, Any],
    matches: Callable[[TensorContract, ir.Value], bool],
) -> StructureResult:
    from jax2onnx.plugins._source_function_structure import FunctionRule
    from jax2onnx.plugins._source_control_flow_structure import BooleanIndexRule, IfRule

    from jax2onnx.plugins.jax.lax._source_loop_structure import WhileRule

    total = len(scope.equations)
    if set(scope.receipts) != set(range(total)) or scope.result_values is None:
        return StructureResult("unverified", total, 0, ("source-coverage-incomplete",))
    region = scope.region
    optional = {
        plan.index
        for plan in plans.values()
        if plan.selector.invocation is scope.invocation and plan.foldable
    }
    if region is not None:
        optional.update(port.local for port in region.captures if port.unused)
    values = _resolve(graph, scope, frozenset(optional))
    if region is None and (
        len(graph.inputs) != len(scope.inputs)
        or any(
            actual is not values[witness] or not matches(use.tensor, actual)
            for actual, witness, use in zip(
                graph.inputs, scope.input_values, scope.inputs, strict=True
            )
        )
    ):
        raise StructureMismatch("source-input-interface-differs")
    checked = coverage.nodes
    constant_cache: dict[int, ConstantContract] = {}
    literal_cache: dict[int, ConstantContract] = {}
    literal_initializers = coverage.constants

    actual_outputs = tuple(graph.outputs)
    if region is not None and region.key.parent in while_plans:
        while_plan = while_plans[region.key.parent]
        roles = ("initial-condition", "body", "next-condition")
        if region.key.role not in roles:
            raise StructureUnavailable("while-region-role-unavailable")
        position = roles.index(region.key.role)
        if while_plan.regions[position] is not scope:
            raise StructureMismatch("while-region-invocation-differs")
        actual_outputs = while_plan.region_outputs[position]
    elif region is not None:
        if region.key.parent not in plans:
            raise StructureUnavailable("region-attachment-unavailable")
        plan = plans[region.key.parent]
        if graph.inputs or len(region.captures) != len(scope.inputs):
            raise StructureMismatch("region-formal-interface-differs")
        for port, outer in zip(region.captures, plan.captures, strict=True):
            local = values.get(port.local)
            if local is None:
                if not port.unused:
                    raise StructureUnavailable("used-region-capture-lost")
                continue
            producer = local.producer()
            if (
                producer is None
                or producer.graph is not graph
                or producer not in set(graph)
                or producer.domain
                or producer.overload
                or producer.op_type != "Identity"
                or producer.attributes
                or len(producer.inputs) != 1
                or producer.inputs[0] is not outer
                or len(producer.outputs) != 1
                or producer.outputs[0] is not local
                or not matches(port.tensor, local)
            ):
                raise StructureMismatch("region-capture-identity-edge-differs")
            checked.add(producer)

    def match_literal(value: ir.Value | None, expected: LiteralOperand) -> None:
        if value is None:
            raise StructureMismatch("literal-input-missing")
        if value.is_graph_input() or any(item is value for item in graph.inputs):
            raise StructureMismatch("literal-runtime-input-substitute")
        if value.graph is not graph:
            raise StructureUnavailable("literal-owner-unavailable")
        producer = value.producer()
        if producer is None:
            if not any(item is value for item in graph.initializers.values()):
                raise StructureUnavailable("literal-definition-unavailable")
            tensor = value.const_value
        elif (
            producer.graph is graph
            and any(node is producer for node in graph)
            and not producer.domain
            and not producer.overload
            and producer.op_type == "Constant"
        ):
            if (
                producer.inputs
                or len(producer.outputs) != 1
                or producer.outputs[0] is not value
                or set(producer.attributes) != {"value"}
                or producer.attributes["value"].type is not ir.AttributeType.TENSOR
            ):
                raise StructureUnavailable("literal-constant-contract-unavailable")
            # A cached output.const_value can be stale after an attribute edit.
            tensor = producer.attributes["value"].as_tensor()
        else:
            raise StructureUnavailable("literal-producer-contract-unavailable")
        if value.dtype is None or value.shape is None or tensor is None:
            raise StructureUnavailable("literal-storage-unavailable")
        if not matches(expected.tensor, value):
            raise StructureMismatch("literal-type-or-shape-differs")
        if id(value) not in literal_cache:
            literal_cache[id(value)] = ConstantContract.from_array(tensor.numpy())
        if literal_cache[id(value)] != expected.storage:
            raise StructureMismatch("literal-storage-differs")
        if producer is None:
            literal_initializers.add(value)
        else:
            checked.add(producer)

    def check_value(use: _Use, witness: _ValueWitness) -> ir.Value:
        value = values[witness]
        if value.dtype is None or value.shape is None:
            raise StructureUnavailable("source-value-type-unavailable")
        if not matches(use.tensor, value):
            raise StructureMismatch("source-value-type-differs")
        expected = ledger.constants.get(witness)
        if expected is not None:
            producer = value.producer()
            constant_tensor: ir.TensorProtocol | None
            if producer is not None:
                if (
                    producer.graph is not graph
                    or producer not in set(graph)
                    or producer.domain
                    or producer.overload
                    or producer.op_type != "Constant"
                    or producer.inputs
                    or len(producer.outputs) != 1
                    or producer.outputs[0] is not value
                    or set(producer.attributes) != {"value"}
                    or producer.attributes["value"].type is not ir.AttributeType.TENSOR
                ):
                    raise StructureUnavailable("source-constant-definition-unavailable")
                constant_tensor = producer.attributes["value"].as_tensor()
                checked.add(producer)
            else:
                if not any(item is value for item in graph.initializers.values()):
                    raise StructureUnavailable("source-constant-definition-unavailable")
                constant_tensor = value.const_value
            if constant_tensor is None:
                raise StructureUnavailable("source-constant-storage-unavailable")
            if id(value) not in constant_cache:
                constant_cache[id(value)] = ConstantContract.from_array(
                    constant_tensor.numpy()
                )
            if constant_cache[id(value)] != expected:
                raise StructureMismatch("source-constant-storage-differs")
            if producer is None:
                coverage.constants.add(value)
        return value

    def check_operand(use: _Use, binding: _Binding) -> RuleOperand:
        if isinstance(binding, LiteralOperand):
            if use.literal is None or binding != LiteralOperand(
                use.tensor, use.literal
            ):
                raise StructureMismatch("literal-source-binding-differs")
            return binding
        if use.literal is not None:
            raise StructureMismatch("literal-source-binding-differs")
        return check_value(use, binding)

    if region is not None:
        # DCE may retain an original unused initializer. It still needs exact
        # source storage and definition authority; absence of consumers is not
        # permission to admit arbitrary additional branch storage.
        for initializer in graph.initializers.values():
            for witness in initializer.meta.get(_VALUES, ()):
                if witness not in scope.variables.values():
                    continue
                storage = ledger.constants.get(witness)
                if storage is None:
                    continue
                resolved = _resolve_bindings(graph, {witness})
                if resolved[witness] is not initializer:
                    raise StructureMismatch("region-initializer-definition-differs")
                tensors = [
                    tensor
                    for tensor, original in scope.body.constants
                    if original == storage and tensor is not None
                ]
                if not tensors:
                    raise StructureUnavailable("region-initializer-source-unavailable")
                if not any(matches(tensor, initializer) for tensor in tensors):
                    raise StructureMismatch("region-initializer-type-differs")
                tensor_storage = initializer.const_value
                if tensor_storage is None:
                    raise StructureUnavailable("region-initializer-storage-unavailable")
                if ConstantContract.from_array(tensor_storage.numpy()) != storage:
                    raise StructureMismatch("region-initializer-storage-differs")
                coverage.constants.add(initializer)

    count = 0
    reasons: list[str] = []
    for index, expected in enumerate(scope.equations):
        receipt = scope.receipts[index]
        handle = EquationHandle(scope.invocation, index)
        if type(expected.rule) is BooleanIndexRule and receipt.outputs[0] not in values:
            users = [plan for plan in plans.values() if plan.selector == handle]
            if not users or any(
                not plan.foldable or plan.node.inputs[0] is not values[plan.boolean]
                for plan in users
            ):
                raise StructureUnavailable("boolean-index-composition-unavailable")
            check_operand(expected.inputs[0], receipt.inputs[0])
            count += 1
            continue
        if type(expected.rule) is IfRule:
            plan = plans[handle]
            for capture_index, (use, binding) in enumerate(
                zip(expected.inputs[1:], receipt.inputs[1:], strict=True)
            ):
                operand = check_operand(use, binding)
                if isinstance(operand, LiteralOperand):
                    match_literal(plan.captures[capture_index], operand)
                elif operand is not plan.captures[capture_index]:
                    raise StructureMismatch("if-capture-operand-edge-differs")
            for use, witness in zip(expected.outputs, receipt.outputs, strict=True):
                check_value(use, witness)
            checked.update((plan.node, *plan.adapters))
            count += 1
            continue
        inputs = tuple(
            check_operand(use, binding)
            for use, binding in zip(expected.inputs, receipt.inputs, strict=True)
        )
        outputs = tuple(
            check_value(use, witness)
            for use, witness in zip(expected.outputs, receipt.outputs, strict=True)
        )
        if expected.rule is None:
            reasons.append("equation-" + str(index) + ":rule-unavailable")
            continue
        if type(expected.rule) is WhileRule:
            plan = while_plans[handle]
            for actual, operand in zip(plan.inputs, inputs, strict=True):
                if isinstance(operand, LiteralOperand):
                    match_literal(actual, operand)
                elif actual is not operand:
                    raise StructureMismatch("while-original-operand-edge-differs")
            checked.add(plan.node)
        elif type(expected.rule) is FunctionRule:
            call = ledger.function_calls.get(EquationHandle(scope.invocation, index))
            if call is None or call.owner not in definitions:
                reasons.append("function-call-coverage-incomplete")
                continue
            checked.update(
                expected.rule.check_call(
                    graph, inputs, outputs, match_literal, call, definitions[call.owner]
                )
            )
        else:
            checked.update(expected.rule.check(graph, inputs, outputs, match_literal))
        count += 1
    if len(actual_outputs) != len(scope.outputs):
        raise StructureMismatch("source-output-arity-differs")
    for actual, use, witness in zip(
        actual_outputs, scope.outputs, scope.result_values, strict=True
    ):
        expected_value = check_operand(use, witness)
        if isinstance(expected_value, LiteralOperand):
            producer = actual.producer()
            if producer is not None and producer.op_type == "Identity":
                if (
                    producer.graph is not graph
                    or not any(node is producer for node in graph)
                    or producer.domain
                    or producer.overload
                    or producer.attributes
                    or len(producer.inputs) != 1
                    or len(producer.outputs) != 1
                    or producer.outputs[0] is not actual
                ):
                    raise StructureMismatch("literal-output-alias-differs")
                if not matches(use.tensor, actual):
                    raise StructureMismatch("literal-type-or-shape-differs")
                match_literal(producer.inputs[0], expected_value)
                checked.add(producer)
            else:
                match_literal(actual, expected_value)
        elif actual is not expected_value:
            # Only facade Identity aliases are approved at the final boundary.
            producer = actual.producer()
            if (
                producer is None
                or producer.graph is not graph
                or producer.domain
                or producer.overload
                or producer.op_type != "Identity"
                or producer.attributes
                or len(producer.inputs) != 1
                or producer.inputs[0] is not expected_value
                or len(producer.outputs) != 1
                or producer.outputs[0] is not actual
            ):
                raise StructureMismatch("source-output-interface-differs")
            checked.add(producer)
        if not matches(use.tensor, actual):
            raise StructureMismatch("source-output-type-differs")
    return StructureResult(
        "unverified" if reasons else "validated", total, count, tuple(reasons[:16])
    )


def _audit(model: ir.Model, ledger: SourceStructureLedger) -> StructureResult:
    from jax2onnx.ir_utils import iter_ir_functions
    from jax2onnx.plugins._source_function_structure import FunctionRule
    from jax2onnx.plugins._source_control_flow_structure import IfRule

    total = ledger.expected_equations
    if ledger.failures:
        return StructureResult("failed", total, 0, tuple(sorted(ledger.failures))[:16])
    if ledger.gaps:
        return StructureResult("unverified", total, 0, tuple(sorted(ledger.gaps))[:16])
    from ._source_cast_planning import plan_is_current

    expected_cast_plans = {
        EquationHandle(
            scope.invocation, scope.cast_plan.eligibility.root_ordinal
        ): scope.cast_plan
        for scope in ledger.scopes
        if scope.cast_plan is not None
    }
    if set(ledger.cast_plans) != set(expected_cast_plans) or any(
        ledger.cast_plans[handle] is not plan or not plan_is_current(plan)
        for handle, plan in expected_cast_plans.items()
    ):
        raise StructureUnavailable("cast-plan-handle-coverage-incomplete")
    expected_cast_selections = {
        EquationHandle(scope.invocation, 0): scope.cast_selection
        for scope in ledger.scopes
        if scope.cast_selection is not None
    }
    if set(ledger.cast_selections) != set(expected_cast_selections) or any(
        ledger.cast_selections[handle] is not rule
        or next(
            scope for scope in ledger.scopes if scope.invocation is handle.invocation
        )
        .equations[0]
        .rule
        is not rule
        for handle, rule in expected_cast_selections.items()
    ):
        raise StructureUnavailable("selected-cast-handle-coverage-incomplete")
    graphs: dict[object, ir.Graph] = {}
    parents: dict[object, tuple[ir.Graph, ir.Node]] = {}

    def collect(
        graph: ir.Graph, parent: tuple[ir.Graph, ir.Node] | None = None
    ) -> None:
        owner = graph.meta.get(_OWNER)
        if type(owner) is not object or owner in graphs:
            raise StructureUnavailable("final-graph-owner-unavailable")
        graphs[owner] = graph
        if parent is not None:
            parents[owner] = parent
        for node in graph:
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    collect(attribute.as_graph(), (graph, node))
                elif attribute.type is ir.AttributeType.GRAPHS:
                    for child in attribute.as_graphs():
                        collect(child, (graph, node))

    collect(model.graph)
    definitions: dict[object, ir.Function] = {}
    for definition in iter_ir_functions(model.functions):
        collect(definition.graph)
        definitions[definition.graph.meta[_OWNER]] = definition
    scopes: dict[object, _Scope] = {}
    owner_scopes: dict[object, list[_Scope]] = {}
    results: dict[object, StructureResult] = {}
    wanted_calls: set[EquationHandle] = set()
    for scope in ledger.scopes:
        if scope.invocation in scopes or scope.owner not in graphs:
            raise StructureUnavailable("scope-owner-coverage-incomplete")
        scopes[scope.invocation] = scope
        owner_scopes.setdefault(scope.owner, []).append(scope)
        if scope.owner in definitions and scope.region is None:
            definition_body = ledger.function_bodies.get(scope.owner)
            if definition_body is None:
                raise StructureUnavailable(
                    "function-definition-source-body-unavailable"
                )
            if definition_body != scope.body:
                raise StructureMismatch("function-definition-source-body-differs")
            if scope.body is None or scope.body.gaps:
                raise StructureUnavailable("function-definition-contract-unavailable")
        wanted_calls.update(
            EquationHandle(scope.invocation, index)
            for index, equation in enumerate(scope.equations)
            if type(equation.rule) is FunctionRule
        )
    if set(owner_scopes) != set(graphs) or wanted_calls != set(ledger.function_calls):
        raise StructureUnavailable("region-or-call-coverage-incomplete")
    if set(definitions) != {call.owner for call in ledger.function_calls.values()}:
        raise StructureUnavailable("function-definition-coverage-incomplete")

    from jax2onnx.plugins.jax.lax._source_loop_structure import (
        ActualShapeFacts,
        WhileRule,
        plan_while_calls,
    )

    if ledger.while_calls:
        # Storage facts do not establish a producer's position. Check actual
        # consumed edges before deriving shapes, including lexical captures.
        for graph in graphs.values():
            for node in graph:
                for value in node.inputs:
                    if value is not None and not _visible_before(value, node, parents):
                        raise StructureMismatch(
                            "while-input-not-visible-before-consumer"
                        )
    facts = ActualShapeFacts(model, definitions) if ledger.while_calls else None
    matches = facts.matches if facts is not None else TensorContract.matches
    plans = _plan_if_calls(ledger, scopes, graphs, parents, matches)
    while_plans = plan_while_calls(ledger, scopes, graphs, parents, matches)
    wanted_regions = {
        cast(RegionReceipt, child.region).key
        for plan in cast(
            "tuple[_IfAudit | WhileAudit, ...]",
            (*plans.values(), *while_plans.values()),
        )
        for child in plan.regions
    }
    observed_regions: dict[RegionKey, object] = {}
    for scope in scopes.values():
        if scope.region is None:
            continue
        key = scope.region.key
        if key in observed_regions or ledger.regions.get(key) is not scope.invocation:
            raise StructureUnavailable("region-invocation-binding-unavailable")
        observed_regions[key] = scope.invocation
    if set(ledger.regions) != wanted_regions or set(observed_regions) != wanted_regions:
        raise StructureUnavailable("region-coverage-incomplete")

    def definition_scope(owner: object) -> _Scope:
        candidates = [scope for scope in owner_scopes[owner] if scope.region is None]
        if len(candidates) != 1:
            raise StructureUnavailable("definition-region-ownership-unavailable")
        return candidates[0]

    children: dict[object, tuple[object, ...]] = {}
    for scope in scopes.values():
        attached: list[object] = []
        for index, equation in enumerate(scope.equations):
            handle = EquationHandle(scope.invocation, index)
            if type(equation.rule) is FunctionRule:
                attached.append(
                    definition_scope(ledger.function_calls[handle].owner).invocation
                )
            elif type(equation.rule) is IfRule:
                attached.extend(child.invocation for child in plans[handle].regions)
            elif type(equation.rule) is WhileRule:
                attached.extend(
                    child.invocation for child in while_plans[handle].regions
                )
        children[scope.invocation] = tuple(attached)
    root_invocation = definition_scope(model.graph.meta[_OWNER]).invocation
    reached: set[object] = set()

    def visit(invocation: object, ancestors: frozenset[object]) -> None:
        if invocation in ancestors:
            raise StructureUnavailable("recursive-function-contract-unavailable")
        reached.add(invocation)
        for child in children[invocation]:
            visit(child, ancestors | {invocation})

    visit(root_invocation, frozenset())
    if reached != set(scopes):
        raise StructureUnavailable("unclaimed-region-invocation")
    coverage = {owner: _Coverage() for owner in graphs}
    for plan in while_plans.values():
        for adapter in plan.adapters:
            coverage[cast(ir.Graph, adapter.graph).meta[_OWNER]].nodes.add(adapter)
    for scope in scopes.values():
        results[scope.invocation] = _audit_scope(
            graphs[scope.owner],
            scope,
            ledger,
            definitions,
            plans,
            coverage[scope.owner],
            while_plans,
            matches,
        )

    def logical_count(invocation: object, ancestors: frozenset[object]) -> int:
        if invocation in ancestors:
            raise StructureUnavailable("recursive-function-contract-unavailable")
        return results[invocation].checked_equations + sum(
            logical_count(child, ancestors | {invocation})
            for child in children[invocation]
        )

    checked = logical_count(root_invocation, frozenset())
    reasons = tuple(
        sorted({reason for result in results.values() for reason in result.reasons})
    )
    used_constants: dict[object, set[ir.Value]] = {owner: set() for owner in graphs}
    for graph in graphs.values():
        consumed = [
            value for node in graph for value in node.inputs if value is not None
        ]
        for value in (*consumed, *graph.outputs):
            if value.is_initializer():
                owner_graph = value.graph
                if (
                    owner_graph is None
                    or graphs.get(owner_graph.meta.get(_OWNER)) is not owner_graph
                ):
                    raise StructureUnavailable("artifact-constant-owner-unavailable")
                used_constants[owner_graph.meta[_OWNER]].add(value)
    for owner, graph in graphs.items():
        if any(
            "source-coverage-incomplete" in results[scope.invocation].reasons
            for scope in owner_scopes[owner]
        ):
            # Preserve the established primary diagnostic for an unobserved
            # original equation; its independently declared count still holds.
            continue
        if coverage[owner].nodes != set(graph):
            reasons = (*reasons, "artifact-node-coverage-incomplete")
        # Lexical consumers can live in descendant graphs; ownership is the
        # actual initializer definition, not the spelling of its name.
        if used_constants[owner] - coverage[owner].constants:
            reasons = (*reasons, "artifact-constant-coverage-incomplete")
        if (
            any(scope.region is not None for scope in owner_scopes[owner])
            and set(graph.initializers.values()) - coverage[owner].constants
        ):
            reasons = (*reasons, "artifact-constant-coverage-incomplete")
    reasons = tuple(dict.fromkeys(reasons))
    if checked != total and not reasons:
        reasons = (*reasons, "expected-region-coverage-incomplete")
    return StructureResult(
        "unverified" if reasons else "validated", total, checked, reasons[:16]
    )


def validate_source_structure(model: ir.Model) -> StructureResult:
    """Observe final IR without changing admission, optimizer or numeric status."""
    ledger = model.meta.get(_LEDGER)
    total = ledger.expected_equations if type(ledger) is SourceStructureLedger else 0
    try:
        result = (
            _audit(model, ledger)
            if type(ledger) is SourceStructureLedger
            else StructureResult("unverified", 0, 0, ("source-ledger-unavailable",))
        )
    except StructureMismatch as exc:
        result = StructureResult("failed", total, 0, (str(exc),))
    except StructureUnavailable as exc:
        result = StructureResult("unverified", total, 0, (str(exc),))
    except Exception as exc:
        result = StructureResult(
            "unverified", total, 0, ("audit:" + type(exc).__name__,)
        )
    model.meta[RESULT_KEY] = result
    return result
