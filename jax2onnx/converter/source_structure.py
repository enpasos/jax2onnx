# jax2onnx/converter/source_structure.py

"""Private compositional SOURCE structure observations, independent of numerics.

The dispatcher freezes the actual source equations before invoking plugins.
Rules owned by plugins check the final IR, after optimization and facade changes.
This first contract covers a single flat graph. Missing rules, scopes or clone
witnesses keep the audit unverified; observation never rejects an export.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Any, Final, Literal, Protocol, TYPE_CHECKING, TypeAlias

import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal as JaxLiteral
from .source_representation import (
    native_shape_contract,
    normalized_shape_dimensions,
    representation_for_aval,
)

if TYPE_CHECKING:
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


@dataclass
class _Scope:
    invocation: object
    owner: object
    equations: tuple[_Equation, ...]
    inputs: tuple[_Use, ...]
    outputs: tuple[_Use, ...]
    input_values: tuple[_ValueWitness, ...]
    variables: dict[int, _ValueWitness]
    pending: dict[int, tuple[_Binding, ...]] = field(default_factory=dict)
    receipts: dict[int, _Receipt] = field(default_factory=dict)
    result_values: tuple[_Binding, ...] | None = None


@dataclass(frozen=True)
class StructureResult:
    status: Literal["validated", "failed", "unverified"]
    expected_equations: int
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
        self.expected_equations = 0

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
        self.expected_equations += len(jaxpr.eqns)

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
                operands = tuple(_use(var, shapes) for var in eqn.invars)
                results = tuple(_use(var, shapes) for var in eqn.outvars)
                return _Equation(
                    id(eqn),
                    operands,
                    results,
                    (
                        freeze_rule(
                            eqn,
                            tuple(use.tensor for use in operands),
                            tuple(use.tensor for use in results),
                        )
                        if not eqn.effects
                        else None
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
            scope = _Scope(
                object(),
                owner,
                equations,
                inputs,
                outputs,
                values,
                variables,
            )
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


def _resolve(model: ir.Model, scope: _Scope) -> dict[_ValueWitness, ir.Value]:
    graph = model.graph
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
    found: dict[_ValueWitness, ir.Value] = {}
    values = [*graph.inputs, *graph.initializers.values()]
    for node in graph:
        values.extend(node.outputs)
    seen_values: set[int] = set()
    for value in values:
        # Overridable inputs are one actual definition listed in both containers.
        if id(value) in seen_values:
            continue
        seen_values.add(id(value))
        for witness in value.meta.get(_VALUES, ()):
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
    if set(found) != wanted:
        raise StructureUnavailable("final-value-witness-lost")
    return found


def _audit(model: ir.Model, ledger: SourceStructureLedger) -> StructureResult:
    total = ledger.expected_equations
    if ledger.failures:
        return StructureResult("failed", total, 0, tuple(sorted(ledger.failures))[:16])
    if ledger.gaps:
        return StructureResult("unverified", total, 0, tuple(sorted(ledger.gaps))[:16])
    if len(ledger.scopes) != 1 or model.functions:
        return StructureResult(
            "unverified", total, 0, ("flat-scope-contract-unavailable",)
        )
    scope = ledger.scopes[0]
    if any(
        attribute.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
        for node in model.graph
        for attribute in node.attributes.values()
    ):
        return StructureResult(
            "unverified", total, 0, ("nested-graph-contract-unavailable",)
        )
    if set(scope.receipts) != set(range(total)) or scope.result_values is None:
        return StructureResult("unverified", total, 0, ("source-coverage-incomplete",))
    values = _resolve(model, scope)
    if len(model.graph.inputs) != len(scope.inputs) or any(
        actual is not values[witness] or not use.tensor.matches(actual)
        for actual, witness, use in zip(
            model.graph.inputs, scope.input_values, scope.inputs, strict=True
        )
    ):
        raise StructureMismatch("source-input-interface-differs")
    checked: set[ir.Node] = set()
    constant_cache: dict[int, ConstantContract] = {}
    literal_cache: dict[int, ConstantContract] = {}
    literal_initializers: set[ir.Value] = set()

    def match_literal(value: ir.Value | None, expected: LiteralOperand) -> None:
        if value is None:
            raise StructureMismatch("literal-input-missing")
        if value.is_graph_input() or any(item is value for item in model.graph.inputs):
            raise StructureMismatch("literal-runtime-input-substitute")
        if value.graph is not model.graph:
            raise StructureUnavailable("literal-owner-unavailable")
        producer = value.producer()
        if producer is None:
            if not any(item is value for item in model.graph.initializers.values()):
                raise StructureUnavailable("literal-definition-unavailable")
            tensor = value.const_value
        elif (
            producer.graph is model.graph
            and any(node is producer for node in model.graph)
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
        if not expected.tensor.matches(value):
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
        if not use.tensor.matches(value):
            raise StructureMismatch("source-value-type-differs")
        expected = ledger.constants.get(witness)
        if expected is not None:
            if value.const_value is None:
                raise StructureUnavailable("source-constant-storage-unavailable")
            if id(value) not in constant_cache:
                constant_cache[id(value)] = ConstantContract.from_array(
                    value.const_value.numpy()
                )
            if constant_cache[id(value)] != expected:
                raise StructureMismatch("source-constant-storage-differs")
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

    count = 0
    reasons: list[str] = []
    for index, expected in enumerate(scope.equations):
        receipt = scope.receipts[index]
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
        checked.update(expected.rule.check(model.graph, inputs, outputs, match_literal))
        count += 1
    if len(model.graph.outputs) != len(scope.outputs):
        raise StructureMismatch("source-output-arity-differs")
    for actual, use, witness in zip(
        model.graph.outputs, scope.outputs, scope.result_values, strict=True
    ):
        expected_value = check_operand(use, witness)
        if isinstance(expected_value, LiteralOperand):
            producer = actual.producer()
            if producer is not None and producer.op_type == "Identity":
                if (
                    producer.graph is not model.graph
                    or not any(node is producer for node in model.graph)
                    or producer.domain
                    or producer.overload
                    or producer.attributes
                    or len(producer.inputs) != 1
                    or len(producer.outputs) != 1
                    or producer.outputs[0] is not actual
                ):
                    raise StructureMismatch("literal-output-alias-differs")
                if not use.tensor.matches(actual):
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
                or producer.graph is not model.graph
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
        if not use.tensor.matches(actual):
            raise StructureMismatch("source-output-type-differs")
    if checked != set(model.graph):
        reasons.append("artifact-node-coverage-incomplete")
    used_constants = {
        value
        for node in model.graph
        for value in node.inputs
        if value is not None and value.is_initializer()
    } | {value for value in model.graph.outputs if value.is_initializer()}
    recorded_constants = {
        values[witness] for witness in ledger.constants if witness in values
    }
    if used_constants - recorded_constants - literal_initializers:
        reasons.append("artifact-constant-coverage-incomplete")
    return StructureResult(
        "unverified" if reasons else "validated", total, count, tuple(reasons[:16])
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
