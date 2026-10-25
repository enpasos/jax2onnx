# jax2onnx/converter/source_structure.py

"""Private compositional SOURCE structure observations, independent of numerics.

The dispatcher freezes the actual source equations before invoking plugins.
Rules owned by plugins check the final IR, after optimization and facade changes.
This first contract covers a single flat graph. Missing rules, scopes or clone
witnesses keep the audit unverified; observation never rejects an export.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Any, Final, Literal, Protocol, TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal as JaxLiteral
from .source_representation import (
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


@dataclass(frozen=True)
class TensorContract:
    logical_dtype: str
    logical_shape: tuple[int | str, ...]
    weak_type: bool
    dtype: ir.DataType
    shape: tuple[int | str, ...]
    encoding: str

    @classmethod
    def from_aval(cls, aval: Any) -> TensorContract:
        representation = representation_for_aval(aval)
        # Symbolic lexical identity needs a separate reviewed carrier. String
        # agreement between DimExprs is not authority for that contract.
        if any(
            not isinstance(dim, (int, np.integer)) or isinstance(dim, (bool, np.bool_))
            for dim in (*representation.logical_shape, *representation.physical_shape)
        ):
            raise StructureUnavailable("symbolic-shape-contract-unavailable")
        return cls(
            str(representation.logical_dtype),
            normalized_shape_dimensions(representation.logical_shape),
            bool(aval.weak_type),
            representation.ir_dtype,
            normalized_shape_dimensions(representation.physical_shape),
            representation.encoding,
        )

    def matches(self, value: ir.Value) -> bool:
        return (
            value.dtype == self.dtype
            and value.shape is not None
            and normalized_shape_dimensions(tuple(value.shape)) == self.shape
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
class _ValueWitness:
    owner: object
    origin: object
    role: Literal["input", "initializer", "producer"]
    port: int


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
        inputs: tuple[ir.Value, ...],
        outputs: tuple[ir.Value, ...],
    ) -> tuple[ir.Node, ...]: ...


@dataclass(frozen=True)
class _Use:
    variable_id: int
    tensor: TensorContract
    literal: ConstantContract | None = None


def _use(var: Any) -> _Use:
    literal = None
    if isinstance(var, JaxLiteral):
        array = np.asarray(var.val, dtype=var.aval.dtype)
        literal = ConstantContract.from_array(
            representation_for_aval(var.aval).encode_array(array)
        )
    return _Use(id(var), TensorContract.from_aval(var.aval), literal)


@dataclass(frozen=True)
class _Equation:
    source_id: int
    inputs: tuple[_Use, ...]
    outputs: tuple[_Use, ...]
    rule: EquationRule | None


@dataclass(frozen=True)
class _Receipt:
    inputs: tuple[_ValueWitness, ...]
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
    pending: dict[int, tuple[_ValueWitness, ...]] = field(default_factory=dict)
    receipts: dict[int, _Receipt] = field(default_factory=dict)
    result_values: tuple[_ValueWitness, ...] | None = None


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
        freeze_rule: Callable[[JaxprEqn], EquationRule | None],
    ) -> Iterator[None]:
        self.expected_equations += len(jaxpr.eqns)

        def begin() -> _Scope:
            graph = ctx.builder.graph
            owner = graph.meta.setdefault(_OWNER, object())
            if type(owner) is not object or jaxpr.effects:
                raise StructureUnavailable("scope-contract-unavailable")
            equations = tuple(
                _Equation(
                    id(eqn),
                    tuple(_use(var) for var in eqn.invars),
                    tuple(_use(var) for var in eqn.outvars),
                    freeze_rule(eqn) if not eqn.effects else None,
                )
                for eqn in jaxpr.eqns
            )
            inputs = tuple(_use(var) for var in jaxpr.invars)
            outputs = tuple(_use(var) for var in jaxpr.outvars)
            if any(
                use.literal is not None
                for use in (*outputs, *(use for eqn in equations for use in eqn.inputs))
            ):
                # require_value_for_var materializes a new literal each call.
                # Actual literal port capture is a future rule; never allocate
                # new IR merely to observe a source use.
                raise StructureUnavailable("literal-use-contract-unavailable")
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
                    actual = tuple(_witness(_lookup(ctx, var)) for var in jaxpr.outvars)
                    for use, value in zip(scope.outputs, actual, strict=True):
                        self._check_source_use(scope, use, value)
                    scope.result_values = actual

                self._observe("scope-results", finish)
            self.active.pop()

    def _check_source_use(self, scope: _Scope, use: _Use, value: _ValueWitness) -> None:
        if use.literal is not None:
            if self.constants.get(value) != use.literal:
                raise StructureMismatch("literal-source-binding-differs")
        elif scope.variables.get(use.variable_id) != value:
            raise StructureMismatch("source-ssa-binding-differs")

    def before_equation(self, ctx: ConverterContext, eqn: JaxprEqn, index: int) -> None:
        def record() -> None:
            scope, expected = self._expected(eqn, index)
            if index in scope.pending or index in scope.receipts:
                raise StructureUnavailable("equation-identity-unavailable")
            inputs = tuple(_witness(_lookup(ctx, var)) for var in eqn.invars)
            if len(inputs) != len(expected.inputs):
                raise StructureMismatch("source-input-arity-differs")
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
    wanted = (
        {
            witness
            for receipt in scope.receipts.values()
            for witness in (*receipt.inputs, *receipt.outputs)
        }
        | set(scope.input_values)
        | set(scope.result_values or ())
    )
    found: dict[_ValueWitness, ir.Value] = {}
    values = [*graph.inputs, *graph.initializers.values()]
    for node in graph:
        values.extend(node.outputs)
    for value in values:
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

    count = 0
    reasons: list[str] = []
    for index, expected in enumerate(scope.equations):
        receipt = scope.receipts[index]
        inputs = tuple(
            check_value(use, witness)
            for use, witness in zip(expected.inputs, receipt.inputs, strict=True)
        )
        outputs = tuple(
            check_value(use, witness)
            for use, witness in zip(expected.outputs, receipt.outputs, strict=True)
        )
        if expected.rule is None:
            reasons.append("equation-" + str(index) + ":rule-unavailable")
            continue
        checked.update(expected.rule.check(model.graph, inputs, outputs))
        count += 1
    if len(model.graph.outputs) != len(scope.outputs):
        raise StructureMismatch("source-output-arity-differs")
    for actual, use, witness in zip(
        model.graph.outputs, scope.outputs, scope.result_values, strict=True
    ):
        expected_value = check_value(use, witness)
        if actual is not expected_value:
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
    if used_constants - recorded_constants:
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
