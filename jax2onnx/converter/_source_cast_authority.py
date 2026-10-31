# jax2onnx/converter/_source_cast_authority.py

"""Conversion-owned, one-use original-cast authority and selected execution.

Public acquisition/planning/options records remain evidence. Only the actual
native producer can hand this transaction an original equation and typed ports.
The permit is private, context-bound and discarded on every conversion exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, TYPE_CHECKING, cast

import onnx_ir as ir

from jax2onnx._compat._cast_lowering import _require_native_handoff
from .precision_policy import PrecisionPolicy

AUTHORITY_KEY: Final[str] = "jax2onnx.source.cast_authority"
_TRANSACTION_SEAL: Final[object] = object()
_PERMIT_SEAL: Final[object] = object()
_SESSION_SEAL: Final[object] = object()
EMISSION_KEY: Final[str] = "jax2onnx.source.cast_emission"

if TYPE_CHECKING:
    from jax2onnx._compat.jax import JaxprEqn
    from jax2onnx.plugins.jax._source_f64_f16_structure import SelectedF64F16Rule


class SourceCastSelectionError(RuntimeError):
    """A previously selected conversion lost its execution authority."""


@dataclass(frozen=True, slots=True)
class SelectedCastContract:
    algorithm: str
    shape: tuple[int, ...]
    opset: int
    build_cohort: str
    shim_sha256: str
    common_sha256: str
    jax_commit: str
    xla_commit: str
    llvm_commit: str
    schema_protos: tuple[str, str]
    provider_proof: str
    options_proof: str
    joined_proof: str
    callable_proof_sha256: str
    source_body_sha256: str
    native_module_sha256: str
    native_options_sha256: str
    native_target_sha256: str
    target: tuple[tuple[str, str], ...]
    scope: str = "current-original-cast-construction-not-historical-executable"
    numerical_activation: bool = False
    nan_contract: str = "original-criterion-specific-no-payload-certification"


@dataclass(frozen=True, slots=True)
class SelectedCastEmission:
    contract: SelectedCastContract
    checked_nodes: int
    literal_roles: int
    constant_definitions: int
    facade_identity: bool
    structure: str = "validated"
    runtime: str = "unverified"


class _CastPermit:
    __slots__ = ("_seal", "_owner", "_context", "_handoff", "_input_value", "_audit")

    def __init__(
        self,
        seal: object,
        owner: _CastTransaction,
        context: object,
        handoff: Any,
        input_value: ir.Value,
        audit: SelectedCastContract,
    ) -> None:
        if seal is not _PERMIT_SEAL:
            raise ValueError("cast permit cannot be caller constructed")
        self._seal = seal
        self._owner: Any = owner
        self._context: Any = context
        self._handoff: Any = handoff
        self._input_value: Any = input_value
        self._audit = audit

    @property
    def audit(self) -> SelectedCastContract:
        return self._audit

    def clear(self) -> None:
        if self._handoff is not None:
            self._handoff.clear()
        self._seal = self._owner = self._context = self._handoff = self._input_value = (
            None
        )

    def __copy__(self) -> Any:
        raise TypeError("cast permit cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("cast permit cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("cast permit cannot be serialized")


class _CastTransaction:
    __slots__ = (
        "_seal",
        "_nonce",
        "_opset",
        "_handoff",
        "_proof",
        "_body_proof",
        "_released",
        "_bound",
        "_permit",
        "_context",
        "_audit",
        "_session",
        "_selected",
    )

    def __init__(self, seal: object, nonce: object, opset: int) -> None:
        if seal is not _TRANSACTION_SEAL:
            raise ValueError("cast transaction cannot be caller constructed")
        self._seal, self._nonce, self._opset = seal, nonce, opset
        self._handoff: Any = None
        self._proof = ""
        self._body_proof = ""
        self._released = self._bound = False
        self._permit: _CastPermit | None = None
        self._context: Any = None
        self._audit: SelectedCastContract | None = None
        self._session: _CastExecutionSession | None = None
        self._selected = False

    @property
    def audit(self) -> SelectedCastContract | None:
        return self._audit

    def stage(self, handoff: object, callable_digest: str, body_digest: str) -> None:
        if (
            self._seal is not _TRANSACTION_SEAL
            or self._released
            or self._handoff is not None
        ):
            raise ValueError("cast transaction already staged or closed")
        value = _require_native_handoff(handoff, self._nonce)
        if self._opset not in (21, 23) or any(
            type(value) is not str or len(value) != 64
            for value in (callable_digest, body_digest)
        ):
            value.clear()
            return
        self._handoff, self._proof = value, callable_digest
        self._body_proof = body_digest

    def finish_cleanup(self, held: bool) -> None:
        if self._seal is None:
            return
        if self._seal is not _TRANSACTION_SEAL or self._released:
            raise ValueError("cast transaction cleanup already finalized")
        if not held:
            self.discard()
            return
        if self._handoff is not None:
            _require_native_handoff(self._handoff, self._nonce)
        self._released = True

    def bind(self, context: object, body: Any) -> None:
        from .ir_context import IRContext
        from ._source_cast_context import effective_cpu_fp16

        if self._seal is None:
            return
        if self._seal is not _TRANSACTION_SEAL or not self._released or self._bound:
            raise ValueError("cast transaction is not released and unbound")
        self._bound = True
        if self._handoff is None:
            return
        handoff = _require_native_handoff(self._handoff, self._nonce)
        if (
            body.constvars
            or body.effects
            or len(body.invars) != 1
            or len(body.outvars) != 1
            or len(body.eqns) != 1
            or body.invars[0] is not handoff._input
            or body.outvars[0] is not handoff._output
            or body.eqns[0] is not handoff._equation
        ):
            self.discard()
            raise ValueError("actual original whole-root cast binding changed")
        if (
            type(context) is not IRContext
            or context.precision_policy is not PrecisionPolicy.SOURCE
            or context.builder.opset != self._opset
            or context._source_cast_permit is not None
        ):
            self.discard()
            raise ValueError("cast authority requires the actual new SOURCE context")
        target = dict(handoff.target)
        feature = effective_cpu_fp16(target.get("features", ""))
        if target.get("triple") != "x86_64-unknown-linux-gnu" or feature.status not in (
            "ENABLED",
            "DISABLED",
        ):
            self.discard()
            return
        value = context.require_value_for_var(handoff._input)
        if (
            value.dtype is not ir.DataType.DOUBLE
            or value.shape is None
            or tuple(value.shape) != handoff.record.module.shape
        ):
            self.discard()
            raise ValueError("actual SOURCE cast input binding differs")
        profile, record = handoff.profile, handoff.record
        audit = SelectedCastContract(
            (
                "DIRECT_RNE_53_TO_11"
                if feature.status == "ENABLED"
                else "DOUBLE_ROUND_RNE_53_TO_24_TO_11"
            ),
            record.module.shape,
            self._opset,
            profile.cohort,
            profile.shim_sha256,
            profile.common_sha256,
            profile.jax_commit,
            profile.xla_commit,
            profile.llvm_commit,
            profile.schema_protos,
            profile.provider_proof,
            profile.options_proof,
            profile.joined_proof,
            self._proof,
            self._body_proof,
            record.module.text_sha256,
            record.options_sha256,
            handoff.target_sha256,
            handoff.target,
        )
        self._audit = audit
        self._context = context
        self._permit = _CastPermit(_PERMIT_SEAL, self, context, handoff, value, audit)
        context._source_cast_permit = self._permit
        context._source_cast_audit = audit
        if self._opset == 23:
            self._session = _CastExecutionSession(
                _SESSION_SEAL, self, context, handoff._equation, value, audit
            )
            self._selected = True
            context._source_cast_selected = True
            context._source_cast_session = self._session
            context._source_cast_rule = self._session.rule
        self._handoff = None
        self._proof = ""
        self._body_proof = ""

    def _release_permit(self) -> None:
        if self._handoff is not None:
            self._handoff.clear()
        if (
            self._context is not None
            and self._context._source_cast_permit is self._permit
        ):
            self._context._source_cast_permit = None
        if self._permit is not None:
            self._permit.clear()
        self._handoff = self._permit = self._context = self._nonce = None
        self._proof = ""
        self._body_proof = ""
        self._seal = None
        self._released = True

    def discard(self) -> None:
        self._release_permit()
        if self._session is not None:
            self._session.close()
        self._session = None

    def __copy__(self) -> Any:
        raise TypeError("cast transaction cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("cast transaction cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("cast transaction cannot be serialized")


def _begin_cast_transaction(nonce: object, opset: int) -> _CastTransaction:
    return _CastTransaction(_TRANSACTION_SEAL, nonce, opset)


def _consume_cast_permit(
    context: object, equation: object
) -> SelectedCastContract | None:
    """Consume actual native provenance once; immutable records are not tokens."""
    from .ir_context import IRContext

    if type(context) is not IRContext:
        return None
    permit = context._source_cast_permit
    if type(permit) is not _CastPermit:
        return None
    owner = getattr(permit, "_owner", None)
    if (
        getattr(permit, "_seal", None) is not _PERMIT_SEAL
        or type(owner) is not _CastTransaction
        or owner._seal is not _TRANSACTION_SEAL
        or owner._permit is not permit
        or owner._context is not context
        or permit._context is not context
    ):
        context._source_cast_permit = None
        if type(owner) is _CastTransaction and owner._permit is permit:
            owner._audit = None
            if owner._context is not None:
                owner._context._source_cast_audit = None
            owner._release_permit()
        return None
    consumed = False
    try:
        # Read fields only from the exact immutable record issued by this
        # transaction. Equal or supported-looking caller records are evidence.
        audit = owner._audit
        if (
            type(audit) is not SelectedCastContract
            or permit._audit is not audit
            or context._source_cast_audit is not audit
        ):
            return None
        handoff = _require_native_handoff(permit._handoff, owner._nonce)
        if (
            context.precision_policy is not PrecisionPolicy.SOURCE
            or context.builder.opset != audit.opset
            or equation is not handoff._equation
            or context.require_value_for_var(handoff._input) is not permit._input_value
        ):
            return None
        value = permit._input_value
        if (
            value.dtype is not ir.DataType.DOUBLE
            or value.shape is None
            or tuple(value.shape) != audit.shape
        ):
            return None
        consumed = True
        return audit
    finally:
        if not consumed:
            owner._audit = None
            context._source_cast_audit = None
        owner._release_permit()


class _CastExecutionSession:
    __slots__ = (
        "_seal",
        "_owner",
        "_context",
        "_equation",
        "_value",
        "_audit",
        "_rule",
        "_input_witness",
        "_output_witness",
        "_state",
        "_receipt",
    )

    def __init__(
        self,
        seal: object,
        owner: _CastTransaction,
        context: Any,
        equation: Any,
        value: ir.Value,
        audit: SelectedCastContract,
    ) -> None:
        from jax2onnx.plugins.jax._source_f64_f16_structure import (
            freeze_selected_cast_rule,
        )
        from .source_structure import _witness

        if seal is not _SESSION_SEAL:
            raise SourceCastSelectionError(
                "selected-cast-session-cannot-be-constructed"
            )
        self._seal = seal
        self._owner: Any = owner
        self._context: Any = context
        self._equation: Any = equation
        self._value: Any = value
        self._audit: SelectedCastContract | None = audit
        self._rule: SelectedF64F16Rule | None = freeze_selected_cast_rule(
            equation, audit
        )
        self._input_witness: Any = _witness(value)
        self._output_witness: Any = None
        self._state = "prepared"
        self._receipt: SelectedCastEmission | None = None

    @property
    def rule(self) -> SelectedF64F16Rule | None:
        return self._rule

    @property
    def audit(self) -> SelectedCastContract | None:
        return self._audit

    def _current(
        self, context: object
    ) -> tuple[SelectedCastContract, SelectedF64F16Rule]:
        from jax2onnx.plugins.jax._source_f64_f16_structure import SelectedF64F16Rule
        from .ir_context import IRContext

        audit, rule = self._audit, self._rule
        if (
            self._seal is not _SESSION_SEAL
            or type(context) is not IRContext
            or context is not self._context
            or type(self._owner) is not _CastTransaction
            or self._owner._session is not self
            or not self._owner._selected
            or not context._source_cast_selected
            or context._source_cast_session is not self
            or type(audit) is not SelectedCastContract
            or self._owner._audit is not audit
            or context._source_cast_audit is not audit
            or audit.opset != 23
            or type(rule) is not SelectedF64F16Rule
            or rule.contract is not audit
            or context._source_cast_rule is not rule
            or context.builder.opset != 23
            or context.precision_policy is not PrecisionPolicy.SOURCE
        ):
            raise SourceCastSelectionError("selected-cast-session-binding-differs")
        return audit, rule

    def prepared_rule(self, context: object, equation: object) -> SelectedF64F16Rule:
        _, rule = self._current(context)
        if self._state != "prepared" or equation is not self._equation:
            raise SourceCastSelectionError("selected-cast-original-equation-differs")
        return rule

    def take(self, context: object, equation: object) -> SelectedCastContract:
        audit, _ = self._current(context)
        if (
            self._state != "prepared"
            or equation is not self._equation
            or self._owner._seal is not _TRANSACTION_SEAL
            or self._context.require_value_for_var(
                cast("JaxprEqn", self._equation).invars[0]
            )
            is not self._value
        ):
            raise SourceCastSelectionError(
                "selected-cast-permit-already-lost-or-consumed"
            )
        if _consume_cast_permit(context, equation) is not audit:
            raise SourceCastSelectionError("selected-cast-native-permit-unavailable")
        self._state = "taken"
        self._equation = self._value = None
        return audit

    def result(self, context: object, value: ir.Value) -> None:
        from .source_structure import _witness

        _, rule = self._current(context)
        if self._state != "taken" or not rule.result.matches(value):
            raise SourceCastSelectionError("selected-cast-result-binding-differs")
        self._output_witness = _witness(value)
        self._state = "emitted"

    def finalize(self, model: ir.Model) -> SelectedCastEmission:
        from .source_structure import _resolve_bindings

        audit, rule = self._current(self._context)
        if self._state != "emitted" or self._receipt is not None or model.functions:
            raise SourceCastSelectionError("selected-cast-emission-incomplete")
        wanted = {self._input_witness, self._output_witness}
        values = _resolve_bindings(model.graph, wanted)
        source, result = values[self._input_witness], values[self._output_witness]
        matched = rule.match(model.graph, source, result)
        self._receipt = SelectedCastEmission(
            audit,
            len(matched.nodes),
            len(matched.literal_obligations),
            len(matched.constants),
            model.graph.outputs[0] is not result,
        )
        self._state = "validated"
        return self._receipt

    def close(self) -> None:
        if self._context is not None:
            self._context._source_cast_session = None
            self._context._source_cast_rule = None
        self._seal = self._owner = self._context = self._equation = self._value = None
        self._audit = self._rule = self._input_witness = self._output_witness = None
        self._state = "closed"

    def __copy__(self) -> Any:
        raise TypeError("selected cast session cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("selected cast session cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("selected cast session cannot be serialized")


def _selected_cast_session(context: object) -> _CastExecutionSession | None:
    from .ir_context import IRContext

    if type(context) is not IRContext:
        return None
    session = context._source_cast_session
    if not context._source_cast_selected:
        if session is not None or context._source_cast_rule is not None:
            raise SourceCastSelectionError("selected-cast-session-without-producer")
        return None
    if type(session) is not _CastExecutionSession:
        raise SourceCastSelectionError("selected-cast-session-unavailable")
    return session


def _selected_cast_rule(context: object, equation: object) -> SelectedF64F16Rule | None:
    session = _selected_cast_session(context)
    return session.prepared_rule(context, equation) if session is not None else None


def _take_selected_cast(
    context: object, equation: object
) -> SelectedCastContract | None:
    session = _selected_cast_session(context)
    return session.take(context, equation) if session is not None else None


def _record_selected_cast_result(context: object, value: ir.Value) -> None:
    session = _selected_cast_session(context)
    if session is None:
        raise SourceCastSelectionError("selected-cast-result-without-session")
    session.result(context, value)


def finalize_selected_cast(
    model: ir.Model, transaction: _CastTransaction | None
) -> None:
    if transaction is not None and transaction._selected:
        if type(transaction._session) is not _CastExecutionSession:
            raise SourceCastSelectionError(
                "selected-cast-session-lost-before-finalization"
            )
        model.meta[EMISSION_KEY] = transaction._session.finalize(model)


def attach_cast_authority_audit(
    model: ir.Model, transaction: _CastTransaction | None
) -> None:
    if transaction is not None and transaction.audit is not None:
        model.meta[AUTHORITY_KEY] = transaction.audit
