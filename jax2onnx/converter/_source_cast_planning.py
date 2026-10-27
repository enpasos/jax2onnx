# jax2onnx/converter/_source_cast_planning.py

"""Private inactive whole-root cast plans, independent of ledger observation.

Observed target/query metadata is evidence, not source compilation authority.
This packet deliberately has no available authority or selected algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Final, TypeGuard

import onnx_ir as ir

from ._source_regions import FrozenBody, FrozenEquation, FrozenUse
from .source_structure import TensorContract

PLAN_KEY: Final[str] = "jax2onnx.source.whole_root_cast_plan"
_UNAVAILABLE: Final[str] = "default-source-compile-context-not-yet-proved"


class UnresolvedAlgorithm(Enum):
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class WholeRootCastEligibility:
    body: FrozenBody
    params: tuple[tuple[str, str | bool | None], ...]
    source_body_sha256: str
    root_ordinal: int = 0


@dataclass(frozen=True, slots=True)
class UnavailableSourceContext:
    reason: str = _UNAVAILABLE


@dataclass(frozen=True, slots=True)
class CompilationObservation:
    """Immutable evidence-only fields; none can authorize an algorithm."""

    acquisition: str = "unavailable"
    client_process_binding: tuple[int, int] | None = None
    target: tuple[tuple[str, str], ...] = ()
    build: tuple[tuple[str, str], ...] = ()
    options: tuple[tuple[str, str], ...] = ()
    source_hashes: tuple[tuple[str, str], ...] = ()
    source_body_sha256: str | None = None
    authority: UnavailableSourceContext | None = None


@dataclass(frozen=True, slots=True)
class WholeRootCastPlan:
    eligibility: WholeRootCastEligibility | None
    observation: CompilationObservation | None
    eligibility_sha256: str | None
    observation_sha256: str | None
    algorithm: UnresolvedAlgorithm
    reasons: tuple[str, ...]


def _sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _digest(value: object) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _tensor(value: TensorContract) -> tuple[object, ...]:
    return (
        value.logical_dtype,
        value.logical_shape,
        value.weak_type,
        int(value.dtype.value),
        value.shape,
        value.encoding,
        value.logical_symbol_axis,
        value.symbol_axis,
    )


def _static_tensor(value: object) -> TypeGuard[TensorContract]:
    return (
        type(value) is TensorContract
        and type(value.logical_dtype) is str
        and type(value.logical_shape) is tuple
        and type(value.shape) is tuple
        and all(
            type(dim) is int and dim >= 0
            for dim in (*value.logical_shape, *value.shape)
        )
        and type(value.weak_type) is bool
        and type(value.dtype) is ir.DataType
        and type(value.encoding) is str
        and value.symbol_token is None
        and value.logical_symbol_axis is None
        and value.symbol_axis is None
    )


def _params(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 3
        and all(type(pair) is tuple and len(pair) == 2 for pair in value)
        and all(type(pair[0]) is str for pair in value)
        and tuple(pair[0] for pair in value) == ("new_dtype", "sharding", "weak_type")
        and type(value[0][1]) is str
        and value[0][1] == "float16"
        and value[1][1] is None
        and type(value[2][1]) is bool
        and value[2][1] is False
    )


def _eligibility_payload(
    body: FrozenBody, params: tuple[tuple[str, str | bool | None], ...]
) -> tuple[object, ...]:
    if (
        type(body) is not FrozenBody
        or type(body.gaps) is not tuple
        or body.gaps
        or type(body.constants) is not tuple
        or body.constants
        or type(body.inputs) is not tuple
        or type(body.equations) is not tuple
        or type(body.outputs) is not tuple
        or len(body.inputs) != 1
        or len(body.equations) != 1
        or len(body.outputs) != 1
        or not _static_tensor(body.inputs[0])
    ):
        raise ValueError("whole-root-cast-body-unavailable")
    equation, source = body.equations[0], body.inputs[0]
    if not _static_tensor(source):
        raise ValueError("whole-root-cast-input-unavailable")
    if (
        type(equation) is not FrozenEquation
        or type(equation.inputs) is not tuple
        or len(equation.inputs) != 1
        or type(equation.outputs) is not tuple
        or len(equation.outputs) != 1
        or type(equation.children) is not tuple
        or equation.children
        or type(equation.semantics) is not tuple
        or len(equation.semantics) != 2
        or type(equation.semantics[0]) is not str
        or not _params(equation.semantics[1])
    ):
        raise ValueError("whole-root-cast-equation-unavailable")
    result = equation.outputs[0]
    if not _static_tensor(result):
        raise ValueError("whole-root-cast-result-unavailable")
    for use in (*equation.inputs, *body.outputs):
        if (
            type(use) is not FrozenUse
            or type(use.kind) is not str
            or type(use.index) is not int
            or type(use.port) is not int
            or not _static_tensor(use.tensor)
            or use.literal is not None
        ):
            raise ValueError("whole-root-cast-ssa-unavailable")
    if (
        source.logical_dtype != "float64"
        or source.dtype is not ir.DataType.DOUBLE
        or result.logical_dtype != "float16"
        or result.dtype is not ir.DataType.FLOAT16
        or source.weak_type is not False
        or result.weak_type is not False
        or source.encoding != "identity"
        or result.encoding != "identity"
        or source.symbol_token is not None
        or result.symbol_token is not None
        or source.logical_symbol_axis is not None
        or result.logical_symbol_axis is not None
        or source.symbol_axis is not None
        or result.symbol_axis is not None
        or source.logical_shape != source.shape
        or result.logical_shape != result.shape
        or source.shape != result.shape
        or any(type(dim) is not int or dim < 0 for dim in source.shape)
        or not _params(params)
        or equation.inputs != (FrozenUse("input", 0, 0, source),)
        or equation.children
        or equation.semantics != ("native-f64-f16-convert", params)
        or body.outputs != (FrozenUse("equation", 0, 0, result),)
    ):
        raise ValueError("whole-root-cast-contract-differs")
    return ("whole-root-dynamic-f64-f16-v1", _tensor(source), _tensor(result), params)


def make_cast_eligibility(
    body: FrozenBody, params: tuple[tuple[str, str | bool | None], ...]
) -> WholeRootCastEligibility:
    """Called by the plugin-owned freezer after native identity/SSA checks."""
    return WholeRootCastEligibility(
        body, params, _sha(_eligibility_payload(body, params))
    )


def _observation_payload(value: CompilationObservation) -> tuple[object, ...]:
    if type(value.acquisition) is not str or value.acquisition not in (
        "unavailable",
        "same-client-empty-query",
        "retained-topology",
        "captured-original-executable",
    ):
        raise ValueError("compilation-acquisition-unavailable")
    binding = value.client_process_binding
    if binding is not None and (
        type(binding) is not tuple
        or len(binding) != 2
        or any(type(item) is not int or item <= 0 for item in binding)
    ):
        raise ValueError("compilation-client-binding-unavailable")
    for fields in (value.target, value.build, value.options, value.source_hashes):
        if (
            type(fields) is not tuple
            or len(fields) > 64
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(item) is not str or len(item) > 8192 for item in pair)
                for pair in fields
            )
            or len({pair[0] for pair in fields}) != len(fields)
        ):
            raise ValueError("compilation-observation-fields-unavailable")
    if value.source_body_sha256 is not None and not _digest(value.source_body_sha256):
        raise ValueError("compilation-source-binding-unavailable")
    authority = value.authority
    if authority is not None and (
        type(authority) is not UnavailableSourceContext
        or type(authority.reason) is not str
        or authority.reason != _UNAVAILABLE
    ):
        raise ValueError("source-context-authority-unavailable")
    return (
        value.acquisition,
        binding,
        value.target,
        value.build,
        value.options,
        value.source_hashes,
        value.source_body_sha256,
        None if authority is None else authority.reason,
    )


def resolve_cast_plan(
    eligibility: object, observation: object = None
) -> WholeRootCastPlan:
    """Pure closed resolver: evidence-only records never select SOURCE math."""
    reasons: list[str] = []
    original: WholeRootCastEligibility | None = None
    observed: CompilationObservation | None = None
    original_sha = observed_sha = None
    if type(eligibility) is WholeRootCastEligibility:
        try:
            payload = _eligibility_payload(eligibility.body, eligibility.params)
            if (
                type(eligibility.root_ordinal) is not int
                or eligibility.root_ordinal != 0
            ):
                raise ValueError("whole-root-cast-ordinal-differs")
            if eligibility.source_body_sha256 != _sha(payload):
                raise ValueError("whole-root-cast-digest-differs")
            original, original_sha = eligibility, eligibility.source_body_sha256
        except (TypeError, ValueError, AttributeError):
            reasons.append("whole-root-cast-contract-differs")
    else:
        reasons.append("whole-root-dynamic-cast-unavailable")
    if observation is not None:
        if type(observation) is CompilationObservation:
            try:
                observed_sha = _sha(_observation_payload(observation))
                observed = observation
                if observation.source_body_sha256 not in (None, original_sha):
                    reasons.append("compilation-source-binding-differs")
            except (TypeError, ValueError, AttributeError):
                observed_sha = None
                reasons.append("compilation-context-record-unavailable")
        else:
            reasons.append("compilation-context-record-unavailable")
    reasons.append(_UNAVAILABLE)
    return WholeRootCastPlan(
        original,
        observed,
        original_sha,
        observed_sha,
        UnresolvedAlgorithm.UNRESOLVED,
        tuple(dict.fromkeys(reasons)),
    )


def plan_is_current(value: object) -> bool:
    if (
        type(value) is not WholeRootCastPlan
        or value.algorithm is not UnresolvedAlgorithm.UNRESOLVED
        or (
            value.eligibility is not None
            and type(value.eligibility) is not WholeRootCastEligibility
        )
        or (
            value.observation is not None
            and type(value.observation) is not CompilationObservation
        )
        or type(value.reasons) is not tuple
        or any(type(reason) is not str for reason in value.reasons)
        or (
            value.eligibility_sha256 is not None
            and not _digest(value.eligibility_sha256)
        )
        or (
            value.observation_sha256 is not None
            and not _digest(value.observation_sha256)
        )
    ):
        return False
    return value == resolve_cast_plan(value.eligibility, value.observation)


def prepare_root_cast_plan(closed: object) -> WholeRootCastPlan:
    # Semantic recognition belongs to the primitive plugin, not the converter.
    from jax2onnx.plugins.jax._source_f64_f16_structure import freeze_root_cast

    try:
        eligibility = freeze_root_cast(closed)
    except (ValueError, TypeError, AttributeError):
        eligibility = None
    return resolve_cast_plan(eligibility)


def attach_cast_plan(model: ir.Model, value: object) -> None:
    if not plan_is_current(value):
        raise ValueError("whole-root-cast-plan-differs")
    model.meta[PLAN_KEY] = value
