# jax2onnx/converter/_source_cast_context.py

"""Evidence-only root cast acquisition and an independent target predicate.

No record in this module authorizes numerical selection. Native original-trace
lowering/options evidence remains separate from target-query construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import re
import sys
import threading
from typing import Any, Final

import onnx_ir as ir

from jax2onnx._compat.jax import (
    acquire_cpu_cast_metadata,
    begin_cpu_cast_context,
    begin_original_cast_lowering,
    lower_original_cast_jaxpr,
)
from jax2onnx._compat._cast_lowering import NativeCastLowering
from ._source_cast_planning import (
    CompilationObservation,
    WholeRootCastPlan,
    resolve_cast_plan,
)
from ._source_cast_provenance import CallableCastLease, CallableCastProof, proof_digest
from ._source_cast_authority import _CastTransaction

CONTEXT_KEY: Final[str] = "jax2onnx.source.cast_context_acquisition"
_PREREQUISITES: Final[frozenset[str]] = frozenset(
    (
        "avx512fp16",
        "avx512bw",
        "avx512f",
        "avx2",
        "fma",
        "f16c",
        "avx",
        "sse4.2",
        "sse4.1",
        "ssse3",
        "sse3",
        "sse2",
        "sse",
    )
)


@dataclass(frozen=True, slots=True)
class EffectiveFp16:
    status: str
    reasons: tuple[str, ...]
    canonical_features: tuple[str, ...]


def effective_cpu_fp16(features: str) -> EffectiveFp16:
    """Interpret a canonical target feature stream, never a CPU model name."""
    malformed = EffectiveFp16("UNAVAILABLE", ("target-feature-stream-malformed",), ())
    if type(features) is not str or len(features) > 65536:
        return malformed
    tokens = tuple(features.split(",")) if features else ()
    if len(tokens) > 4096 or any(
        re.fullmatch(r"[+-][a-z0-9][a-z0-9_.-]*", token) is None for token in tokens
    ):
        return malformed
    positives = tuple(token[1:] for token in tokens if token[0] == "+")
    negatives = tuple(token[1:] for token in tokens if token[0] == "-")
    canonical = tuple("+" + name for name in sorted(set(positives))) + tuple(
        "-" + name for name in sorted(set(negatives))
    )
    if tokens != canonical or set(positives).intersection(negatives):
        return malformed
    disabled = sorted(_PREREQUISITES.intersection(negatives))
    if disabled:
        return EffectiveFp16(
            "DISABLED",
            tuple("fp16-prerequisite-disabled:" + name for name in disabled),
            tokens,
        )
    if "avx512fp16" in positives:
        return EffectiveFp16("ENABLED", ("fp16-feature-enabled",), tokens)
    return EffectiveFp16("UNAVAILABLE", ("fp16-feature-not-declared",), tokens)


@dataclass(frozen=True, slots=True)
class RootCastAcquisition:
    callable_proof: CallableCastProof | None
    native_fields: tuple[tuple[str, str], ...]
    native_options: bytes
    target_payload: bytes
    effective_fp16: EffectiveFp16
    original_logging: str
    original_input_shardings: str
    original_output_shardings: str
    reasons: tuple[str, ...]
    original_lowering: NativeCastLowering | None = None


class RootCastContextLease:
    """Conversion-owned acquisition; no caller-supplied observation is trusted."""

    def __init__(
        self,
        function: object,
        *,
        has_input_params: bool,
        inputs: object = None,
        transaction: _CastTransaction | None = None,
    ) -> None:
        self._profiles: Any = (sys.getprofile(), threading.getprofile())
        self._transaction = transaction
        self._callable = CallableCastLease(function)
        self._native: Any = None
        self._lowering: Any = None
        self._lowering_reason: str | None = None
        self._finished = False
        self._closed = False
        self.record: RootCastAcquisition | None = None
        self.plan: WholeRootCastPlan | None = None
        self._reason = (
            "fixed-input-parameters-unavailable"
            if has_input_params
            else self._callable.reason
        )
        if self._reason is None:
            try:
                self._native = begin_cpu_cast_context()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                self._reason = "native-default-context-unavailable"
            if inputs is not None and self._native is not None:
                try:
                    state = self._callable._state
                    if state is None:
                        raise ValueError("original callable state unavailable")
                    self._lowering = begin_original_cast_lowering(
                        inputs, state.function.__name__
                    )
                    if self._transaction is not None:
                        self._lowering.bind_transaction(self._transaction._nonce)
                except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                    self._lowering_reason = (
                        "original-input-or-native-lowering-bindings-unavailable"
                    )

    def finish(
        self, plan: WholeRootCastPlan, *, closed: object = None
    ) -> WholeRootCastPlan:
        if self._finished or self._closed:
            raise ValueError("cast acquisition lease finalized or closed")
        self._finished = True
        reasons = [
            "original-mlir-logging-unavailable",
            "original-resolved-shardings-unavailable",
        ]
        fields: tuple[tuple[str, str], ...] = ()
        options = target = b""
        feature = effective_cpu_fp16("")
        proof = self._callable.proof
        lowering: NativeCastLowering | None = None
        try:
            if (
                sys.getprofile() is not self._profiles[0]
                or threading.getprofile() is not self._profiles[1]
            ):
                raise ValueError("ambient profile identity changed")
            if self._reason is not None:
                raise ValueError(self._reason)
            self._callable.assert_current()
            if plan.eligibility is None or self._native is None:
                raise ValueError("whole-root-cast-eligibility-unavailable")
            acquired = acquire_cpu_cast_metadata(self._native)
            self._callable.assert_current()
            fields, options, target = (
                acquired.fields,
                acquired.options,
                acquired.target_payload,
            )
            feature = effective_cpu_fp16(dict(fields)["features"])
            if self._lowering is not None and closed is not None:
                # A caller-filled eligibility cannot substitute a different
                # native tree for this actual original trace.
                from ._source_cast_planning import prepare_root_cast_plan

                actual_plan = prepare_root_cast_plan(closed)
                if actual_plan.eligibility != plan.eligibility:
                    raise ValueError("original native trace binding changed")
                lowering = lower_original_cast_jaxpr(
                    self._native, closed, self._lowering
                )
                self._callable.assert_current()
                if self._transaction is not None:
                    handoff = self._lowering.take_handoff(
                        self._transaction._nonce, lowering
                    )
                    if handoff is not None:
                        if proof is None:
                            handoff.clear()
                            raise ValueError("original callable proof unavailable")
                        self._transaction.stage(
                            handoff,
                            proof_digest(proof),
                            actual_plan.eligibility.source_body_sha256,
                        )
                reasons.clear()
            elif self._lowering_reason is not None:
                reasons.append(self._lowering_reason)
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as error:
            if self._transaction is not None:
                self._transaction.discard()
            lowering = None
            reasons.append(
                self._reason
                or "native-cast-acquisition-unavailable:" + type(error).__name__
            )
            try:
                self._callable.assert_current()
            except (ValueError, TypeError, RuntimeError, AttributeError):
                proof = None
        self.record = RootCastAcquisition(
            proof,
            fields,
            options,
            target,
            feature,
            "AVAILABLE" if lowering is not None else "UNAVAILABLE",
            "AVAILABLE" if lowering is not None else "UNAVAILABLE",
            "AVAILABLE" if lowering is not None else "UNAVAILABLE",
            tuple(reasons),
            lowering,
        )
        source_options = (
            (
                ("original_native_mlir_sha256", lowering.module.text_sha256),
                ("original_native_options_sha256", lowering.options_sha256),
                ("original_logging", str(lowering.detailed_logging)),
                ("original_input_propagation", repr(lowering.input_propagation)),
                ("original_output_propagation", repr(lowering.output_propagation)),
                ("original_options_scope", lowering.options_scope),
            )
            if lowering is not None
            else ()
        )
        observation = CompilationObservation(
            acquisition=dict(fields).get("acquisition", "unavailable"),
            client_process_binding=(
                (os.getpid(), int(dict(fields)["client_identity"])) if fields else None
            ),
            target=tuple(
                (key, value)
                for key, value in fields
                if key in ("triple", "cpu", "features")
            ),
            build=tuple(
                (key, value)
                for key, value in fields
                if key in ("jax", "jaxlib", "native_sha256")
            ),
            options=tuple(
                (key, value)
                for key, value in fields
                if key.startswith("options_")
                or (key.startswith("original_") and not source_options)
            )
            + source_options,
            source_hashes=(
                (("callable", proof_digest(proof)),) if proof is not None else ()
            ),
            source_body_sha256=plan.eligibility_sha256,
        )
        self.plan = resolve_cast_plan(plan.eligibility, observation)
        return self.plan

    def close(self) -> None:
        if self._closed:
            return
        completed = False
        try:
            held = (
                self._profiles is not None
                and sys.getprofile() is self._profiles[0]
                and threading.getprofile() is self._profiles[1]
            )
            try:
                # Temporary query/lowering owners have already been dropped by
                # their finally blocks. Keep every verifier alive through the last
                # source/config/environment/factory/build fence, then release.
                if self._callable.proof is not None:
                    self._callable.assert_current()
                if self._lowering is not None:
                    self._lowering.assert_current()
                if self._native is not None:
                    held = bool(self._native.close() and held)
                if self._lowering is not None:
                    self._lowering.assert_current()
                if self._callable.proof is not None:
                    self._callable.assert_current()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                held = False
            finally:
                try:
                    if self._native is not None and not self._native._closed:
                        held = bool(self._native.close() and held)
                finally:
                    self._native = None
                    try:
                        if self._lowering is not None:
                            self._lowering.close()
                    finally:
                        self._lowering = None
                        self._callable.close()
            held = bool(
                held
                and self._profiles is not None
                and sys.getprofile() is self._profiles[0]
                and threading.getprofile() is self._profiles[1]
            )
            try:
                if self._transaction is not None:
                    self._transaction.finish_cleanup(held)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                held = False
                if self._transaction is not None:
                    self._transaction.discard()
            if not held and self.record is not None:
                self.record = replace(
                    self.record,
                    original_logging="UNAVAILABLE",
                    original_input_shardings="UNAVAILABLE",
                    original_output_shardings="UNAVAILABLE",
                    original_lowering=None,
                    reasons=(
                        *self.record.reasons,
                        (
                            "ambient-profile-changed-during-cleanup"
                            if (
                                self._profiles is not None
                                and (
                                    sys.getprofile() is not self._profiles[0]
                                    or threading.getprofile() is not self._profiles[1]
                                )
                            )
                            else "cast-context-changed-during-cleanup"
                        ),
                    ),
                )
                if self.plan is not None and self.plan.observation is not None:
                    observation = self.plan.observation
                    options = tuple(
                        (key, value)
                        for key, value in observation.options
                        if not key.startswith("original_")
                    )
                    observation = replace(
                        observation,
                        options=(
                            *options,
                            ("original_logging", "unavailable"),
                            ("original_resolved_shardings", "unavailable"),
                            ("original_availability", "failed-during-cleanup"),
                        ),
                    )
                    self.plan = resolve_cast_plan(self.plan.eligibility, observation)
            completed = True
        finally:
            if not completed and self._transaction is not None:
                self._transaction.discard()
            self._transaction = None
            self._profiles = None
            self._closed = True


def attach_cast_acquisition(
    model: ir.Model, record: RootCastAcquisition | None
) -> None:
    if type(record) is not RootCastAcquisition:
        raise ValueError("cast acquisition record unavailable")
    model.meta[CONTEXT_KEY] = record
