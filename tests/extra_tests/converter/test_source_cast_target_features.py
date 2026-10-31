# tests/extra_tests/converter/test_source_cast_target_features.py

"""Independent target-bit mathematics; no SOURCE compilation authority."""

from dataclasses import FrozenInstanceError
from typing import Any

import onnx_ir as ir
import pytest

from jax2onnx.converter._source_cast_context import EffectiveFp16, effective_cpu_fp16
from jax2onnx.converter import _source_cast_planning as planning
from jax2onnx.converter._source_regions import FrozenBody, FrozenEquation, FrozenUse
from jax2onnx.converter.source_structure import TensorContract


# Independent transcription of both pinned LLVM prerequisite graphs, retained in
# cast-fp16-feature-prerequisite-source-proof-v1.json (44799ba4563eb911b746d64f2d9e2033
# c2b2db2ed8bcb4aad41394542ab769e6). Do not derive this oracle from the implementation.
_PREREQUISITES = (
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
_CLOSURE = ("avx512fp16",) + _PREREQUISITES


def _stream(positive: tuple[str, ...] = (), negative: tuple[str, ...] = ()) -> str:
    """Construct valid fixture wire order, without deciding feature truth."""
    assert len(set(positive)) == len(positive)
    assert len(set(negative)) == len(negative)
    assert not set(positive).intersection(negative)
    return ",".join(
        ["+" + name for name in sorted(positive)]
        + ["-" + name for name in sorted(negative)]
    )


def _expect(
    features: Any, status: str, reasons: tuple[str, ...], tokens: tuple[str, ...]
) -> EffectiveFp16:
    result = effective_cpu_fp16(features)
    assert type(result) is EffectiveFp16
    assert type(result.status) is str and result.status == status
    assert type(result.reasons) is tuple and result.reasons == reasons
    assert type(result.canonical_features) is tuple
    assert result.canonical_features == tokens
    assert all(
        type(value) is str for value in result.reasons + result.canonical_features
    )
    return result


def _malformed(features: Any) -> None:
    _expect(features, "UNAVAILABLE", ("target-feature-stream-malformed",), ())


@pytest.mark.parametrize(
    "positive", [("avx512fp16",), _CLOSURE], ids=["implicit", "explicit"]
)
def test_explicit_fp16_enables_the_reviewed_transitive_closure(
    positive: tuple[str, ...],
) -> None:
    wire = _stream(positive)
    _expect(wire, "ENABLED", ("fp16-feature-enabled",), tuple(wire.split(",")))


@pytest.mark.parametrize("omitted", _PREREQUISITES)
def test_enabling_fp16_does_not_require_repeating_each_implicit_prerequisite(
    omitted: str,
) -> None:
    wire = _stream(tuple(name for name in _CLOSURE if name != omitted))
    _expect(wire, "ENABLED", ("fp16-feature-enabled",), tuple(wire.split(",")))


@pytest.mark.parametrize("disabled", _CLOSURE)
def test_every_negative_in_the_reviewed_closure_clears_the_bit_without_a_cpu_name(
    disabled: str,
) -> None:
    wire = "-" + disabled
    _expect(wire, "DISABLED", ("fp16-prerequisite-disabled:" + disabled,), (wire,))


@pytest.mark.parametrize("disabled", _PREREQUISITES)
def test_later_negative_prerequisites_override_explicit_positive_fp16(
    disabled: str,
) -> None:
    wire = _stream(("avx512fp16",), (disabled,))
    _expect(
        wire,
        "DISABLED",
        ("fp16-prerequisite-disabled:" + disabled,),
        tuple(wire.split(",")),
    )


@pytest.mark.parametrize(
    "disabled",
    [
        ("sse", "avx512bw"),
        ("fma", "avx", "sse4.2"),
        _CLOSURE,
    ],
    ids=["root-and-descendant", "three-ancestors", "all"],
)
def test_all_disabled_prerequisites_are_reported_in_name_order(
    disabled: tuple[str, ...],
) -> None:
    positive = () if "avx512fp16" in disabled else ("avx512fp16",)
    wire = _stream(positive + ("unrelated",), disabled + ("other",))
    _expect(
        wire,
        "DISABLED",
        tuple("fp16-prerequisite-disabled:" + name for name in sorted(disabled)),
        tuple(wire.split(",")),
    )


@pytest.mark.parametrize(
    "wire",
    [
        "",
        "+avx2,+f16c,+fma",
        "+avx512bw,+avx512f",
        "+unrelated,-other",
        "+avx512fp16x",
        "+xavx512fp16",
        "+avx512fp16-extra",
        "+avx512_fp16",
        "+amx-fp16",
        "-amx-fp16",
        "+64bit,+a.b-c_d,-zero",
        "+soft-float",
    ],
    ids=[
        "empty",
        "other-cast-features",
        "parents-only",
        "unrelated-signs",
        "suffix",
        "prefix",
        "hyphen-suffix",
        "underscore",
        "other-fp16-family",
        "other-fp16-negative",
        "legal-token-characters",
        "separate-soft-float-gate",
    ],
)
def test_absence_and_substrings_do_not_supply_the_decisive_feature(wire: str) -> None:
    _expect(
        wire,
        "UNAVAILABLE",
        ("fp16-feature-not-declared",),
        tuple(wire.split(",")) if wire else (),
    )


def test_all_proper_prerequisites_positive_still_do_not_declare_fp16() -> None:
    wire = _stream(_PREREQUISITES)
    _expect(wire, "UNAVAILABLE", ("fp16-feature-not-declared",), tuple(wire.split(",")))


@pytest.mark.parametrize("sign", ["+", "-"], ids=["positive", "negative"])
def test_soft_float_remains_a_separate_pipeline_gate_not_an_fp16_prerequisite(
    sign: str,
) -> None:
    wire = "+avx512fp16," + sign + "soft-float"
    _expect(wire, "ENABLED", ("fp16-feature-enabled",), tuple(wire.split(",")))


@pytest.mark.parametrize(
    "wire",
    [
        "+avx512fp16,+avx512fp16",
        "-sse,-sse",
        "+other,+other,-sse",
        "+avx512fp16,-avx512fp16",
        "+other,-other",
        "+avx512fp16,+other,-other",
        "-sse,+avx512fp16",
        "+z,+avx512fp16",
        "-sse,-avx",
        "+avx512fp16,-z,-avx",
        "+avx512fp16,-other,+z",
        "+avx512fp16,-sse,-sse",
    ],
    ids=[
        "duplicate-enable",
        "duplicate-disable",
        "unrelated-positive-duplicate",
        "contradict-fp16",
        "contradict-unrelated",
        "contradict-with-enable",
        "negative-before-positive",
        "positive-unsorted",
        "negative-unsorted",
        "negative-tail-unsorted",
        "positive-after-negative",
        "duplicate-after-disable",
    ],
)
def test_duplicates_contradictions_and_noncanonical_order_are_not_reinterpreted(
    wire: str,
) -> None:
    _malformed(wire)


@pytest.mark.parametrize(
    "wire",
    [
        "avx512fp16",
        "+",
        "-",
        "+avx512fp16,",
        ",+avx512fp16",
        "+avx512fp16,,+z",
        " +avx512fp16",
        "+avx512fp16 ",
        "+avx512fp16\n",
        "+avx512fp16\t",
        "+avx512fp16\x00",
        "+AVX512FP16",
        "++avx512fp16",
        "--avx512fp16",
        "+-avx512fp16",
        "+.avx512fp16",
        "+_avx512fp16",
        "+cpu=avx512fp16",
        "+x/avx512fp16",
        "+avx512fp16:1",
        "+avx512fp16;+z",
        "+avx512fp16 +z",
        "+avx512fp16, -sse",
        "+avx512fp１６",
        "+avx512fp16\r,+z",
    ],
    ids=[
        "unsigned",
        "bare-plus",
        "bare-minus",
        "trailing-comma",
        "leading-comma",
        "empty-token",
        "leading-space",
        "trailing-space",
        "newline",
        "tab",
        "nul",
        "uppercase",
        "double-plus",
        "double-minus",
        "mixed-signs",
        "leading-dot",
        "leading-underscore",
        "assignment",
        "slash",
        "colon",
        "semicolon",
        "space-separated",
        "token-space",
        "unicode-digits",
        "carriage-return",
    ],
)
def test_malformed_token_grammar_does_not_enable_or_disable(wire: str) -> None:
    _malformed(wire)


@pytest.mark.parametrize("length", [65536, 65537], ids=["exact-limit", "over-limit"])
def test_character_bound_is_exact_without_silent_truncation(length: int) -> None:
    prefix = "+avx512fp16,+"
    wire = prefix + "z" * (length - len(prefix))
    assert len(wire) == length
    if length == 65536:
        _expect(wire, "ENABLED", ("fp16-feature-enabled",), tuple(wire.split(",")))
    else:
        _malformed(wire)


@pytest.mark.parametrize("count", [4096, 4097], ids=["exact-limit", "over-limit"])
def test_token_bound_is_exact_without_silent_truncation(count: int) -> None:
    positive = ("avx512fp16",) + tuple(
        "f" + str(index).zfill(4) for index in range(count - 1)
    )
    wire = _stream(positive)
    assert len(wire) < 65536 and len(wire.split(",")) == count
    if count == 4096:
        _expect(wire, "ENABLED", ("fp16-feature-enabled",), tuple(wire.split(",")))
    else:
        _malformed(wire)


@pytest.mark.parametrize(
    "kind",
    [
        "none",
        "bool",
        "int",
        "bytes",
        "list",
        "tuple",
        "mapping",
        "string-subclass",
        "foreign",
    ],
)
def test_only_exact_strings_are_read_without_foreign_protocol_calls(kind: str) -> None:
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("foreign-protocol")
        raise AssertionError("foreign input protocol was executed")

    class ForeignString(str):
        __len__ = forbidden
        __str__ = forbidden
        __iter__ = forbidden
        __eq__ = forbidden
        __hash__ = forbidden
        split = forbidden

    class Foreign:
        __getattribute__ = forbidden
        __str__ = forbidden
        __len__ = forbidden
        __iter__ = forbidden
        __eq__ = forbidden

    # Keep foreign objects out of decorator values: collection must never inspect
    # them. The exact-type rejection itself must not inspect or coerce them either.
    values: dict[str, Any] = {
        "none": None,
        "bool": True,
        "int": 1,
        "bytes": b"+avx512fp16",
        "list": ["+avx512fp16"],
        "tuple": ("+avx512fp16",),
        "mapping": {"cpu": "label-a", "features": "+avx512fp16"},
        "string-subclass": ForeignString("+avx512fp16"),
        "foreign": Foreign(),
    }
    _malformed(values[kind])
    assert calls == []


def test_result_is_frozen_primitive_evidence_with_exact_signed_tokens() -> None:
    wire = "+avx512fp16,+unrelated,-other"
    result = _expect(wire, "ENABLED", ("fp16-feature-enabled",), tuple(wire.split(",")))
    with pytest.raises(FrozenInstanceError):
        result.status = "SOURCE_EQUIVALENT"  # type: ignore[misc]
    assert result.status == "ENABLED"
    assert result.canonical_features == ("+avx512fp16", "+unrelated", "-other")


def _synthetic_eligibility() -> planning.WholeRootCastEligibility:
    # This deliberately constructed, valid immutable contract proves only that
    # a filled metadata record cannot turn structural eligibility into authority.
    # It makes no claim of having captured or compiled an original source call.
    source = TensorContract(
        "float64", (6,), False, ir.DataType.DOUBLE, (6,), "identity", None, None, None
    )
    result = TensorContract(
        "float16", (6,), False, ir.DataType.FLOAT16, (6,), "identity", None, None, None
    )
    params = (("new_dtype", "float16"), ("sharding", None), ("weak_type", False))
    body = FrozenBody(
        (source,),
        (),
        (
            FrozenEquation(
                (FrozenUse("input", 0, 0, source),),
                (result,),
                ("native-f64-f16-convert", params),
            ),
        ),
        (FrozenUse("equation", 0, 0, result),),
    )
    return planning.make_cast_eligibility(body, params)


@pytest.mark.parametrize(
    "cpu",
    ["label-a", "label-b", "+avx512fp16"],
    ids=["label-a", "label-b", "feature-like-label"],
)
@pytest.mark.parametrize(
    "features,status",
    [("+avx512fp16", "ENABLED"), ("+avx512fp16,-sse", "DISABLED"), ("", "UNAVAILABLE")],
    ids=["enabled", "disabled", "absent"],
)
@pytest.mark.parametrize("authority", ["none", "unavailable"])
def test_cpu_labels_and_filled_feature_evidence_never_select_a_source_algorithm(
    cpu: str,
    features: str,
    status: str,
    authority: str,
) -> None:
    eligibility = _synthetic_eligibility()
    feature = effective_cpu_fp16(features)
    assert feature.status == status
    observation = planning.CompilationObservation(
        acquisition="captured-original-executable",  # Synthetic claim, not a capture.
        client_process_binding=(123, 456),
        target=(
            ("cpu", cpu),
            ("features", features),
            ("triple", "x86_64-unknown-linux-gnu"),
        ),
        build=(("jax", "fixture"), ("jaxlib", "fixture"), ("native_sha256", "a" * 64)),
        options=(
            ("effective_fp16", feature.status),
            ("source_equivalence", "unproved"),
        ),
        source_hashes=(("fixture", "b" * 64),),
        source_body_sha256=eligibility.source_body_sha256,
        authority=None if authority == "none" else planning.UnavailableSourceContext(),
    )
    plan = planning.resolve_cast_plan(eligibility, observation)
    assert plan.eligibility is eligibility and plan.observation is observation
    assert plan.eligibility_sha256 == eligibility.source_body_sha256
    assert plan.observation_sha256 is not None
    assert planning.plan_is_current(plan)
    assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert plan.reasons == ("default-source-compile-context-not-yet-proved",)
