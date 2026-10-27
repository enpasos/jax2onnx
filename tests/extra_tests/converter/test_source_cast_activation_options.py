# tests/extra_tests/converter/test_source_cast_activation_options.py

"""Unused options semantics, independent of source/build/kernel authority."""

from dataclasses import fields
import hashlib
import json
import struct
import sys
from collections.abc import Callable
from typing import Any

import pytest

from jax2onnx._compat import _cast_activation_options as options


_VERSIONS = ("0.10.2", "0.12.0.1")


def _integer(value: int) -> bytes:
    value &= (1 << 64) - 1
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _v(number: int, value: int | bool) -> bytes:
    return _integer(number << 3) + _integer(value)


def _b(number: int, value: bytes) -> bytes:
    return _integer((number << 3) | 2) + _integer(len(value)) + value


def _f(number: int, value: float) -> bytes:
    return _integer((number << 3) | 5) + struct.pack("<f", value)


# Independently declared reviewed route; this is untrusted synthetic input,
# never a claim that a native constructor produced these bytes.
_DEBUG_REQUIRED = {
    9: False,
    31: 3,
    70: True,
    71: True,
    72: True,
    73: False,
    97: False,
    99: False,
    104: False,
    108: 0,
    142: -1,
    187: 0,
    307: False,
    308: 256,
    323: 32,
    333: "",
    335: False,
    337: False,
    344: False,
    359: True,
    363: False,
    364: False,
    365: 0,
    370: False,
    379: False,
    380: False,
    397: False,
    399: (),
    400: (),
    408: False,
    412: False,
    419: False,
    422: (4, 5, 3),
    425: True,
    429: False,
    439: False,
    448: 0,
    467: 0,
    479: True,
    493: True,
}
_OLD_REQUIRED = {300: True, 376: True, 452: False, 489: False, 512: False}
_MODERN_REQUIRED = {
    452: True,
    489: True,
    496: "",
    512: False,
    529: True,
    539: False,
    543: "",
    544: "",
    547: 0,
}


def _encoded(number: int, value: Any) -> bytes:
    if isinstance(value, str):
        return _b(number, value.encode())
    if isinstance(value, tuple):
        return _b(number, b"".join(_integer(item) for item in value)) if value else b""
    return _v(number, value)


def _packet(
    schema: str,
    *,
    debug: dict[int, bytes | None] | None = None,
    build: dict[int, bytes | None] | None = None,
    outer: dict[int, bytes | None] | None = None,
) -> bytes:
    declared = dict(_DEBUG_REQUIRED)
    declared.update(_MODERN_REQUIRED if schema == "0.12.0.1" else _OLD_REQUIRED)
    debug_fields = {
        number: _encoded(number, value) for number, value in declared.items()
    }
    for number, raw in (debug or {}).items():
        debug_fields[number] = raw or b""
    device = _v(1, 1) + _v(2, 1) + _b(3, _b(1, _integer(0)))
    build_fields = {
        1: _v(1, -1),
        3: _b(3, b"".join(debug_fields.values())),
        4: _v(4, 1),
        5: _v(5, 1),
        6: _v(6, True),
        9: _b(9, device),
        12: _b(12, b"\x01"),
        18: _b(18, b"\x00"),
        19: _v(19, True),
        23: _v(23, 1),
        25: _v(25, 29),
    }
    for number, raw in (build or {}).items():
        build_fields[number] = raw or b""
    outer_fields = {3: _b(3, b"".join(build_fields.values())), 5: _v(5, -1)}
    for number, raw in (outer or {}).items():
        outer_fields[number] = raw or b""
    return b"".join(outer_fields.values())


def _facts(
    result: options.CurrentCastOptions, scope: str
) -> dict[int, options.OptionFact]:
    return {fact.field: fact for fact in result.facts if fact.scope == scope}


@pytest.mark.parametrize("schema", _VERSIONS)
def test_supported_semantic_input_retains_bytes_facts_and_no_authority(
    schema: str,
) -> None:
    payload = _packet(schema)
    result = options.validate_current_cast_options(payload, schema)
    assert result.status == "SUPPORTED_CURRENT_OPTIONS" and result.reasons == ()
    assert result.payload == payload
    assert result.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.schema == schema and result.pending_bindings
    assert result.scope == "untrusted-current-options-semantics-not-kernel-authority"
    assert not {field.name for field in fields(result)}.intersection(
        {"algorithm", "issuer", "capability", "approved_build", "authorized"}
    )
    debug = _facts(result, "DebugOptions")
    assert len(debug) == (349 if schema == "0.12.0.1" else 311)
    assert debug[422].value == (4, 5, 3)
    assert debug[493].present and debug[493].value is True
    assert _facts(result, "ExecutableBuildOptions")[25].value == 29


_REQUIRED_MUTATIONS = tuple(
    (schema, number, value)
    for schema in _VERSIONS
    for number, value in (
        _DEBUG_REQUIRED | (_MODERN_REQUIRED if schema == "0.12.0.1" else _OLD_REQUIRED)
    ).items()
)


@pytest.mark.parametrize("schema,number,value", _REQUIRED_MUTATIONS)
def test_each_reviewed_debug_semantic_or_route_change_is_unverified(
    schema: str, number: int, value: Any
) -> None:
    changed = (
        (
            not value
            if type(value) is bool
            else (
                "changed"
                if type(value) is str
                else tuple(reversed(value)) if value else (2,)
            )
        )
        if type(value) in (bool, str, tuple)
        else value + 1
    )
    payload = _packet(schema, debug={number: _encoded(number, changed)})
    result = options.validate_current_cast_options(payload, schema)
    assert result.status == "UNVERIFIED"
    assert f"unreviewed-current-option:DebugOptions.{number}" in result.reasons


@pytest.mark.parametrize("schema", _VERSIONS)
@pytest.mark.parametrize("number", [31, 70, 71, 72, 142, 308, 323, 359, 425, 493])
def test_omission_never_invents_nonzero_cpp_factory_defaults(
    schema: str, number: int
) -> None:
    result = options.validate_current_cast_options(
        _packet(schema, debug={number: None}), schema
    )
    assert result.status == "UNVERIFIED"
    assert _facts(result, "DebugOptions")[number].present is False


@pytest.mark.parametrize("schema,number", [("0.10.2", 376), ("0.12.0.1", 529)])
def test_epoch_specific_true_emitter_route_must_really_be_present(
    schema: str, number: int
) -> None:
    result = options.validate_current_cast_options(
        _packet(schema, debug={number: None}), schema
    )
    fact = _facts(result, "DebugOptions")[number]
    assert result.status == "UNVERIFIED" and not fact.present and fact.value is False


@pytest.mark.parametrize("schema", _VERSIONS)
def test_proto_zero_is_effective_when_fast_math_fields_are_absent(schema: str) -> None:
    result = options.validate_current_cast_options(
        _packet(schema, debug={99: None, 335: None}), schema
    )
    assert result.status == "SUPPORTED_CURRENT_OPTIONS"
    assert _facts(result, "DebugOptions")[99].value is False
    assert _facts(result, "DebugOptions")[99].present is False


@pytest.mark.parametrize("schema", _VERSIONS)
@pytest.mark.parametrize("number", [120, 121, 126, 129, 122, 140, 252])
def test_reviewed_honor_operation_and_logging_fields_are_not_guessed_restrictions(
    schema: str, number: int
) -> None:
    result = options.validate_current_cast_options(
        _packet(schema, debug={number: _v(number, False)}), schema
    )
    assert result.status == "SUPPORTED_CURRENT_OPTIONS"
    assert _facts(result, "DebugOptions")[number].value is False


@pytest.mark.parametrize("schema", _VERSIONS)
@pytest.mark.parametrize(
    "number,raw",
    [(24, _v(24, 29)), (25, _v(25, 0)), (25, b""), (4, b""), (5, b"")],
)
def test_actual_optimization_memory_and_replica_defaults_are_required(
    schema: str, number: int, raw: bytes
) -> None:
    result = options.validate_current_cast_options(
        _packet(schema, build={number: raw}), schema
    )
    assert result.status == "UNVERIFIED"
    assert (
        f"unreviewed-current-option:ExecutableBuildOptions.{number}" in result.reasons
    )


@pytest.mark.parametrize("number", [20, 21])
@pytest.mark.parametrize("value", [0.25, float("inf"), float("nan")])
def test_old_effort_fixed32_values_require_finite_zero(
    number: int, value: float
) -> None:
    result = options.validate_current_cast_options(
        _packet("0.10.2", build={number: _f(number, value)}), "0.10.2"
    )
    assert result.status == ("UNVERIFIED" if value == 0.25 else "FAILED")


def test_old_explicit_negative_zero_effort_is_finite_semantic_zero() -> None:
    assert (
        options.validate_current_cast_options(
            _packet("0.10.2", build={20: _f(20, -0.0), 21: _f(21, 0.0)}), "0.10.2"
        ).status
        == "SUPPORTED_CURRENT_OPTIONS"
    )


@pytest.mark.parametrize(
    "schema,scope,number",
    [
        ("0.12.0.1", "debug", 300),
        ("0.12.0.1", "debug", 376),
        ("0.12.0.1", "build", 20),
        ("0.12.0.1", "build", 21),
        ("0.12.0.1", "build", 26),
        ("0.10.2", "build", 27),
        ("0.10.2", "outer", 12),
        ("0.10.2", "debug", 600),
        ("0.12.0.1", "debug", 600),
    ],
)
def test_closed_catalog_rejects_unknown_and_version_reserved_numbers(
    schema: str, scope: str, number: int
) -> None:
    result = options.validate_current_cast_options(
        _packet(schema, **{scope: {number: _v(number, 0)}}), schema
    )
    assert result.status == "UNVERIFIED"
    assert "unknown-or-reserved-field" in result.reasons[0]


@pytest.mark.parametrize("raw", [_b(547, b"0"), _v(547, 3), _v(547, 1)])
def test_modern_547_is_the_exact_imported_enum_not_a_string(raw: bytes) -> None:
    result = options.validate_current_cast_options(
        _packet("0.12.0.1", debug={547: raw}), "0.12.0.1"
    )
    assert result.status == ("UNVERIFIED" if raw == _v(547, 1) else "FAILED")


@pytest.mark.parametrize("schema", _VERSIONS)
def test_repeated_selector_order_and_packed_unpacked_forms(schema: str) -> None:
    raw = _v(422, 4) + _b(422, b"\x05") + _v(422, 3)
    result = options.validate_current_cast_options(
        _packet(schema, debug={422: raw}), schema
    )
    assert result.status == "SUPPORTED_CURRENT_OPTIONS"
    assert _facts(result, "DebugOptions")[422].value == (4, 5, 3)
    assert (
        options.validate_current_cast_options(
            _packet(schema, debug={422: _b(422, b"\x03\x04\x05")}), schema
        ).status
        == "UNVERIFIED"
    )


@pytest.mark.parametrize(
    "schema,scope,number,raw",
    [
        ("0.10.2", "debug", 99, _v(99, 2)),
        ("0.12.0.1", "debug", 99, _b(99, b"\x00")),
        ("0.12.0.1", "debug", 333, _b(333, b"\xff")),
        ("0.10.2", "build", 20, _v(20, 0)),
        ("0.12.0.1", "build", 4, _b(4, b"\x01")),
        ("0.10.2", "debug", 99, _v(99, 0) + _v(99, 0)),
        ("0.12.0.1", "outer", 5, _v(5, -1) + _v(5, -1)),
    ],
)
def test_types_utf8_duplicates_and_boolean_domain_cannot_be_coerced(
    schema: str, scope: str, number: int, raw: bytes
) -> None:
    assert (
        options.validate_current_cast_options(
            _packet(schema, **{scope: {number: raw}}), schema
        ).status
        == "FAILED"
    )


@pytest.mark.parametrize(
    "payload",
    [b"\x08", b"\x1a\x05\x08", b"\x80\x00", b"\x80" * 10, b"\x0b", b"\x00"],
)
def test_malformed_and_truncated_wire_bytes_fail_closed(payload: bytes) -> None:
    assert options.validate_current_cast_options(payload, "0.12.0.1").status == "FAILED"


def test_exact_input_types_and_resource_bounds_do_not_call_foreign_objects() -> None:
    calls = []

    class Foreign:
        def __len__(self) -> int:
            calls.append("len")
            raise AssertionError("foreign input evaluated")

        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("foreign schema coerced")

    foreign: Any = Foreign()
    assert options.validate_current_cast_options(foreign, "0.12.0.1").status == "FAILED"
    assert options.validate_current_cast_options(b"", foreign).status == "UNVERIFIED"
    assert (
        options.validate_current_cast_options(b"x" * 65537, "0.12.0.1").status
        == "FAILED"
    )
    assert calls == []


@pytest.mark.parametrize("schema", _VERSIONS)
@pytest.mark.parametrize(
    "scope,number",
    [
        ("debug", 30),
        ("debug", 124),
        ("debug", 500),
        ("outer", 7),
        ("build", 13),
        ("build", 14),
        ("outer", 6),
        ("outer", 11),
    ],
)
def test_custom_override_fdo_environment_and_variant_routes_are_unverified(
    schema: str, scope: str, number: int
) -> None:
    raw = _b(number, b"" if number in (13, 11) else b"changed")
    if number in (500, 7):
        value = b"value" if number == 500 else _v(2, 1)
        raw = _b(number, _b(1, b"key") + _b(2, value))
    assert (
        options.validate_current_cast_options(
            _packet(schema, **{scope: {number: raw}}), schema
        ).status
        == "UNVERIFIED"
    )


def test_duplicate_map_keys_and_entry_fields_cannot_hide_overrides() -> None:
    entry = _b(1, b"same") + _b(2, b"value")
    for raw in (_b(500, entry) * 2, _b(500, entry + _b(1, b"other"))):
        assert (
            options.validate_current_cast_options(
                _packet("0.12.0.1", debug={500: raw}), "0.12.0.1"
            ).status
            == "FAILED"
        )


@pytest.mark.parametrize("scope,number", [("outer", 1), ("outer", 8), ("build", 2)])
def test_unparsed_layout_and_target_messages_remain_explicitly_unverified(
    scope: str, number: int
) -> None:
    result = options.validate_current_cast_options(
        _packet("0.12.0.1", **{scope: {number: _b(number, b"")}}), "0.12.0.1"
    )
    assert result.status == "UNVERIFIED"
    assert any(
        reason.startswith("unreviewed-present-option:") for reason in result.reasons
    )


@pytest.mark.parametrize(
    "raw",
    [
        _b(30, b"") * 4097,
        _b(422, b"\x01" * 4097),
        _b(333, b"x" * 4097),
    ],
    ids=["repeated-strings", "packed-selector", "oversized-string"],
)
def test_aggregate_repetition_and_string_bounds_apply_before_route_acceptance(
    raw: bytes,
) -> None:
    number = (
        30
        if raw.startswith(_integer((30 << 3) | 2))
        else (422 if raw.startswith(_integer((422 << 3) | 2)) else 333)
    )
    assert (
        options.validate_current_cast_options(
            _packet("0.12.0.1", debug={number: raw}), "0.12.0.1"
        ).status
        == "FAILED"
    )


@pytest.mark.parametrize("number,raw", [(62, _v(62, 2)), (31, _v(31, (1 << 32) - 1))])
def test_irrelevant_bool_and_signed_int32_still_require_exact_wire_domains(
    number: int, raw: bytes
) -> None:
    assert (
        options.validate_current_cast_options(
            _packet("0.12.0.1", debug={number: raw}), "0.12.0.1"
        ).status
        == "FAILED"
    )


def test_known_nested_debug_message_is_parsed_instead_of_ignored_as_a_blob() -> None:
    malformed_range = _b(424, _v(1, 0))
    assert (
        options.validate_current_cast_options(
            _packet("0.12.0.1", debug={424: malformed_range}), "0.12.0.1"
        ).status
        == "FAILED"
    )
    valid_range = _b(424, _b(1, _v(1, -1) + _v(2, 10)))
    assert (
        options.validate_current_cast_options(
            _packet("0.12.0.1", debug={424: valid_range}), "0.12.0.1"
        ).status
        == "SUPPORTED_CURRENT_OPTIONS"
    )


@pytest.mark.parametrize("schema", _VERSIONS)
def test_device_and_propagation_facts_require_real_closed_singleton_structure(
    schema: str,
) -> None:
    for change in ({9: _b(9, _v(1, 1))}, {12: _b(12, b"\x01\x00")}, {18: None}):
        assert (
            options.validate_current_cast_options(
                _packet(schema, build=change), schema
            ).status
            == "UNVERIFIED"
        )


# Retained actual serialized executable/query options; neither is a current
# conversion-owned original-trace acquisition. Provenance is retained externally.
_RETAINED = {
    ("0.10.2", "original"): (
        "1ae50708ffffffffffffffffff011210100a1a01062a060a01008001013201001aa407f8"
        "0103980201e00300ea030e2e2f637564615f73646b5f6c6962b00401b80401c00401c804"
        "00880600980600a00600b00601c80600d00600a00700c00701c80701d00701d80704f007"
        "01880801980800a008ffffffffffffffffff01c80800d00800e00800f008ffffffffffff"
        "ffffff01800900a80900e8098780800f980affffffffffffffffff01a00a01b00a01d00a"
        "01a80b00b00b00c80b01d00b00d80b00e00b01e80b01f00b01d80c00e80c01800d05880d"
        "01920d00a00d8780800fa80d8780800fc00d01c80d00d80d00800e008d0ecdcc8c3f980e"
        "00a00e80808004b80e01d80e00e00e00e80effffffffffffffff7f800f00b00f00b80f01"
        "c00f00d00f01d80f0fe00f00e80f01f00f00f80f00801000881000921007050208030709"
        "01981001a0105faa1000b01000c810a08d06d81000e01000e81000801101881100901100"
        "a81100c01101d81164e01100e81100f81100801200901200981200a81210b01203c01200"
        "e01201e81200801301881300a0138002b21300b81310c01300d81300e513cdcccc3de813"
        "00f01305981420a01401b8140ac0141ec81400d81440ea1400f01400f814008015018815"
        "009815ffffffffffffffffff01a01500a81501b01501b81500c01500c81500d81500e015"
        "00e81500f01500f81500881600a01601aa16130a0d6368756e6b5f707265705f75731202"
        "2d31aa16160a106368756e6b5f73697a655f627974657312022d31aa16130a0d67707573"
        "5f7065725f6e6f646512022d31aa16170a116e63636c5f6f705f6c61756e63685f757312"
        "022d31aa16140a0e6e69635f73706565645f6762707312022d31aa160c0a067274745f75"
        "7312022d31b81601d01601d81600e81600f01600801700901700a01700b01700b81701c0"
        "1701ca1700d0178780800fd81700e01700e81701f01700fa1700801800981801a01800a8"
        "1801b01801b81814c01828c81800d81800e81800881900981900a01902b01900ba1900c0"
        "1900c81901d01901d81900e01900e81900f01900881a28901a14981a00a01a01a81a00b2"
        "1a03040503c81a01d01a00e01a00e81a00f81a00801b00981b01a01b01a81b01b01b880e"
        "b81b00c01b00c81b00e01b00801c00881c80808002901c019a1c03696e66a01c00a81c00"
        "b01c00b81c00c01c00c81c00d81c20e01c01e81c8010f81c00881d02901d01a81d00b01d"
        "00b81d00c01d00c81d01d81d00e01d01f01d00f81d01801e01901e01981e02a01e00a81e"
        "01b01e00b81e00c01e00c81e00d01e00e01e00e81e01a81f01b01f008020002001280130"
        "014a09080110011a030a010062010192010101980101b80101c8011d28ffffffffffffff"
        "ffff014801"
    ),
    ("0.10.2", "query"): (
        "1ac40708ffffffffffffffffff011202100d1aa407f80103980201e00300ea030e2e2f63"
        "7564615f73646b5f6c6962b00401b80401c00401c80400880600980600a00600b00601c8"
        "0600d00600a00700c00701c80701d00701d80704f00701880801980800a008ffffffffff"
        "ffffffff01c80800d00800e00800f008ffffffffffffffffff01800900a80900e8098780"
        "800f980affffffffffffffffff01a00a01b00a01d00a01a80b00b00b00c80b01d00b00d8"
        "0b00e00b01e80b01f00b01d80c00e80c01800d05880d01920d00a00d8780800fa80d8780"
        "800fc00d01c80d00d80d00800e008d0ecdcc8c3f980e00a00e80808004b80e01d80e00e0"
        "0e00e80effffffffffffffff7f800f00b00f00b80f01c00f00d00f01d80f0fe00f01e80f"
        "01f00f00f80f0080100088100092100705020803070901981001a0105faa1000b01000c8"
        "10a08d06d81000e01000e81000801101881100901100a81100c01101d81164e01100e811"
        "00f81100801200901200981200a81210b01203c01200e01201e81200801301881300a013"
        "8002b21300b81310c01300d81300e513cdcccc3de81300f01305981420a01401b8140ac0"
        "141ec81400d81440ea1400f01400f814008015018815009815ffffffffffffffffff01a0"
        "1500a81501b01501b81500c01500c81500d81500e01500e81500f01500f81500881600a0"
        "1601aa16130a0d6368756e6b5f707265705f757312022d31aa16160a106368756e6b5f73"
        "697a655f627974657312022d31aa16130a0d677075735f7065725f6e6f646512022d31aa"
        "16170a116e63636c5f6f705f6c61756e63685f757312022d31aa16140a0e6e69635f7370"
        "6565645f6762707312022d31aa160c0a067274745f757312022d31b81601d01601d81600"
        "e81600f01600801700901700a01700b01700b81701c01701ca1700d0178780800fd81700"
        "e01700e81701f01700fa1700801800981801a01800a81801b01801b81814c01828c81800"
        "d81800e81800881900981900a01902b01900ba1900c01900c81901d01901d81900e01900"
        "e81900f01900881a28901a14981a00a01a01a81a00b21a03040503c81a01d01a00e01a00"
        "e81a00f81a00801b00981b01a01b01a81b01b01b880eb81b00c01b00c81b00e01b00801c"
        "00881c80808002901c019a1c03696e66a01c00a81c00b01c00b81c00c01c00c81c00d81c"
        "20e01c01e81c8010f81c00881d02901d01a81d00b01d00b81d00c01d00c81d01d81d00e0"
        "1d01f01d00f81d01801e01901e01981e02a01e00a81e01b01e00b81e00c01e00c81e00d0"
        "1e00e01e00e81e01a81f01b01f008020002001280162010092010100b801014801"
    ),
    ("0.12.0.1", "original"): (
        "1af7081210100a1a01062a060a01008001013201001ac108f80103980201e00300ea030e"
        "2e2f637564615f73646b5f6c6962b00401b80401c00401c80400880600980600a00600b0"
        "0601c80601d00600a00700c00701c80701d00701d80704f00701880801980800a008ffff"
        "ffffffffffffff01c80800d00800e00800f008ffffffffffffffffff01800900a80900e8"
        "098780800f980affffffffffffffffff01a00a01b00a01d00a01a80b00b00b00c80b01d8"
        "0b00e00b01e80b01f00b01d80c00e80c01800d05880d01920d00a00d8780800fa80d8780"
        "800fc00d01d80d00800e008d0ecdcc8c3fa00e80808004e00e00e80effffffffffffffff"
        "7f800f00b00f00b80f01c00f00d00f01d80f0fe00f00e80f01f00f00f80f008010008810"
        "009210080d05020803070901981001a0105faa1000b01000c810a08d06d81000e8100080"
        "1101881100901100a81100c01101d81164e01100e81100f81100801200901200981200a8"
        "1210b01203c01200e81200801301881300a0138002b21300b81310c01300d81300e513cd"
        "cccc3de81300f01305981420a01401b8140ac0141ec81400d81440ea1400f01400f81400"
        "8015018815009815ffffffffffffffffff01a01500a81501b01501b81500c01500c81500"
        "d81500e01500e81500f01500f81500881600a01601aa16130a0d6368756e6b5f70726570"
        "5f757312022d31aa16160a106368756e6b5f73697a655f627974657312022d31aa16130a"
        "0d677075735f7065725f6e6f646512022d31aa16170a116e63636c5f6f705f6c61756e63"
        "685f757312022d31aa16140a0e6e69635f73706565645f6762707312022d31aa160c0a06"
        "7274745f757312022d31b81601d01601d81600e81600f01600801700901700a01700b017"
        "00b81701ca1700d0178780800fd81700e01700e81701f01700fa1700801800b01801b818"
        "14c01828c81800d81800e81800881900981900a01902b01900c01900c81901d01901d819"
        "00e01900f01900f81901881a28901a14981a00a81a00b21a03040503c81a01d01a00e01a"
        "00e81a00f81a00801b00a01b01a81b01b01b880eb81b00c01b00c81b00d21b0f01020406"
        "0708090a0b0d0e10111213e01b00801c00881c808080029a1c03696e66a01c01a81c00b0"
        "1c00b81c00c01c01c81c00d81c20e01c01e81c8010f01c01f81c00881d02901d01a81d00"
        "b01d00b81d00c01d00c81d01d81d01e01d01f01d00f81d01901e01981e02a01e00a81e01"
        "b01e00b81e00c01e00c81e01d01e00e01e00e81e01f01e01f81e018a1f0108901f00a81f"
        "01b01f00c01f01c81f01d01f00802000882000902080808020982001a2200101a82000b0"
        "2000b82002c02000c82000d020ffffffffffffffffff01d82000e02000e82000f020ffff"
        "ffffffffffffff01f8200180218050882101902100982100a02100a82100b02101b82101"
        "c22100d02101d82100ea210108f021ffffffffffffffff7ffa2100822200882200902200"
        "982200a82200b02200b82200c02200ca2203696e66d22200d82200e02201e82200200128"
        "0130014a09080110011a030a010062010192010101980101b80101c8011d28ffffffffff"
        "ffffffff014801"
    ),
    ("0.12.0.1", "query"): (
        "1ae1081202100d1ac108f80103980201e00300ea030e2e2f637564615f73646b5f6c6962"
        "b00401b80401c00401c80400880600980600a00600b00601c80601d00600a00700c00701"
        "c80701d00701d80704f00701880801980800a008ffffffffffffffffff01c80800d00800"
        "e00800f008ffffffffffffffffff01800900a80900e8098780800f980affffffffffffff"
        "ffff01a00a01b00a01d00a01a80b00b00b00c80b01d80b00e00b01e80b01f00b01d80c00"
        "e80c01800d05880d01920d00a00d8780800fa80d8780800fc00d01d80d00800e008d0ecd"
        "cc8c3fa00e80808004e00e00e80effffffffffffffff7f800f00b00f00b80f01c00f00d0"
        "0f01d80f0fe00f01e80f01f00f00f80f008010008810009210080d050208030709019810"
        "01a0105faa1000b01000c810a08d06d81000e81000801101881100901100a81100c01101"
        "d81164e01100e81100f81100801200901200981200a81210b01203c01200e81200801301"
        "881300a0138002b21300b81310c01300d81300e513cdcccc3de81300f01305981420a014"
        "01b8140ac0141ec81400d81440ea1400f01400f814008015018815009815ffffffffffff"
        "ffffff01a01500a81501b01501b81500c01500c81500d81500e01500e81500f01500f815"
        "00881600a01601aa16130a0d6368756e6b5f707265705f757312022d31aa16160a106368"
        "756e6b5f73697a655f627974657312022d31aa16130a0d677075735f7065725f6e6f6465"
        "12022d31aa16170a116e63636c5f6f705f6c61756e63685f757312022d31aa16140a0e6e"
        "69635f73706565645f6762707312022d31aa160c0a067274745f757312022d31b81601d0"
        "1601d81600e81600f01600801700901700a01700b01700b81701ca1700d0178780800fd8"
        "1700e01700e81701f01700fa1700801800b01801b81814c01828c81800d81800e8180088"
        "1900981900a01902b01900c01900c81901d01901d81900e01900f01900f81901881a2890"
        "1a14981a00a81a00b21a03040503c81a01d01a00e01a00e81a00f81a00801b00a01b01a8"
        "1b01b01b880eb81b00c01b00c81b00d21b0f010204060708090a0b0d0e10111213e01b00"
        "801c00881c808080029a1c03696e66a01c01a81c00b01c00b81c00c01c01c81c00d81c20"
        "e01c01e81c8010f01c01f81c00881d02901d01a81d00b01d00b81d00c01d00c81d01d81d"
        "01e01d01f01d00f81d01901e01981e02a01e00a81e01b01e00b81e00c01e00c81e01d01e"
        "00e01e00e81e01f01e01f81e018a1f0108901f00a81f01b01f00c01f01c81f01d01f0080"
        "2000882000902080808020982001a2200101a82000b02000b82002c02000c82000d020ff"
        "ffffffffffffffff01d82000e02000e82000f020ffffffffffffffffff01f82001802180"
        "50882101902100982100a02100a82100b02101b82101c22100d02101d82100ea210108f0"
        "21ffffffffffffffff7ffa2100822200882200902200982200a82200b02200b82200c022"
        "00ca2203696e66d22200d82200e02201e82200200128014a09080110011a030a01006201"
        "0092010100b801014801"
    ),
}


@pytest.mark.parametrize("schema", _VERSIONS)
def test_retained_historical_and_query_bytes_do_not_become_current_authority(
    schema: str,
) -> None:
    original = bytes.fromhex(_RETAINED[(schema, "original")])
    query = bytes.fromhex(_RETAINED[(schema, "query")])
    before = options.validate_current_cast_options(original, schema)
    other = options.validate_current_cast_options(query, schema)
    assert original != query and before.payload_sha256 != other.payload_sha256
    # Stored executable layouts and query construction are different scopes.
    assert before.status == other.status == "UNVERIFIED"
    assert before.pending_bindings == other.pending_bindings
    assert _facts(before, "DebugOptions")[252].value is False
    assert _facts(other, "DebugOptions")[252].value is True
    assert _facts(before, "ExecutableBuildOptions")[25].value == 29
    assert _facts(other, "ExecutableBuildOptions")[25].value == 0


def test_actual_conversion_owned_current_options_without_extra_source_work(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    import jax
    import numpy as np
    from jax._src import compiler
    from jax._src.interpreters import pxla
    from jaxlib import _jax as native

    from jax2onnx.converter import _source_cast_context as context
    from jax2onnx.converter import _source_cast_planning as planning
    from tests.extra_tests.converter.test_source_cast_native_lowering import (
        _export,
        _factory,
    )

    function = _factory()
    acquire_original = context.acquire_cpu_cast_metadata
    lower_original = context.lower_original_cast_jaxpr
    calls, events, denied = [], [], []

    def observer(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is function.__code__:
            calls.append("original-trace")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        denied.append("extra-source-work")
        raise AssertionError("options validation must not compile or execute source")

    def acquire(*args: Any, **kwargs: Any) -> Any:
        result = acquire_original(*args, **kwargs)
        events.append(dict(result.fields)["work"])
        return result

    def lower(*args: Any, **kwargs: Any) -> Any:
        with monkeypatch.context() as patch:
            patch.setattr(compiler, "compile_or_get_cached", forbidden)
            patch.setattr(compiler, "backend_compile_and_load", forbidden)
            patch.setattr(pxla.MeshComputation, "compile", forbidden)
            patch.setattr(native.Client, "compile_and_load", forbidden)
            patch.setattr(native.Client, "serialize_executable", forbidden)
            patch.setattr(native.LoadedExecutable, "execute_sharded", forbidden)
            result = lower_original(*args, **kwargs)
        assert result.resolver_requests == 1
        assert result.source_compile_requests == result.source_execute_requests == 0
        return result

    previous = sys.getprofile()
    try:
        sys.setprofile(observer)
        with jax.enable_x64(True), monkeypatch.context() as patch:
            patch.setattr(context, "acquire_cpu_cast_metadata", acquire)
            patch.setattr(context, "lower_original_cast_jaxpr", lower)
            model = _export(function, jax.ShapeDtypeStruct((6,), np.float64))
    finally:
        sys.setprofile(previous)
    assert calls == ["original-trace"] and denied == []
    assert events == [
        (
            "compile=1,serialize-executable=1,execute=0"
            if jax.__version__ == "0.10.2"
            else "compile=0,serialize-executable=0,execute=0"
        )
    ]
    record = model.meta[context.CONTEXT_KEY]
    assert record.reasons == () and record.original_lowering is not None
    current = record.original_lowering
    assert current.options_scope == (
        "current-native-original-trace-default-construction-not-historical-executable"
    )
    checked = options.validate_current_cast_options(current.options, jax.__version__)
    assert checked.status == "SUPPORTED_CURRENT_OPTIONS", checked.reasons
    assert checked.payload_sha256 == current.options_sha256
    debug = _facts(checked, "DebugOptions")
    build = _facts(checked, "ExecutableBuildOptions")
    assert debug[252].value is current.detailed_logging
    assert build[18].value == current.input_propagation
    assert build[12].value == current.output_propagation
    assert checked.pending_bindings
    record_property(
        "current-original-cast-options-semantic-input",
        json.dumps(
            {
                "schema": checked.schema,
                "options_scope": current.options_scope,
                "options_hex": current.options.hex(),
                "options_sha256": current.options_sha256,
                "status": checked.status,
                "pending_bindings": checked.pending_bindings,
                "original_source_calls": len(calls),
                "source_compile_requests": current.source_compile_requests,
                "source_execute_requests": current.source_execute_requests,
                "query_work": events,
                "kernel_authority": False,
            },
            sort_keys=True,
        ),
    )
    assert (
        model.meta[planning.PLAN_KEY].algorithm
        is planning.UnresolvedAlgorithm.UNRESOLVED
    )
    assert [node.op_type for node in model.graph] == ["Cast"]
