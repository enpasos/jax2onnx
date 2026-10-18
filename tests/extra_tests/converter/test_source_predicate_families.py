# tests/extra_tests/converter/test_source_predicate_families.py

"""Native bindings and admitted predicate/bitwise value-flow contracts.

Native tests preserve the original 650-case binder/trace evidence. Product tests
use the common private facade and sample API; their exact discrete controls stay
separate from floating composition criteria and unfinished semantic families.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
import ast
import json
from pathlib import Path
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx import onnx_function
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl
from jax2onnx.precision import SampleCriteria
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _artifact,
    _accept,
    _FINITE,
)


_COMPARISONS = ("equal", "not_equal", "greater", "greater_equal", "less", "less_equal")
_BITWISE_BINARY = ("bitwise_and", "bitwise_or", "bitwise_xor")
_BITWISE_UNARY = ("bitwise_not", "invert")
_LOGICAL = ("logical_and", "logical_or", "logical_xor", "logical_not")
_OWNED = tuple(name for name in _COMPARISONS if name != "not_equal") + (
    *_BITWISE_BINARY,
    *_BITWISE_UNARY,
)
_ALL = (*_COMPARISONS, *_BITWISE_BINARY, *_BITWISE_UNARY, *_LOGICAL)
_ORIGINALS = {name: getattr(jnp, name) for name in (*_ALL, "where", "exp", "multiply")}
_CLASSES = {
    "equal": "JnpEqualPlugin",
    "greater": "JnpGreaterPlugin",
    "greater_equal": "JnpGreaterEqualPlugin",
    "less": "JnpLessPlugin",
    "less_equal": "JnpLessEqualPlugin",
    "bitwise_and": "JnpBitwiseAndPlugin",
    "bitwise_or": "JnpBitwiseOrPlugin",
    "bitwise_xor": "JnpBitwiseXorPlugin",
    "bitwise_not": "JnpBitwiseNotPlugin",
    "invert": "JnpInvertPlugin",
}
_INTEGER_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")

# Frozen before target execution. No tolerance applies to predicate bits or
# integer data. The two float rows distinguish NaN classification from exact
# storage/zero-sign promises; they do not waive either comparison result.
_REFERENCE_CRITERIA = {
    "predicate_bool": SampleCriteria("bitwise"),
    "bitwise_bool": SampleCriteria("bitwise"),
    "bitwise_integer": SampleCriteria("bitwise"),
    "integer_identity": SampleCriteria("bitwise"),
    "float_nan_class": SampleCriteria("exact", equal_nan=True),
    "float_identity_and_signed_zero": SampleCriteria("bitwise"),
}


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _settings() -> tuple[Any, ...]:
    return (
        bool(jax.config.x64_enabled),
        jax.config.jax_default_matmul_precision,
        jax.config.jax_numpy_dtype_promotion,
        jax.config.jax_numpy_rank_promotion,
        jax.config.jax_platforms,
    )


def _assert_originals() -> None:
    assert all(getattr(jnp, name) is original for name, original in _ORIGINALS.items())


@pytest.fixture(autouse=True)
def _restore_context() -> Iterator[None]:
    previous_globals, previous_effective = _globals(), _settings()
    previous_policy = active_precision_policy()
    _assert_originals()
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in previous_globals.items():
            jax.config.update(name, value)
        assert _globals() == previous_globals
        assert _settings() == previous_effective
        assert active_precision_policy() is previous_policy
        _assert_originals()


@contextmanager
def _source() -> Iterator[None]:
    settings, globals_before = _settings(), _globals()
    policy = active_precision_policy()
    try:
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            yield
    finally:
        assert _settings() == settings and _globals() == globals_before
        assert active_precision_policy() is policy
        _assert_originals()


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    aval = jax.typeof(value)
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)


def _target(name: str) -> Callable[..., Any]:
    def model(*values: Any) -> Any:
        return getattr(jnp, name)(*values)

    return model


def _payloads(values: Any) -> list[np.ndarray]:
    return [np.asarray(value) for value in jax.tree.leaves(values)]


def _assert_payloads(actual: list[np.ndarray], expected: list[np.ndarray]) -> None:
    for left, right in zip(actual, expected, strict=True):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()


def _native_reference(
    fn: Callable[..., Any], values: tuple[Any, ...]
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    _assert_originals()
    trace = typed_trace_manifest(jax.make_jaxpr(fn)(*map(_spec, values)))
    eager = _payloads(fn(*values))
    compiled = _payloads(jax.jit(fn)(*values))
    _assert_payloads(compiled, eager)
    return trace, eager, compiled


def _assert_native_forwarding(
    fn: Callable[..., Any], values: tuple[Any, ...]
) -> dict[str, Any]:
    reference, eager, compiled = _native_reference(fn, values)
    with _source():
        actual = typed_trace_manifest(jax.make_jaxpr(fn)(*map(_spec, values)))
        assert actual == reference
        _assert_payloads(_payloads(fn(*values)), eager)
        _assert_payloads(_payloads(jax.jit(fn)(*values)), compiled)
    return reference


def _values(dtype: str) -> tuple[Any, Any]:
    if dtype == "bool":
        left, right = [False, False, True, True], [False, True, False, True]
    elif dtype in _INTEGER_DTYPES:
        info = np.iinfo(dtype)
        left = np.array(
            [info.min, info.min + 1, info.max - 1, info.max, 0, 1], dtype=dtype
        )
        right = np.array(
            [info.min + 1, info.min, info.max, info.max - 1, 1, 0], dtype=dtype
        )
    else:
        left, right = [-1.5, -0.0, 0.0, 2.0], [2.0, 0.0, -0.0, -1.5]
    return jnp.asarray(left, dtype=dtype), jnp.asarray(right, dtype=dtype)


def test_criteria_are_exact_and_separate_nan_class_from_storage() -> None:
    assert {criteria.mode for criteria in _REFERENCE_CRITERIA.values()} == {
        "exact",
        "bitwise",
    }
    assert _REFERENCE_CRITERIA["float_nan_class"].equal_nan is True
    assert all(
        criteria.mode == "bitwise"
        for name, criteria in _REFERENCE_CRITERIA.items()
        if name != "float_nan_class"
    )


def test_only_the_sixteen_reviewed_jnp_binders_opt_in() -> None:
    directory = Path(__file__).resolve().parents[3] / "jax2onnx/plugins/jax/numpy"
    selected = []
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "jnp_binding_specs"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg == "source_native":
                    assert isinstance(keyword.value, ast.Constant)
                    assert keyword.value.value is True
                    selected.append(path.stem)
    assert sorted(selected) == sorted(
        (*_OWNED, "add", "cos", "exp", "maximum", "minimum", "sin")
    )


@pytest.mark.parametrize("name", _OWNED)
@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
@pytest.mark.parametrize("failure", [False, True])
def test_actual_family_binders_call_original_or_custom_once(
    name: str, policy: PrecisionPolicy, failure: bool, monkeypatch: Any
) -> None:
    module = import_module(f"jax2onnx.plugins.jax.numpy.{name}")
    plugin = getattr(module, _CLASSES[name])
    primitive = plugin._PRIM
    left, right, option, result = object(), object(), object(), object()
    args = (left,) if name in _BITWISE_UNARY else (left, right)
    kwargs = {"original_keyword": option}
    error = ValueError(f"unchanged {name} error")
    calls = []

    def invoke(kind: str, *actual_args: Any, **actual_kwargs: Any) -> Any:
        calls.append((kind, actual_args, actual_kwargs))
        if failure:
            raise error
        return result

    def original(*actual_args: Any, **actual_kwargs: Any) -> Any:
        return invoke("native", *actual_args, **actual_kwargs)

    def custom(*actual_args: Any, **actual_kwargs: Any) -> Any:
        return invoke("custom", *actual_args, **actual_kwargs)

    # make_value stores its original on the actual production primitive. Track
    # that slot as well as bind so running this test alone leaves no spy behind.
    storage_slot = f"__orig_impl___{name}"
    monkeypatch.setattr(
        primitive, storage_slot, getattr(primitive, storage_slot, None), raising=False
    )
    monkeypatch.setattr(primitive, "bind", custom)
    spec = next(
        spec for spec in plugin.binding_specs() if isinstance(spec, MonkeyPatchSpec)
    )
    assert spec.attr == name
    wrapper = spec.make_value(original)
    with precision_policy_scope(policy):
        if failure:
            with pytest.raises(ValueError) as actual:
                wrapper(*args, **kwargs)
            assert actual.value is error
        else:
            assert wrapper(*args, **kwargs) is result
    assert calls == [
        ("native" if policy is PrecisionPolicy.SOURCE else "custom", args, kwargs)
    ]
    assert get_orig_impl(primitive, name) is original


@pytest.mark.parametrize("name", _COMPARISONS)
@pytest.mark.parametrize("dtype", ("bool", *_INTEGER_DTYPES, *_FLOAT_DTYPES))
@pytest.mark.parametrize("x64", [False, True])
def test_comparison_dtype_extrema_and_native_weakness(
    name: str, dtype: str, x64: bool, record_property: Any
) -> None:
    with jax.enable_x64(x64):
        trace = _assert_native_forwarding(_target(name), _values(dtype))
    # Native output weakness is observed, never inferred from arithmetic rules.
    assert trace["outputs"][0]["aval"][0] == "bool"
    record_property(
        "native-signatures",
        json.dumps({"inputs": trace["inputs"], "outputs": trace["outputs"]}),
    )


@pytest.mark.parametrize("name", (*_BITWISE_BINARY, *_BITWISE_UNARY))
@pytest.mark.parametrize("dtype", ("bool", *_INTEGER_DTYPES))
@pytest.mark.parametrize("x64", [False, True])
def test_bitwise_dtype_truth_tables_and_integer_extrema(
    name: str, dtype: str, x64: bool
) -> None:
    with jax.enable_x64(x64):
        values = _values(dtype)
        if name in _BITWISE_UNARY:
            values = values[:1]
        _assert_native_forwarding(_target(name), values)


@pytest.mark.parametrize("name", _ALL)
@pytest.mark.parametrize("shape", ["scalar", "empty", "same", "broadcast", "rank3"])
def test_native_static_shape_families(name: str, shape: str) -> None:
    shapes = {
        "scalar": ((), ()),
        "empty": ((0, 3), (1, 3)),
        "same": ((2, 3), (2, 3)),
        "broadcast": ((2, 3), (1, 3)),
        "rank3": ((2, 1, 3), (1, 4, 1)),
    }
    left_shape, right_shape = shapes[shape]
    values = (
        jnp.ones(left_shape, dtype=np.int32),
        jnp.zeros(right_shape, dtype=np.int32),
    )
    if name in (*_BITWISE_UNARY, "logical_not"):
        values = values[:1]
    _assert_native_forwarding(_target(name), values)


@pytest.mark.parametrize("name", (*_COMPARISONS, *_BITWISE_BINARY, *_BITWISE_UNARY))
@pytest.mark.parametrize("batch", [2, 5])
def test_native_leading_batch_and_explicit_one_broadcast(name: str, batch: int) -> None:
    values = (jnp.ones((batch, 2, 3), np.int32), jnp.zeros((1, 2, 3), np.int32))
    if name in _BITWISE_UNARY:
        values = values[:1]
    _assert_native_forwarding(_target(name), values)


@pytest.mark.parametrize("name", (*_COMPARISONS, *_BITWISE_BINARY, *_BITWISE_UNARY))
@pytest.mark.parametrize("scope_case", ["shared", "leading_one", "unrelated"])
def test_native_symbolic_trace_transparency_without_shape_admission(
    name: str, scope_case: str, record_property: Any
) -> None:
    shape = export.symbolic_shape("B, 3")
    right_shape = {
        "shared": shape,
        "leading_one": (1, 3),
        "unrelated": export.symbolic_shape("B, 3"),
    }[scope_case]
    specs = (
        jax.ShapeDtypeStruct(shape, np.int32),
        jax.ShapeDtypeStruct(right_shape, np.int32),
    )
    if name in _BITWISE_UNARY:
        specs = specs[:1]
    fn = _target(name)
    _assert_originals()
    # Preserve actual SymbolicScope objects into tracing. This checks binder
    # transparency, including the native unrelated-scope error where present;
    # product validation and its finite shape proof remain a separate gate.
    try:
        reference = typed_trace_manifest(jax.make_jaxpr(fn)(*specs))
    except Exception as reference_error:
        with _source():
            with pytest.raises(type(reference_error)) as actual_error:
                jax.make_jaxpr(fn)(*specs)
        assert (
            str(actual_error.value).splitlines()[0]
            == str(reference_error).splitlines()[0]
        )
        record_property("native-symbolic-outcome", type(reference_error).__name__)
    else:
        with _source():
            assert typed_trace_manifest(jax.make_jaxpr(fn)(*specs)) == reference
        record_property("native-symbolic-outcome", json.dumps(reference))


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_native_captured_payloads_direct_outputs_and_repeated_aliases(
    dtype: str,
) -> None:
    with jax.enable_x64(True):
        values = _values(dtype)
        captured = np.asarray(values[1]).copy()
        original_bytes = captured.tobytes()

        def model(left: Any, right: Any) -> Any:
            mask = jnp.greater_equal(left, captured)
            bits = jnp.bitwise_xor(left, right)
            return left, left, captured, captured, mask, mask, bits, bits

        _assert_native_forwarding(model, values)
        assert captured.tobytes() == original_bytes


@pytest.mark.parametrize("name", _OWNED)
def test_native_vmap_paths(name: str) -> None:
    values = (
        jnp.asarray([[1, 2], [3, 4]], np.int32),
        jnp.asarray([[4, 3], [2, 1]], np.int32),
    )
    if name in _BITWISE_UNARY:
        values = values[:1]
    _assert_native_forwarding(jax.vmap(_target(name)), values)


@pytest.mark.parametrize("name", (*_COMPARISONS, *_BITWISE_BINARY, *_BITWISE_UNARY))
@pytest.mark.parametrize(
    "case", ["weak_weak", "strong_weak", "weak_strong", "strong_scalar"]
)
@pytest.mark.parametrize("x64", [False, True])
def test_native_weak_scalar_rules(
    name: str, case: str, x64: bool, record_property: Any
) -> None:
    with jax.enable_x64(x64):
        weak, strong = jnp.asarray(1), jnp.asarray(1, dtype=np.int16)
        pairs = {
            "weak_weak": (weak, jnp.asarray(2)),
            "strong_weak": (jnp.asarray([1, 2], dtype=np.int16), weak),
            "weak_strong": (weak, jnp.asarray([1, 2], dtype=np.int16)),
            "strong_scalar": (strong, strong),
        }
        values = pairs[case]
        if name in _BITWISE_UNARY:
            values = values[:1]
        trace = _assert_native_forwarding(_target(name), values)
        record_property(
            "native-weakness",
            json.dumps({"inputs": trace["inputs"], "outputs": trace["outputs"]}),
        )


@pytest.mark.parametrize("name", _COMPARISONS)
@pytest.mark.parametrize(
    "case", ["weak_float16", "int_float", "signed_unsigned", "int64_float64"]
)
def test_native_mixed_promotion_and_explicit_trace_boundaries(
    name: str, case: str
) -> None:
    with jax.enable_x64(True):
        pairs = {
            "weak_float16": (jnp.asarray([1, 2], np.float16), jnp.asarray(1.0)),
            "int_float": (
                jnp.asarray([16777217, -1], np.int32),
                jnp.asarray([16777216, -1], np.float32),
            ),
            "signed_unsigned": (
                jnp.asarray([-1, 1], np.int32),
                jnp.asarray([2**32 - 1, 1], np.uint32),
            ),
            "int64_float64": (
                jnp.asarray([2**53 + 1, -(2**53 + 1)], np.int64),
                jnp.asarray([2**53, -(2**53)], np.float64),
            ),
        }
        _assert_native_forwarding(_target(name), pairs[case])


@pytest.mark.parametrize("name", _COMPARISONS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_native_nan_class_and_signed_zero_comparison_controls(
    name: str, dtype: str
) -> None:
    with jax.enable_x64(dtype == "float64"):
        values = (
            jnp.asarray(
                [np.nan, np.nan, np.inf, -np.inf, -0.0, 0.0, -0.0, 0.0], dtype=dtype
            ),
            jnp.asarray([np.nan, 1, np.inf, np.inf, 0.0, -0.0, -0.0, 0.0], dtype=dtype),
        )

        def model(left: Any, right: Any) -> Any:
            return getattr(jnp, name)(left, right), left, right

        _assert_native_forwarding(model, values)


@pytest.mark.parametrize("name", _LOGICAL)
def test_unwrapped_logical_functions_keep_actual_native_trace(name: str) -> None:
    values = (
        jnp.asarray([0.0, -0.0, np.nan, np.inf, -2], np.float32),
        jnp.asarray([False, True, False, True, True]),
    )
    if name == "logical_not":
        values = values[:1]
    _assert_native_forwarding(_target(name), values)


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint64"])
def test_invert_alias_uses_each_actual_native_trace(
    dtype: str, record_property: Any
) -> None:
    with jax.enable_x64(True):
        values = _values(dtype)[:1]
        first = _assert_native_forwarding(_target("bitwise_not"), values)
        second = _assert_native_forwarding(_target("invert"), values)
        # Alias identity and trace equality are measurements, not prerequisites.
        record_property(
            "native-alias-identity", _ORIGINALS["invert"] is _ORIGINALS["bitwise_not"]
        )
        record_property("native-alias-traces-equal", first == second)


@pytest.mark.parametrize("name", _OWNED)
@pytest.mark.parametrize("reverse", [False, True])
def test_native_legacy_source_cache_orders_without_clearing(
    name: str, reverse: bool
) -> None:
    fn = _target(name)
    values = _values("int32")[:1] if name in _BITWISE_UNARY else _values("int32")
    reference, eager, compiled = _native_reference(fn, values)
    sequence = ("native", "legacy", "source", "native")
    for phase in reversed(sequence) if reverse else sequence:
        if phase == "native":
            actual, actual_eager, actual_compiled = _native_reference(fn, values)
            assert actual == reference
            _assert_payloads(actual_eager, eager)
            _assert_payloads(actual_compiled, compiled)
        elif phase == "source":
            with _source():
                assert (
                    typed_trace_manifest(jax.make_jaxpr(fn)(*map(_spec, values)))
                    == reference
                )
                _assert_payloads(_payloads(jax.jit(fn)(*values)), compiled)
        else:
            with (
                precision_policy_scope(PrecisionPolicy.LEGACY),
                _activate_plugin_worlds(),
            ):
                legacy = typed_trace_manifest(jax.make_jaxpr(fn)(*map(_spec, values)))
                assert any(
                    eqn["primitive"] == f"jax.numpy.{name}"
                    for eqn in legacy["equations"]
                )
                # These existing exporter-only primitives do not advertise an
                # MLIR/JIT lowering. Their eager implementation and cached trace
                # remain anchors between the two native JIT phases.
                _assert_payloads(_payloads(fn(*values)), eager)


@pytest.mark.parametrize("name", _OWNED)
@pytest.mark.parametrize(
    "case", ["arity", "keyword", "shape", "dtype_or_promotion", "rank"]
)
def test_unchanged_native_errors_and_context_restoration(name: str, case: str) -> None:
    unary = name in _BITWISE_UNARY
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion(
            "strict" if case == "dtype_or_promotion" else "standard"
        ),
        jax.numpy_rank_promotion("raise" if case == "rank" else "allow"),
    ):
        if case == "shape":
            values = (jnp.ones((2, 3), np.int32), jnp.ones((4, 3), np.int32))
        elif case == "rank":
            values = (jnp.ones((3,), np.int32), jnp.ones((2, 3), np.int32))
        elif case == "dtype_or_promotion":
            values = (jnp.ones((2, 3), np.float32), jnp.ones((2, 3), np.int32))
        else:
            values = _values("int32")
        if unary:
            values = values[:1]
            if case in {"shape", "rank"}:
                # Unary functions have no binary broadcast/rank error; preserve
                # the native successful rank/empty behavior instead.
                _assert_native_forwarding(_target(name), values)
                return
        if case == "arity":
            values = ()

        def model(*args: Any) -> Any:
            return getattr(jnp, name)(
                *args, **({"unknown_option": True} if case == "keyword" else {})
            )

        _assert_originals()
        with pytest.raises(Exception) as reference:
            jax.make_jaxpr(model)(*map(_spec, values))
        with _source():
            with pytest.raises(type(reference.value)) as actual:
                jax.make_jaxpr(model)(*map(_spec, values))
        assert str(actual.value).splitlines()[0] == str(reference.value).splitlines()[0]


@pytest.mark.parametrize("name", _COMPARISONS)
def test_actual_complex_comparison_outcome_is_preserved(
    name: str, record_property: Any
) -> None:
    values = (jnp.asarray([1 + 2j], np.complex64), jnp.asarray([1 - 2j], np.complex64))
    # This is binding transparency evidence only; it does not admit complex
    # comparison into the real/discrete product family.
    fn = _target(name)
    try:
        jax.make_jaxpr(fn)(*map(_spec, values))
    except Exception as reference:
        with _source():
            with pytest.raises(type(reference)) as actual:
                jax.make_jaxpr(fn)(*map(_spec, values))
        assert str(actual.value).splitlines()[0] == str(reference).splitlines()[0]
        record_property("native-outcome", type(reference).__name__)
    else:
        _assert_native_forwarding(fn, values)
        record_property("native-outcome", "typed trace and exact native outputs")


def test_nested_caller_settings_success_and_failure_restore_hidden_globals() -> None:
    before_globals, before_effective = _globals(), _settings()

    def locally_permitted(left: Any, right: Any) -> Any:
        with jax.numpy_dtype_promotion("standard"), jax.numpy_rank_promotion("allow"):
            return jnp.less(left, right), jnp.bitwise_xor(jnp.less(left, right), True)

    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        values = (jnp.asarray([1, 2], np.int16), jnp.asarray([[1, 3]], np.float32))
        _assert_native_forwarding(locally_permitted, values)
        caller_globals, caller_effective = _globals(), _settings()
        with pytest.raises(RuntimeError, match="after native forwarding"):
            with _source():
                locally_permitted(*values)
                raise RuntimeError("after native forwarding")
        assert _globals() == caller_globals and _settings() == caller_effective
    assert _globals() == before_globals and _settings() == before_effective


@pytest.mark.parametrize("name", _COMPARISONS)
@pytest.mark.parametrize("dtype", ("bool", *_INTEGER_DTYPES, *_FLOAT_DTYPES))
def test_product_comparison_family_has_exact_bits_and_canonical_siblings(
    name: str, dtype: str
) -> None:
    with jax.enable_x64(True):
        values = _values(dtype)

        def model(left: Any, right: Any) -> Any:
            return getattr(jnp, name)(left, right), left, right, left

        feeds = [values]
        _, proto, _, references = _artifact(
            model, list(map(_spec, values)), feeds=feeds
        )
        if dtype != "bfloat16":
            _accept(proto, feeds, references, _REFERENCE_CRITERIA["predicate_bool"])


@pytest.mark.parametrize("name", (*_BITWISE_BINARY, *_BITWISE_UNARY))
@pytest.mark.parametrize("dtype", ("bool", *_INTEGER_DTYPES))
def test_product_bitwise_family_preserves_every_integer_bit(
    name: str, dtype: str
) -> None:
    with jax.enable_x64(True):
        values = _values(dtype)
        if name in _BITWISE_UNARY:
            values = values[:1]

        def model(*args: Any) -> Any:
            return getattr(jnp, name)(*args), *args, args[0]

        feeds = [values]
        _, proto, _, references = _artifact(
            model, list(map(_spec, values)), feeds=feeds
        )
        _accept(proto, feeds, references, _REFERENCE_CRITERIA["bitwise_integer"])


def test_product_ordered_boolean_truth_tables_use_exact_logical_encoding() -> None:
    values = _values("bool")

    def model(left: Any, right: Any) -> Any:
        return (
            jax.lax.gt(left, right),
            jax.lax.ge(left, right),
            jax.lax.lt(left, right),
            jax.lax.le(left, right),
        )

    feeds = [values]
    _, proto, _, references = _artifact(model, list(map(_spec, values)), feeds=feeds)
    assert not any(
        node.op_type in {"Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Cast"}
        for node in proto.graph.node
    )
    _accept(proto, feeds, references, _REFERENCE_CRITERIA["predicate_bool"])


@pytest.mark.parametrize("name", _COMPARISONS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_product_nan_and_signed_zero_comparisons_return_exact_booleans(
    name: str, dtype: Any
) -> None:
    with jax.enable_x64(True):
        values = (
            np.asarray([np.nan, np.nan, np.inf, -np.inf, -0.0, 0.0], dtype),
            np.asarray([np.nan, 1, np.inf, np.inf, 0.0, -0.0], dtype),
        )
        fn = _target(name)
        _, proto, _, references = _artifact(
            fn, [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values], feeds=[values]
        )
        _accept(proto, [values], references, _REFERENCE_CRITERIA["predicate_bool"])


@pytest.mark.parametrize("name", _COMPARISONS)
@pytest.mark.parametrize("weak", [False, True])
def test_product_native_promotion_and_weak_result_signatures(
    name: str, weak: bool
) -> None:
    with jax.enable_x64(True):
        left = jnp.asarray(1) if weak else jnp.asarray([16777217, -1], np.int32)
        right = jnp.asarray(2) if weak else jnp.asarray([16777216, -1], np.float32)
        values = (left, right)
        _, proto, report, references = _artifact(
            _target(name), list(map(_spec, values)), feeds=[values]
        )
        native = typed_trace_manifest(
            jax.make_jaxpr(_target(name))(*map(_spec, values))
        )
        assert report.traced_outputs[0].weak_type is native["outputs"][0]["aval"][2]
        _accept(proto, [values], references, _REFERENCE_CRITERIA["predicate_bool"])


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
@pytest.mark.parametrize("opset", [23, 26])
def test_runtime_predicate_mask_controls_integer_selection_on_one_batch_artifact(
    dtype: Any, opset: int
) -> None:
    with jax.enable_x64(True):
        shape = export.symbolic_shape("B, 3")

        def model(left: Any, right: Any) -> Any:
            mask = jnp.logical_and(jnp.greater(left, right), jnp.not_equal(left, right))
            bits = jnp.bitwise_xor(left, right)
            return jnp.where(mask, bits, left), mask, bits, left, right

        feeds = []
        for batch in (2, 5):
            limits = np.iinfo(dtype)
            extremes = [limits.min, limits.max, limits.max - 1, 0, 1, 2**24 + 1]
            if dtype in {np.int64, np.uint64}:
                extremes.extend((2**53 + 1, 2**63 - 1))
            if dtype is np.uint64:
                extremes.extend((2**63, 2**63 + 1, 2**63 + 3))
            left = np.resize(np.asarray(extremes, dtype=dtype), batch * 3).reshape(
                batch, 3
            )
            right = left[::-1].copy()
            feeds.extend(((left, right), (right, left)))
        specs = [jax.ShapeDtypeStruct(shape, dtype), jax.ShapeDtypeStruct(shape, dtype)]
        _, proto, _, references = _artifact(model, specs, feeds=feeds, opset=opset)
        assert any(
            np.any(first[0][1] != second[0][1])
            for first, second in zip(references[::2], references[1::2], strict=True)
        )
        _accept(proto, feeds, references, _REFERENCE_CRITERIA["bitwise_integer"])


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_integer_selection_carriers_preserve_all_bits_and_sibling_consumers(
    dtype: str, shape: tuple[int, ...]
) -> None:
    with jax.enable_x64(True):
        limits = np.iinfo(dtype)
        pattern = np.asarray(
            [limits.min, limits.max, limits.max - 1, 0, 1, 2], dtype=dtype
        )
        left = np.resize(pattern, int(np.prod(shape))).reshape(shape)
        right = np.resize(pattern[::-1], int(np.prod(shape))).reshape(shape)
        masks = [np.full(shape, flag) for flag in (False, True)]

        def model(mask: Any, x: Any, y: Any) -> Any:
            selected = jnp.where(mask, x, y)
            return selected, x, y, jnp.bitwise_xor(x, y), selected, x

        feeds = [(mask, left, right) for mask in masks]
        specs = [
            jax.ShapeDtypeStruct(shape, bool),
            *(jax.ShapeDtypeStruct(shape, dtype) for _ in range(2)),
        ]
        _, proto, _, references = _artifact(model, specs, feeds=feeds)
        _accept(proto, feeds, references, _REFERENCE_CRITERIA["bitwise_integer"])


_COMPOSITION_CAPTURE = np.asarray(
    [[-(2**53 + 3), 2**53 + 1, np.iinfo(np.int64).max]], np.int64
)


@pytest.mark.parametrize("captured_value", [0, 2**63 - 1, 2**63, 2**64 - 1])
@pytest.mark.parametrize("opset", [23, 26])
def test_uint64_selection_constants_keep_high_bits_in_both_arms_and_siblings(
    captured_value: int, opset: int
) -> None:
    with jax.enable_x64(True):
        shape = export.symbolic_shape("B, 3")
        captured = np.asarray(captured_value, np.uint64)
        before = captured.tobytes()

        def model(mask: Any, x: Any) -> Any:
            broadcast = jnp.broadcast_to(captured, x.shape)
            left = jnp.where(mask, x, broadcast)
            right = jnp.where(mask, broadcast, x)
            bits = jnp.bitwise_xor(x, broadcast)
            return left, right, x, captured, bits, captured, x

        feeds = []
        for batch in (2, 5):
            values = np.resize(
                np.asarray([0, 2**63 - 1, 2**63, 2**64 - 1], np.uint64), batch * 3
            ).reshape(batch, 3)
            feeds.extend((np.full((batch, 3), flag), values) for flag in (False, True))
        _, proto, _, references = _artifact(
            model,
            [jax.ShapeDtypeStruct(shape, bool), jax.ShapeDtypeStruct(shape, np.uint64)],
            feeds=feeds,
            opset=opset,
        )
        assert captured.tobytes() == before
        _accept(proto, feeds, references, _REFERENCE_CRITERIA["bitwise_integer"])
        assert captured.tobytes() == before


@onnx_function(name="TypedMaskImplicitScalarInner")
def _composition_implicit_scalar_inner(
    x: Any, weights: Any, threshold: Any, data: Any, flag: Any
) -> Any:
    def floating(a: Any, b: Any) -> Any:
        return jnp.multiply(jnp.exp(jnp.matmul(a, b)), np.asarray(0.5, dtype=a.dtype))

    value = jax.jit(floating)(x, weights)
    mask = jnp.logical_xor(jnp.greater(value, threshold), flag)
    bits = jnp.bitwise_xor(data, _COMPOSITION_CAPTURE)
    selected = jax.lax.cond(
        flag,
        lambda: jnp.where(mask, bits, data),
        lambda: jnp.where(mask, bits, _COMPOSITION_CAPTURE),
    )
    return value, mask, selected, data, bits, _COMPOSITION_CAPTURE


@onnx_function(name="TypedMaskImplicitScalarOuter")
def _composition_implicit_scalar_outer(
    x: Any, weights: Any, threshold: Any, data: Any, flag: Any
) -> Any:
    return _composition_implicit_scalar_inner(x, weights, threshold, data, flag)


def test_implicit_scalar_batch_composition_retains_its_temporary_shape_gap() -> None:
    from jax2onnx.converter.source_profile import SourcePrecisionError

    with jax.enable_x64(True):
        shape = export.symbolic_shape("B, 3")
        specs = [
            jax.ShapeDtypeStruct(shape, np.float32),
            jax.ShapeDtypeStruct((3, 3), np.float32),
            jax.ShapeDtypeStruct((), np.float32),
            jax.ShapeDtypeStruct(shape, np.int64),
            jax.ShapeDtypeStruct((), bool),
        ]
        feeds = [
            (
                np.full((batch, 3), 0.25, np.float32),
                np.eye(3, dtype=np.float32),
                np.asarray(0.5, np.float32),
                np.full((batch, 3), np.iinfo(np.int64).max - 1, np.int64),
                np.asarray(flag),
            )
            for batch in (2, 5)
            for flag in (False, True)
        ]
        with pytest.raises(SourcePrecisionError) as error:
            _artifact(
                _composition_implicit_scalar_outer, specs, feeds=feeds, named=True
            )
        assert error.value.phase == "profile-validation"
        assert error.value.category == "profile-exclusion"
        assert error.value.primitive == "mul"
        assert (
            error.value.detail
            == "pointwise operands do not share the same leading-batch shape"
        )
        assert (
            error.value.requirement
            == "equal rank-two/rank-three tails with the same native B or a literal leading extent 1"
        )


@onnx_function(name="TypedMaskFloatingInner")
def _composition_inner(
    x: Any, weights: Any, threshold: Any, data: Any, flag: Any
) -> Any:
    def floating(a: Any, b: Any) -> Any:
        return jnp.multiply(
            jnp.exp(jnp.matmul(a, b)), np.full((1, 3), 0.5, dtype=a.dtype)
        )

    value = jax.jit(floating)(x, weights)
    mask = jnp.logical_xor(
        jnp.greater(value, threshold), jnp.broadcast_to(flag, value.shape)
    )
    bits = jnp.bitwise_xor(data, _COMPOSITION_CAPTURE)
    selected = jax.lax.cond(
        flag,
        lambda: jnp.where(mask, bits, data),
        lambda: jnp.where(mask, bits, _COMPOSITION_CAPTURE),
    )
    return value, mask, selected, data, bits, _COMPOSITION_CAPTURE


@onnx_function(name="TypedMaskFloatingOuter")
def _composition_outer(
    x: Any, weights: Any, threshold: Any, data: Any, flag: Any
) -> Any:
    return _composition_inner(x, weights, threshold, data, flag)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
@pytest.mark.parametrize("entrypoint", ["decorated", "caller"])
def test_named_jit_matrix_exp_mask_integer_capture_and_cond_are_one_live_composition(
    dtype: Any, mode: str, entrypoint: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        shape = export.symbolic_shape("B, 3")
        specs = [
            jax.ShapeDtypeStruct(shape, dtype),
            jax.ShapeDtypeStruct((3, 3), dtype),
            jax.ShapeDtypeStruct((1, 3), dtype),
            jax.ShapeDtypeStruct(shape, np.int64),
            jax.ShapeDtypeStruct((), bool),
        ]
        feeds = []
        for batch in (2, 5):
            x = np.resize(np.asarray([-0.25, 0.25, 0.125], dtype), (batch, 3))
            data = np.full((batch, 3), np.iinfo(np.int64).max - 1, dtype=np.int64)
            for flag in (False, True):
                feeds.append(
                    (
                        x,
                        np.eye(3, dtype=dtype),
                        np.full((1, 3), 0.5, dtype),
                        data,
                        np.asarray(flag),
                    )
                )

        def model(*args: Any) -> Any:
            return _composition_outer(*args)

        _, proto, report, references = _artifact(
            _composition_outer if entrypoint == "decorated" else model,
            specs,
            feeds=feeds,
            named=True,
            mode=mode,
            path=tmp_path / "typed-composition.onnx" if mode == "file" else None,
        )
        assert len(proto.functions) >= (1 if entrypoint == "decorated" else 2)
        assert [value.dtype for value in report.traced_outputs] == [
            np.dtype(dtype).name,
            "bool",
            "int64",
            "int64",
            "int64",
            "int64",
        ]
        # The existing sample API always compares integer/BOOL outputs exactly,
        # even when the separately typed floating output uses the frozen finite
        # Exp criterion. Zero-sign and payload identity have strict tests above.
        _accept(proto, feeds, references, _FINITE[np.dtype(dtype).name])


@pytest.mark.parametrize(
    "mutation", ["dtype", "weakness", "shape", "arity", "params", "effects"]
)
def test_predicate_signature_corruption_is_rejected_before_emission(
    mutation: str,
) -> None:
    from tests.extra_tests.converter.test_source_profile import (
        _graph,
        _spec as profile_spec,
    )

    inputs = [profile_spec((2,), np.int64), profile_spec((2,), np.int64)]
    output = profile_spec((2,), bool)
    if mutation == "dtype":
        inputs[1] = profile_spec((2,), np.uint64)
    elif mutation == "weakness":
        output = jax.ShapeDtypeStruct((2,), bool, weak_type=True)
    elif mutation == "shape":
        output = profile_spec((3,), bool)
    graph = _graph(
        "gt", inputs, [output], {"out_sharding": None} if mutation == "params" else {}
    )
    if mutation == "arity":
        graph.eqns[0].invars.pop()
    elif mutation == "effects":
        graph.eqns[0].effects = (object(),)
        graph.effects = graph.eqns[0].effects
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(graph, 23)
    assert error.value.category == (
        "profile-exclusion" if mutation in {"params", "effects"} else "lowering-defect"
    )
