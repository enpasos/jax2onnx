# tests/extra_tests/converter/test_source_callable_preparation.py

"""Explicit pretrace choices preserve actual source, owners and ordinary JAX calls."""

import ast
from collections.abc import Iterator
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
import linecache
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx._source_preparation import SourcePreparationError
from jax2onnx._source_callable_preparation import (
    BindingRef,
    CallableRecipe,
    DtypeSite,
    HelperBinding,
    SharedBinding,
    callable_preimage,
    prepare_callable,
)
from jax2onnx.converter.precision_policy import active_precision_policy, PrecisionPolicy
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _settings() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


@pytest.fixture(autouse=True)
def _preserve_settings() -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    bindings = jnp.asarray, jnp.astype, jax.jit
    try:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)
        assert (jnp.asarray, jnp.astype, jax.jit) == bindings
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _same(actual: Any, expected: Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape)
    assert actual.tobytes() == expected.tobytes()


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _node(fn: Any) -> Any:
    return ast.parse(callable_preimage(fn).source).body[0]


def _paths(node: Any, expression: str, path: tuple[Any, ...] = ()) -> list[Any]:
    found = (
        [path]
        if isinstance(node, (ast.Name, ast.Attribute))
        and ast.unparse(node) == expression
        else []
    )
    if isinstance(node, ast.AST):
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                found.extend(_paths(value, expression, (*path, name)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        found.extend(_paths(child, expression, (*path, name, index)))
    return [p for p in found if p and p[0] == "body"]


def _at(node: Any, path: Any) -> Any:
    for item in path:
        node = node[item] if isinstance(item, int) else getattr(node, item)
    return node


def _recipe(
    fn: Any,
    *,
    replacement: Any = jnp.float64,
    expression: str = "jnp.float32",
    indices: tuple[int, ...] | None = None,
    root: BindingRef | None = None,
    expected: Any = jnp.float32,
    helpers: tuple[Any, ...] = (),
    shared: tuple[Any, ...] = (),
    intent: str = "Intentionally prepare only the explicitly selected native dtype choices",
) -> Any:
    preimage = callable_preimage(fn)
    node = ast.parse(preimage.source).body[0]
    paths = _paths(node, expression)
    selected = paths if indices is None else [paths[i] for i in indices]
    reference = root if root is not None else BindingRef("global", "jnp", jnp)
    sites = tuple(
        DtypeSite(
            path,
            hashlib.sha256(
                ast.dump(_at(node, path), include_attributes=False).encode()
            ).hexdigest(),
            reference,
            expected,
            replacement,
        )
        for path in selected
    )
    return CallableRecipe(
        fn,
        preimage.source_sha256,
        preimage.code_sha256,
        sites,
        len(sites),
        intent,
        helpers=helpers,
        shared=shared,
    )


def _prepare_rounding() -> tuple[Any, Any]:
    return prepare_callable(_rounding, _recipe(_rounding))


def _rounding(x: Any) -> Any:
    y = x.astype(jnp.float32)
    return (y + 1) - y


def _two_casts(x: Any) -> Any:
    selected = x.astype(jnp.float32)
    sibling = x.astype(jnp.float32)
    return (selected + 1) - selected, sibling


_ROUNDING = _rounding
_ORIGINAL_SIBLING = _rounding
_DTYPE32 = np.float32


def _name_dtype(x: Any) -> Any:
    y = x.astype(_DTYPE32)
    return (y + 1) - y


def _dtype_branch(x: Any) -> Any:
    y = x.astype(jnp.float32)
    if y.dtype == np.dtype(np.float64):
        return y * 2
    return y + 3


def _global_root(x: Any) -> Any:
    return _ROUNDING(x), _ORIGINAL_SIBLING(x)


def _protected_root(x: Any) -> Any:
    selected = _ROUNDING(x)
    rounded = x.astype(jnp.float32)
    return (
        selected,
        selected.astype(jnp.float32),
        rounded,
        _ORIGINAL_SIBLING(x),
        jax.lax.bitcast_convert_type(rounded, jnp.uint32),
        jnp.sum(rounded, dtype=jnp.float32),
    )


def _selected_branch(x: Any) -> Any:
    return _ROUNDING(x).astype(x.dtype)


def _original_branch(x: Any) -> Any:
    return _ORIGINAL_SIBLING(x).astype(x.dtype)


def _conditional_root(x: Any, selector: Any) -> Any:
    return jax.lax.cond(selector, _selected_branch, _original_branch, x)


def _root_recipe(
    fn: Any,
    name: str,
    original: Any,
    prepared: Any,
    record: Any,
    *,
    scope: str = "global",
    shared: tuple[Any, ...] = (),
) -> Any:
    return _recipe(
        fn,
        indices=(),
        helpers=(HelperBinding(BindingRef(scope, name, original), prepared, record),),
        shared=shared,
    )


def _prepared_global(fn: Any = _global_root) -> tuple[Any, Any, Any, Any]:
    helper, dependency = _prepare_rounding()
    prepared, record = prepare_callable(
        fn, _root_recipe(fn, "_ROUNDING", _rounding, helper, dependency)
    )
    return prepared, record, helper, dependency


def _closure_root(helper: Any, payload: Any) -> Any:
    sibling = helper

    def root(x: Any) -> Any:
        return helper(x), sibling(x), payload, payload

    return root


def _real_owners() -> tuple[Any, Any]:
    from jax2onnx.plugins.examples.eqx.gpt_oss import _apply_linear_nd
    from jax2onnx.plugins.examples.nnx.gpt_oss_flax import _softmax_torch_approx

    return _apply_linear_nd, _softmax_torch_approx


def _owner_snapshot(fn: Any) -> tuple[Any, ...]:
    path = Path(inspect.getsourcefile(fn))
    return (
        fn.__code__,
        fn.__globals__,
        fn.__defaults__,
        fn.__kwdefaults__,
        fn.__annotations__,
        tuple(sorted((name, id(value)) for name, value in fn.__dict__.items())),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _linear_fixture() -> Any:
    model = eqx.nn.Linear(3, 2, key=jax.random.key(19), dtype=jnp.float32)
    return eqx.tree_at(
        lambda m: (m.weight, m.bias),
        model,
        (
            jnp.asarray([[1, 0, 0], [0, 2, 0]], jnp.float64),
            jnp.asarray([0.5, -1], jnp.float64),
        ),
    )


def _linear_values() -> np.ndarray:
    return np.asarray([[2**26 + 1, 2**25 + 1, 0.5], [1 + 2**-35, -2.5, 3]], np.float64)


def _linear_expected(value: Any) -> np.ndarray:
    # Independent coordinate oracle: only exact power-of-two products and sums.
    value = np.asarray(value)
    return np.stack(
        (
            value[..., 0] + np.float64(0.5),
            value[..., 1] * np.float64(2) - np.float64(1),
        ),
        axis=-1,
    )


def _owned_root(helper: Any, owner: Any = None) -> Any:
    if owner is None:

        def softmax_root(x: Any) -> Any:
            return helper(x, axis=-1)

        return softmax_root

    def linear_root(x: Any) -> Any:
        return helper(owner, x)

    return linear_root


def _body(closed: Any) -> Any:
    body = getattr(closed, "jaxpr", closed)
    return body if body is not closed else closed


def _equations(closed: Any) -> list[Any]:
    equations = []
    for eqn in _body(closed).eqns:
        equations.append(eqn)
        for value in eqn.params.values():
            child = getattr(value, "jaxpr", None)
            if child is not None and child is not value:
                equations.extend(_equations(value))
            elif hasattr(value, "eqns"):
                equations.extend(_equations(value))
            elif isinstance(value, (tuple, list)):
                for part in value:
                    if hasattr(part, "jaxpr") or hasattr(part, "eqns"):
                        equations.extend(_equations(part))
    return equations


def test_selected_native_rounding_and_record_are_independent_of_conversion(
    tmp_path: Path,
) -> None:
    before = _owner_snapshot(_rounding)
    prepared, record = _prepare_rounding()
    assert type(prepared) is FunctionType and prepared is not _rounding
    values = jnp.asarray([1e8, 1e8 + 1, -1e8, 0, 16777216], np.float64)
    old, wanted = np.asarray([0, 0, 0, 1, 0], np.float32), np.ones(5, np.float64)
    for call in (prepared, jax.jit(_fresh(prepared))):
        _same(call(values), wanted)
    for call in (_rounding, jax.jit(_fresh(_rounding))):
        _same(call(values), old)
    data = record.to_dict()
    assert (
        data["schema_version"] == 1
        and data["kind"] == "selected-callable-dtype-preparation-v1"
    )
    assert data["original"]["id"] == id(_rounding) and data["prepared"]["id"] == id(
        prepared
    )
    assert data["acceptance"] == "UNVERIFIED_PREPARATION_ONLY"
    assert (
        data["sites"][0]["old_dtype"] == "float32"
        and data["sites"][0]["new_dtype"] == "float64"
    )
    assert record.sha256 == hashlib.sha256(record.to_json().encode()).hexdigest()
    assert json.loads(record.to_json()) == data
    data["sites"].clear()
    assert len(record.to_dict()["sites"]) == 1
    record.assert_current()
    (tmp_path / "preparation.json").write_text(record.to_json())
    assert _owner_snapshot(_rounding) == before


def test_selected_site_does_not_widen_protected_rounding_bitcast_castback_or_sum() -> (
    None
):
    prepared, record, _, _ = _prepared_global(_protected_root)
    values = jnp.asarray([1e8, 1e8 + 1, -1e8, 0, 16777216], np.float64)
    rounded = values.astype(np.float32)
    old = _protected_root(values)
    for call in (prepared, jax.jit(_fresh(prepared))):
        actual = call(values)
        _same(actual[0], np.ones(5, np.float64))
        _same(actual[1], np.ones(5, np.float32))
        _same(actual[2], rounded)
        for left, right in zip(actual[2:], old[2:], strict=True):
            _same(left, right)
        _same(actual[4], rounded.view(np.uint32))
        _same(actual[5], np.asarray(116777216, np.float32))
    assert _protected_root.__globals__["_ROUNDING"] is _rounding is _ORIGINAL_SIBLING
    record.assert_current()


def test_name_binding_and_dtype_dependent_branch_are_prepared_before_tracing() -> None:
    named, _ = prepare_callable(
        _name_dtype,
        _recipe(
            _name_dtype,
            expression="_DTYPE32",
            root=BindingRef("global", "_DTYPE32", np.float32),
            expected=np.float32,
            replacement=np.float64,
        ),
    )
    branch, _ = prepare_callable(_dtype_branch, _recipe(_dtype_branch))
    values = jnp.asarray([2**26 + 1, -0.25], np.float64)
    for call in (named, jax.jit(_fresh(named))):
        _same(call(values), np.ones(2, np.float64))
    for call in (branch, jax.jit(_fresh(branch))):
        _same(call(values), values * np.float64(2))
    assert not np.array_equal(
        np.asarray(_dtype_branch(values)).astype(np.float64), values * 2
    )


def test_two_prepared_versions_original_and_unselected_cast_interleave_under_jit() -> (
    None
):
    first, first_record = prepare_callable(
        _two_casts, _recipe(_two_casts, indices=(0,))
    )
    second, second_record = prepare_callable(
        _two_casts, _recipe(_two_casts, indices=(1,))
    )
    assert len({id(first), id(second), id(_two_casts)}) == 3
    values = jnp.asarray([1e8 + 1, 16777216], np.float64)
    rounded = values.astype(np.float32)
    compiled = {fn: jax.jit(_fresh(fn)) for fn in (_two_casts, first, second)}
    expected = {
        _two_casts: (np.zeros(2, np.float32), rounded),
        first: (np.ones(2, np.float64), rounded),
        second: (np.zeros(2, np.float32), values),
    }
    for fn in (first, _two_casts, second, first, second, _two_casts):
        for call in (fn, compiled[fn]):
            for actual, wanted in zip(call(values), expected[fn], strict=True):
                _same(actual, wanted)
    first_record.assert_current()
    second_record.assert_current()


def test_wrong_sibling_selection_is_exposed_by_the_independent_intended_program() -> (
    None
):
    correct, _ = prepare_callable(_two_casts, _recipe(_two_casts, indices=(0,)))
    wrong, _ = prepare_callable(_two_casts, _recipe(_two_casts, indices=(1,)))
    values = jnp.asarray([1e8 + 1], np.float64)
    _same(correct(values)[0], np.ones(1, np.float64))
    assert np.asarray(wrong(values)[0]).dtype == np.float32
    assert np.asarray(wrong(values)[0]).tobytes() != np.ones(1, np.float64).tobytes()


def test_global_and_closure_helper_bindings_preserve_original_siblings_and_aliases() -> (
    None
):
    global_root, global_record, helper, helper_record = _prepared_global()
    payload = jnp.asarray([0.5, 1.0], jnp.float64)
    original = _closure_root(_rounding, payload)
    closure, closure_record = prepare_callable(
        original,
        _root_recipe(
            original, "helper", _rounding, helper, helper_record, scope="closure"
        ),
    )
    values = jnp.asarray([1e8 + 1], np.float64)
    for root in (global_root, closure):
        for call in (root, jax.jit(_fresh(root))):
            result = call(values)
            _same(result[0], np.ones(1, np.float64))
            _same(result[1], np.zeros(1, np.float32))
    result = closure(values)
    assert result[2] is result[3] is payload
    assert (
        dict(zip(closure.__code__.co_freevars, closure.__closure__, strict=True))[
            "payload"
        ].cell_contents
        is payload
    )
    assert _global_root.__globals__["_ROUNDING"] is _rounding
    global_record.assert_current()
    closure_record.assert_current()


def test_prepared_global_namespace_is_a_frozen_name_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, record, helper, _ = _prepared_global()
    values = jnp.asarray([1e8 + 1], np.float64)
    with monkeypatch.context() as patch:
        patch.setitem(_global_root.__globals__, "_ROUNDING", _original_branch)
        assert prepared.__globals__["_ROUNDING"] is helper
        _same(prepared(values)[0], np.ones(1, np.float64))
        record.assert_current()
    assert "frozen" in record.to_dict()["namespace_semantics"]


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "code",
        "ast",
        "path",
        "binding",
        "dtype",
        "required",
        "intent",
        "wrong-original",
        "duplicate",
        "call-node",
    ],
)
def test_explicit_preimage_site_binding_and_required_count_controls_fail_closed(
    fault: str,
) -> None:
    recipe = _recipe(_rounding)
    baseline, record = prepare_callable(_rounding, recipe)
    _same(baseline(jnp.asarray([1e8 + 1], jnp.float64)), np.ones(1, np.float64))
    record.assert_current()
    site = recipe.sites[0]
    if fault == "source":
        recipe = replace(recipe, source_sha256="0" * 64)
    elif fault == "code":
        recipe = replace(recipe, code_sha256="0" * 64)
    elif fault == "ast":
        recipe = replace(recipe, sites=(replace(site, ast_sha256="0" * 64),))
    elif fault == "path":
        recipe = replace(recipe, sites=(replace(site, path=("body", 999)),))
    elif fault == "binding":
        recipe = replace(
            recipe, sites=(replace(site, root=BindingRef("global", "jnp", np)),)
        )
    elif fault == "dtype":
        recipe = replace(recipe, sites=(replace(site, expected_dtype=jnp.float16),))
    elif fault == "required":
        recipe = replace(recipe, sites=())
    elif fault == "intent":
        recipe = replace(recipe, intent="")
    elif fault == "duplicate":
        recipe = replace(recipe, sites=(site, site), required_site_count=2)
    elif fault == "call-node":
        path = site.path[:-2]
        fingerprint = hashlib.sha256(
            ast.dump(_at(_node(_rounding), path), include_attributes=False).encode()
        ).hexdigest()
        recipe = replace(
            recipe, sites=(replace(site, path=path, ast_sha256=fingerprint),)
        )
    else:
        recipe = replace(recipe, original=_two_casts)
    with pytest.raises(SourcePreparationError):
        prepare_callable(_rounding, recipe)
    record.assert_current()


def test_shadowed_original_global_dtype_module_is_not_authorized_by_source_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe(_rounding)
    prepared, _ = prepare_callable(_rounding, recipe)
    foreign = ModuleType("foreign_dtype_module")
    foreign.float32 = np.float64
    with monkeypatch.context() as patch:
        patch.setitem(_rounding.__globals__, "jnp", foreign)
        with pytest.raises(SourcePreparationError, match="binding.*identity"):
            prepare_callable(_rounding, recipe)
    _same(prepared(jnp.asarray([1e8 + 1], jnp.float64)), np.ones(1, np.float64))


@pytest.mark.parametrize("fault", ["private-dtype", "helper-code"])
def test_transitive_prepared_helper_records_reject_later_dependency_corruption(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, record, helper, dependency = _prepared_global(_selected_branch)
    outer_recipe = _root_recipe(
        _conditional_root, "_selected_branch", _selected_branch, prepared, record
    )
    _, outer_record = prepare_callable(_conditional_root, outer_recipe)
    record.assert_current()
    outer_record.assert_current()
    if fault == "private-dtype":
        key = dependency.to_dict()["sites"][0]["private_name"]
        monkeypatch.setitem(helper.__globals__, key, jnp.float32)
    else:
        monkeypatch.setattr(helper, "__code__", _rounding.__code__)
    with pytest.raises(SourcePreparationError):
        dependency.assert_current()
    with pytest.raises(SourcePreparationError):
        record.assert_current()
    with pytest.raises(SourcePreparationError):
        outer_record.assert_current()
    with pytest.raises(SourcePreparationError):
        prepare_callable(_conditional_root, outer_recipe)
    assert _global_root.__globals__["_ROUNDING"] is _rounding
    assert prepared.__globals__["_ROUNDING"] is helper


def test_helper_identity_and_accidental_original_binding_are_rejected() -> None:
    prepared, dependency = _prepare_rounding()
    root, record = prepare_callable(
        _global_root,
        _root_recipe(_global_root, "_ROUNDING", _rounding, prepared, dependency),
    )
    _same(root(jnp.asarray([1e8 + 1], jnp.float64))[0], np.ones(1, np.float64))
    bad = _root_recipe(_global_root, "_ROUNDING", _rounding, _rounding, dependency)
    with pytest.raises(SourcePreparationError, match="helper.*identity|record"):
        prepare_callable(_global_root, bad)
    record.assert_current()


def test_default_annotation_and_identity_decorator_factories_are_not_reexecuted() -> (
    None
):
    calls = []

    def default():
        calls.append("default")
        return []

    def keyword():
        calls.append("keyword")
        return {}

    def annotation():
        calls.append("annotation")
        return object

    def decorate(fn):
        calls.append("decorator")
        return fn

    @decorate
    def target(x: annotation(), value=default(), *, flag=keyword()) -> annotation():
        return x.astype(jnp.float32)

    before = tuple(calls)
    shared = (
        SharedBinding("default", 0, target.__defaults__[0]),
        SharedBinding("kwdefault", "flag", target.__kwdefaults__["flag"]),
    )
    with pytest.raises(SourcePreparationError, match="shared ownership"):
        prepare_callable(target, _recipe(target))
    prepared, record = prepare_callable(target, _recipe(target, shared=shared))
    assert tuple(calls) == before
    assert prepared.__defaults__ is target.__defaults__
    assert prepared.__kwdefaults__ is target.__kwdefaults__
    assert prepared.__annotations__ is target.__annotations__
    assert inspect.signature(prepared) == inspect.signature(target)
    assert all(
        entry["ownership"] == "shared-reference" for entry in record.to_dict()["shared"]
    )
    record.assert_current()


def test_numpy_capture_requires_explicit_shared_ownership_without_copying_or_mutation() -> (
    None
):
    capture = np.asarray([0.5], np.float64)

    def target(x: Any) -> Any:
        return x.astype(jnp.float32) + capture[0]

    before = capture.copy()
    with pytest.raises(SourcePreparationError, match="shared ownership"):
        prepare_callable(target, _recipe(target))
    prepared, record = prepare_callable(
        target, _recipe(target, shared=(SharedBinding("closure", "capture", capture),))
    )
    cells = dict(zip(prepared.__code__.co_freevars, prepared.__closure__, strict=True))
    assert cells["capture"].cell_contents is capture
    _same(capture, before)
    record.assert_current()


def test_stale_file_and_wrapped_source_do_not_masquerade_as_live_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "existing_source_owner.py"
    original = "import jax.numpy as jnp\ndef existing(x):\n    y = x.astype(jnp.float32)\n    return (y + 1) - y\n"
    path.write_text(original)
    spec = importlib.util.spec_from_file_location("existing_source_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = _recipe(module.existing)
    prepare_callable(module.existing, baseline)[1].assert_current()
    path.write_text(original.replace("(y + 1)", "(y + 2)"))
    linecache.clearcache()
    # Recompute both public hashes from mismatched current source/live code:
    # correspondence proof, rather than hash freshness alone, must reject it.
    with pytest.raises(SourcePreparationError, match="correspond|preimage"):
        prepare_callable(module.existing, _recipe(module.existing))
    with monkeypatch.context() as patch:
        patch.setattr(_rounding, "__wrapped__", _two_casts, raising=False)
        with pytest.raises(SourcePreparationError, match="wrapped"):
            callable_preimage(_rounding)


@pytest.mark.parametrize(
    "replacement",
    [
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
        jnp.float64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_exact_supported_dtype_class_choices_keep_normal_callable_identity(
    replacement: Any,
) -> None:
    prepared, record = prepare_callable(
        _rounding, _recipe(_rounding, replacement=replacement)
    )
    result = prepared(jnp.asarray([0.0], jnp.float64))
    _same(result, np.ones(1, dtype=np.dtype(replacement)))
    record.assert_current()


def test_fp64_prerequisite_and_exception_preserve_hidden_and_effective_settings() -> (
    None
):
    before = _globals(), _settings()
    with jax.enable_x64(False):
        local = _globals(), _settings()
        with pytest.raises(SourcePreparationError, match="already-effective JAX x64"):
            _prepare_rounding()
        assert (_globals(), _settings()) == local
    assert (_globals(), _settings()) == before


@pytest.mark.parametrize("owner", ["linear", "softmax"])
def test_actual_existing_helpers_keep_exact_selected_sites_final_casts_and_files(
    owner: str, tmp_path: Path
) -> None:
    linear, softmax = _real_owners()
    original = linear if owner == "linear" else softmax
    before = _owner_snapshot(original)
    recipe = _recipe(original)
    assert len(recipe.sites) == (3 if owner == "linear" else 1)
    prepared, record = prepare_callable(original, recipe)
    (tmp_path / "actual-owner-preparation.json").write_text(record.to_json())
    if owner == "linear":
        model, values = _linear_fixture(), jnp.asarray(_linear_values())
        frozen = tuple(np.asarray(v).copy() for v in (model.weight, model.bias))
        root = _owned_root(original, model)
        selected, root_record = prepare_callable(
            root,
            _root_recipe(
                root,
                "helper",
                original,
                prepared,
                record,
                scope="closure",
                shared=(SharedBinding("closure", "owner", model),),
            ),
        )
        for call in (selected, jax.jit(_fresh(selected))):
            _same(call(values), _linear_expected(values))
        assert not np.array_equal(np.asarray(root(values)), _linear_expected(values))
        trace = jax.make_jaxpr(_fresh(selected))(values)
        dot = next(e for e in _equations(trace) if e.primitive is jax.lax.dot_general_p)
        assert dot.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2
        assert all(
            np.dtype(v.aval.dtype) == np.float64 for v in (*dot.invars, *dot.outvars)
        )
        for value, wanted in zip((model.weight, model.bias), frozen, strict=True):
            _same(value, wanted)
        root_record.assert_current()
        # The real helper still honors its unselected final weight-dtype cast.
        low_model = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            model,
            (model.weight.astype(jnp.float32), model.bias.astype(jnp.float32)),
        )

        def low_root(x: Any) -> Any:
            return prepared(low_model, x)

        for call in (low_root, jax.jit(_fresh(low_root))):
            _same(call(values), _linear_expected(values).astype(np.float32))
        low_trace = jax.make_jaxpr(_fresh(low_root))(values)
        assert np.dtype(_body(low_trace).outvars[0].aval.dtype) == np.float32
        assert any(
            e.primitive.name == "convert_element_type"
            and np.dtype(e.params["new_dtype"]) == np.float32
            for e in _equations(low_trace)
        )
    else:
        values = jnp.zeros((2, 4), jnp.float32)
        for call in (prepared, jax.jit(_fresh(prepared))):
            _same(call(values), np.full((2, 4), 0.25, np.float32))
        trace = jax.make_jaxpr(_fresh(prepared))(values)
        exps = [e for e in _equations(trace) if e.primitive.name == "exp"]
        assert exps and all(
            np.dtype(e.outvars[0].aval.dtype) == np.float64 for e in exps
        )
        assert np.dtype(_body(trace).outvars[0].aval.dtype) == np.float32
    (tmp_path / "actual-owner-native-trace.json").write_text(
        json.dumps(typed_trace_manifest(trace), indent=2)
    )
    assert _owner_snapshot(original) == before
    record.assert_current()


def _unsupported_nested(x):
    def child(y):
        return y

    return child(x.astype(jnp.float32))


def _unsupported_lambda(x):
    return (lambda y: y)(x.astype(jnp.float32))


def _unsupported_recursive(x):
    return _unsupported_recursive(x.astype(jnp.float32))


def _unsupported_dynamic(x):
    return globals()["_rounding"](x.astype(jnp.float32))


def _unsupported_generator(x):
    yield x.astype(jnp.float32)


def _unsupported_global_write(x):
    global _UNWRITTEN_STATE
    _UNWRITTEN_STATE = x
    return x.astype(jnp.float32)


def _unsupported_subscript_write(x):
    state = [None]
    state[0] = x
    return x.astype(jnp.float32)


@pytest.mark.parametrize(
    "fn",
    [
        _unsupported_nested,
        _unsupported_lambda,
        _unsupported_recursive,
        _unsupported_dynamic,
        _unsupported_generator,
        _unsupported_global_write,
        _unsupported_subscript_write,
    ],
)
def test_unsupported_python_state_and_dispatch_forms_fail_before_target_execution(
    fn: Any,
) -> None:
    _, baseline = _prepare_rounding()
    baseline.assert_current()
    with pytest.raises(SourcePreparationError):
        prepare_callable(fn, _recipe(fn))
    assert "_UNWRITTEN_STATE" not in globals()


def test_local_shadow_and_nonclass_dtype_requests_remain_explicit_preparation_errors() -> (
    None
):
    _, baseline = _prepare_rounding()
    baseline.assert_current()

    def target(x, jnp):
        return x.astype(jnp.float32)

    with pytest.raises(SourcePreparationError, match="shadowed"):
        prepare_callable(target, _recipe(target))
    with pytest.raises(SourcePreparationError, match="actual supported native"):
        prepare_callable(
            _rounding, _recipe(_rounding, replacement=np.dtype(np.float64))
        )


def _builtin_length(x):
    count = len(x)
    return x.astype(jnp.float32) + count


def test_cached_builtin_mapping_is_preserved_and_later_private_shadowing_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setitem(
            _builtin_length.__globals__, "__builtins__", {"len": lambda value: 999}
        )
        prepared, record = prepare_callable(_builtin_length, _recipe(_builtin_length))
        assert prepared.__builtins__ is _builtin_length.__builtins__
        _same(prepared(jnp.zeros(2, jnp.float64)), np.full(2, 2, np.float64))
        record.assert_current()
        patch.setitem(prepared.__globals__, "len", lambda value: 777)
        with pytest.raises(SourcePreparationError):
            record.assert_current()


def test_empty_retained_closure_cell_is_a_preparation_error() -> None:
    helper, dependency = _prepare_rounding()
    original = _closure_root(_rounding, jnp.asarray([0.5], jnp.float64))
    recipe = _root_recipe(
        original, "helper", _rounding, helper, dependency, scope="closure"
    )
    _, record = prepare_callable(original, recipe)
    record.assert_current()
    cells = dict(zip(original.__code__.co_freevars, original.__closure__, strict=True))
    del cells["payload"].cell_contents
    with pytest.raises(SourcePreparationError):
        record.assert_current()
    with pytest.raises(SourcePreparationError):
        prepare_callable(original, recipe)


def _raising_target(x: Any) -> Any:
    y = x.astype(jnp.float32)
    if y.ndim != 1:
        raise ValueError("native target requires a vector")
    return (y + 1) - y


def test_preparation_and_native_target_exception_do_not_mutate_caller_settings() -> (
    None
):
    before = _globals(), _settings()
    prepared, record = prepare_callable(_raising_target, _recipe(_raising_target))
    for target in (_raising_target, prepared):
        with pytest.raises(ValueError, match="native target requires a vector"):
            target(jnp.asarray(0, jnp.float64))
        assert (_globals(), _settings()) == before
    record.assert_current()


def _captured_dtype(dtype: Any) -> Any:
    def target(x: Any) -> Any:
        y = x.astype(dtype)
        return (y + 1) - y

    return target


def test_actual_closure_dtype_name_and_scope_identity_are_validated() -> None:
    original = _captured_dtype(jnp.float32)
    recipe = _recipe(
        original, expression="dtype", root=BindingRef("closure", "dtype", jnp.float32)
    )
    prepared, record = prepare_callable(original, recipe)
    values = jnp.asarray([1e8 + 1], jnp.float64)
    for call in (prepared, jax.jit(_fresh(prepared))):
        _same(call(values), np.ones(1, np.float64))
    cell = dict(zip(original.__code__.co_freevars, original.__closure__, strict=True))[
        "dtype"
    ]
    assert cell.cell_contents is jnp.float32
    record.assert_current()
    cell.cell_contents = jnp.float16
    try:
        with pytest.raises(SourcePreparationError, match="binding.*identity"):
            prepare_callable(original, recipe)
    finally:
        cell.cell_contents = jnp.float32
    record.assert_current()


def _private_positional(
    x, __jax2onnx_prepared_dtype_0=None, *, __jax2onnx_prepared_dtype_0_=None
):
    return x.astype(jnp.float32)


def _private_keyword(x, *, __jax2onnx_prepared_dtype_0=None):
    return x.astype(jnp.float32)


def _private_varargs(x, *__jax2onnx_prepared_dtype_0):
    return x.astype(jnp.float32)


@pytest.mark.parametrize("kind", ["positional", "keyword", "varargs"])
def test_unused_argument_names_cannot_shadow_injected_dtype_bindings(kind: str) -> None:
    original = {
        "positional": _private_positional,
        "keyword": _private_keyword,
        "varargs": _private_varargs,
    }[kind]
    prepared, record = prepare_callable(original, _recipe(original))
    assert (
        record.to_dict()["sites"][0]["private_name"]
        not in original.__code__.co_varnames
    )
    value = jnp.asarray([1 + 2**-35], jnp.float64)
    actual = (
        prepared(value, **{"__jax2onnx_prepared_dtype_0": jnp.float16})
        if kind == "keyword"
        else (
            prepared(
                value, jnp.float16, **{"__jax2onnx_prepared_dtype_0_": jnp.bfloat16}
            )
            if kind == "positional"
            else prepared(value, jnp.float16)
        )
    )
    _same(actual, np.asarray([1 + 2**-35], np.float64))
    record.assert_current()


def test_identical_prepared_body_versions_keep_distinct_private_dtype_jit_caches() -> (
    None
):
    first, first_record = prepare_callable(
        _rounding, _recipe(_rounding, replacement=jnp.float32)
    )
    second, second_record = prepare_callable(
        _rounding, _recipe(_rounding, replacement=jnp.float64)
    )
    assert first.__code__.co_code == second.__code__.co_code
    assert first.__globals__ is not second.__globals__
    value = jnp.asarray([1e8 + 1], jnp.float64)
    compiled = {fn: jax.jit(fn) for fn in (_rounding, first, second)}
    for fn in (first, second, _rounding, second, first, _rounding):
        expected = np.ones(1, np.float64) if fn is second else np.zeros(1, np.float32)
        for call in (fn, compiled[fn]):
            _same(call(value), expected)
    first_record.assert_current()
    second_record.assert_current()


def _module_dispatch_closure(carrier: ModuleType, *, nested: bool = False) -> Any:
    if nested:

        def selected(x: Any) -> Any:
            return carrier.inner.dispatch(x).astype(jnp.float32)

    else:

        def selected(x: Any) -> Any:
            return carrier.dispatch(x).astype(jnp.float32)

    return selected


@pytest.mark.parametrize("nested", [False, True])
def test_actual_closure_module_self_dispatch_rejects_before_native_execution(
    nested: bool,
) -> None:
    carrier = ModuleType("preparation_recursive_closure")
    target = carrier
    if nested:
        target = ModuleType("preparation_recursive_closure_inner")
        carrier.inner = target
    original = _module_dispatch_closure(carrier, nested=nested)
    target.dispatch = original
    source = callable_preimage(original)
    with pytest.raises(SourcePreparationError, match="module self-dispatch"):
        prepare_callable(original, _recipe(original))
    assert callable_preimage(original) == source
    assert original.__closure__[0].cell_contents is carrier
    assert target.dispatch is original


@pytest.mark.parametrize("closure_self", [False, True])
def test_module_recursion_guard_uses_actual_closure_over_unread_global_shadow(
    closure_self: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = ModuleType("preparation_actual_closure")
    unread = ModuleType("preparation_unread_global")
    original = _module_dispatch_closure(actual)
    actual.dispatch = original if closure_self else _rounding
    unread.dispatch = _rounding if closure_self else original
    monkeypatch.setitem(globals(), "carrier", unread)
    recipe = _recipe(original)
    if closure_self:
        with pytest.raises(SourcePreparationError, match="module self-dispatch"):
            prepare_callable(original, recipe)
        return
    prepared, record = prepare_callable(original, recipe)
    values = jax.device_put(np.asarray([1e8 + 1, 16777216], np.float64))
    # The foreign helper is deliberately unselected: only the caller's final
    # cast changes. Retaining its FP32 rounding is part of this native witness.
    for call in (prepared, jax.jit(_fresh(prepared))):
        _same(call(values), np.zeros(2, np.float64))
    assert prepared.__closure__[0] is original.__closure__[0]
    assert actual.dispatch is _rounding and unread.dispatch is original
    record.assert_current()


def _local_module_dispatch(x: Any, carrier: ModuleType) -> Any:
    return carrier.dispatch(x).astype(jnp.float32)


def test_local_module_parameter_is_not_resolved_through_unread_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = ModuleType("preparation_actual_parameter")
    actual.dispatch = _rounding
    unread = ModuleType("preparation_unread_parameter_global")
    unread.dispatch = _local_module_dispatch
    monkeypatch.setitem(globals(), "carrier", unread)
    prepared, record = prepare_callable(
        _local_module_dispatch, _recipe(_local_module_dispatch)
    )
    values = jax.device_put(np.asarray([1e8 + 1, 16777216], np.float64))
    _same(prepared(values, actual), np.zeros(2, np.float64))
    _same(
        jax.jit(prepared, static_argnames=("carrier",))(values, actual),
        np.zeros(2, np.float64),
    )
    assert actual.dispatch is _rounding
    assert unread.dispatch is _local_module_dispatch
    record.assert_current()


_HEADER_CARRIER = ModuleType("preparation_header_carrier")
_HEADER_CARRIER.float32 = jnp.float32
_HEADER_CARRIER.dispatch = int


def _header_module_metadata(x: Any, retained: Any = _HEADER_CARRIER.dispatch) -> Any:
    return x.astype(_HEADER_CARRIER.float32)


def test_stripped_header_module_path_is_not_reexecuted_or_treated_as_body_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = _header_module_metadata.__defaults__
    assert defaults == (int,)
    monkeypatch.setattr(_HEADER_CARRIER, "dispatch", _header_module_metadata)
    prepared, record = prepare_callable(
        _header_module_metadata,
        _recipe(
            _header_module_metadata,
            expression="_HEADER_CARRIER.float32",
            root=BindingRef("global", "_HEADER_CARRIER", _HEADER_CARRIER),
        ),
    )
    assert prepared.__defaults__ is defaults
    assert prepared.__defaults__[0] is int
    values = jax.device_put(np.asarray([1e8 + 1, 16777216], np.float64))
    for call in (prepared, jax.jit(_fresh(prepared))):
        _same(call(values), np.asarray(values))
    assert _HEADER_CARRIER.dispatch is _header_module_metadata
    record.assert_current()


_GLOBAL_DISPATCH_CARRIER = ModuleType("preparation_global_dispatch")


def _global_module_recursive(x: Any) -> Any:
    return _GLOBAL_DISPATCH_CARRIER.dispatch(x).astype(jnp.float32)


def test_existing_global_module_self_dispatch_rejection_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _GLOBAL_DISPATCH_CARRIER, "dispatch", _global_module_recursive, raising=False
    )
    with pytest.raises(SourcePreparationError, match="module self-dispatch"):
        prepare_callable(_global_module_recursive, _recipe(_global_module_recursive))


def _builtin_module_dispatch(x: Any) -> Any:
    return _BUILTIN_CARRIER.dispatch(x).astype(jnp.float32)  # noqa: F821


def _builtin_direct_dispatch(x: Any) -> Any:
    return _BUILTIN_DISPATCH(x).astype(jnp.float32)  # noqa: F821


def _builtin_dispatch_fixture(template: Any) -> tuple[Any, Any]:
    namespace = dict(template.__globals__)
    namespace.pop("_BUILTIN_CARRIER", None)
    namespace.pop("_BUILTIN_DISPATCH", None)
    cached = dict(template.__builtins__)
    namespace["__builtins__"] = cached
    original = FunctionType(template.__code__, namespace, template.__name__)
    return original, cached


@pytest.mark.parametrize("route", ["direct", "module"])
def test_actual_cached_builtin_self_dispatch_is_rejected(route: str) -> None:
    template = (
        _builtin_direct_dispatch if route == "direct" else _builtin_module_dispatch
    )
    original, cached = _builtin_dispatch_fixture(template)
    if route == "direct":
        cached["_BUILTIN_DISPATCH"] = original
    else:
        carrier = ModuleType("preparation_builtin_carrier")
        carrier.dispatch = original
        cached["_BUILTIN_CARRIER"] = carrier
    assert original.__builtins__ is cached
    with pytest.raises(SourcePreparationError, match="recursive|module self-dispatch"):
        prepare_callable(original, _recipe(original))


@pytest.mark.parametrize("global_self", [False, True])
def test_actual_global_binding_has_precedence_over_cached_builtin_dispatch(
    global_self: bool,
) -> None:
    original, cached = _builtin_dispatch_fixture(_builtin_module_dispatch)
    global_carrier = ModuleType("preparation_actual_global_carrier")
    builtin_carrier = ModuleType("preparation_shadowed_builtin_carrier")
    global_carrier.dispatch = original if global_self else _rounding
    builtin_carrier.dispatch = _rounding if global_self else original
    original.__globals__["_BUILTIN_CARRIER"] = global_carrier
    cached["_BUILTIN_CARRIER"] = builtin_carrier
    if global_self:
        with pytest.raises(SourcePreparationError, match="module self-dispatch"):
            prepare_callable(original, _recipe(original))
        return
    prepared, record = prepare_callable(original, _recipe(original))
    values = jax.device_put(np.asarray([1e8 + 1, 16777216], np.float64))
    for call in (prepared, jax.jit(_fresh(prepared))):
        _same(call(values), np.zeros(2, np.float64))
    assert prepared.__builtins__ is original.__builtins__ is cached
    assert global_carrier.dispatch is _rounding
    assert builtin_carrier.dispatch is original
    record.assert_current()


def test_unknown_builtin_mapping_rejects_without_invoking_mapping_hooks() -> None:
    calls = []

    class BuiltinMapping(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:
            calls.append("get")
            raise AssertionError("custom builtin mapping hook executed")

        def __getitem__(self, key: Any) -> Any:
            calls.append("getitem")
            raise AssertionError("custom builtin mapping hook executed")

    namespace = dict(_builtin_length.__globals__)
    namespace.pop("len", None)
    cached = BuiltinMapping()
    dict.__setitem__(cached, "len", len)
    namespace["__builtins__"] = cached
    original = FunctionType(
        _builtin_length.__code__, namespace, _builtin_length.__name__
    )
    assert original.__builtins__ is cached
    with pytest.raises(SourcePreparationError, match="non-dictionary builtin dispatch"):
        prepare_callable(original, _recipe(original))
    assert not calls
