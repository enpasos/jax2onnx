# tests/extra_tests/framework/test_platform_index.py

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import to_onnx, user_interface as ui
from jax2onnx._compat.jax import JaxprEqn, Primitive, fresh_var_like
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.core.platform_index import PlatformIndexPlugin
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy.eye import JnpEyePlugin
from jax2onnx.precision import SampleCriteria, SourceSample, validate_source_samples
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


def _dispatch_diagonal(x: Any) -> Any:
    def default(a: Any) -> Any:
        indices = jnp.arange(a.shape[0])
        return a[indices, indices]

    def mosaic(a: Any) -> Any:
        return jax.lax.reduce(
            jax.lax.mul(jnp.eye(a.shape[0], dtype=a.dtype), a),
            np.asarray(0, dtype=a.dtype),
            jax.lax.add,
            (0,),
        )

    return jax.lax.platform_dependent(x, default=default, mosaic=mosaic)


def _selectors(value: Any) -> Iterator[JaxprEqn]:
    # JAX >=0.11 Jaxpr.jaxpr is self; descend only through its actual equations.
    if isinstance(value, dict):
        for child in value.values():
            yield from _selectors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _selectors(child)
    else:
        body = getattr(value, "jaxpr", value)
        for eqn in getattr(body, "eqns", ()):
            if eqn.primitive.name == "platform_index":
                yield eqn
            yield from _selectors(eqn.params)


def _native_dispatch_graph() -> Any:
    return jax.make_jaxpr(_dispatch_diagonal)(
        jax.ShapeDtypeStruct((2, 2), np.float32)
    ).jaxpr


def _selector() -> JaxprEqn:
    (eqn,) = _selectors(_native_dispatch_graph())
    return eqn


def _replace_equation(graph: Any, original: Any, replacement: Any) -> Any:
    return graph.replace(
        eqns=[replacement if eqn is original else eqn for eqn in graph.eqns]
    )


def _fresh_trace(fn: Any, *specs: Any) -> Any:
    # Distinct plain wrappers prevent a prior native trace cache from hiding
    # changed parameters/avals or equations inside either conditional branch.
    def fresh(*values: Any) -> Any:
        return fn(*values)

    return typed_trace_manifest(jax.make_jaxpr(fresh)(*specs))


@pytest.mark.parametrize("enable_double_precision", [False, True])
def test_native_selector_binds_scalar_int32_default(
    enable_double_precision: bool,
) -> None:
    eqn = _selector()
    assert eqn.params == {"platforms": (("mosaic",), None)}
    ctx = IRContext(
        opset=23,
        enable_double_precision=enable_double_precision,
        input_specs=[],
    )
    PlatformIndexPlugin().lower(ctx, eqn)
    value = ctx.require_value_for_var(eqn.outvars[0])
    assert value.dtype == ir.DataType.INT32 and tuple(value.shape) == ()
    assert value.const_value is not None
    actual = value.const_value.numpy()
    expected = np.asarray(1, dtype=np.int32)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
def test_native_diagonal_keeps_default_gather_with_offdiagonal_nans(
    tmp_path: Path, record_property: Any, policy: PrecisionPolicy
) -> None:
    pytest.importorskip("onnxruntime")
    # Capture the originals before export activates any converter patches.
    native_diagonal = jnp.diagonal
    native_slogdet = jnp.linalg.slogdet
    inputs = np.asarray([[3.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
    input_bytes = inputs.tobytes()
    expected = np.asarray([3.0, 2.0], dtype=np.float32)
    # Inspect actual nested native schemas, including the original failed path.
    for name, fn in (("diagonal", native_diagonal), ("slogdet", native_slogdet)):
        closed = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((2, 2), np.float32))
        selectors = list(_selectors(closed))
        record_property(f"native_{name}_platform_selectors", len(selectors))
        for eqn in selectors:
            assert eqn.params == {"platforms": (("mosaic",), None)}
            assert not eqn.invars and not eqn.effects
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == ()
            assert eqn.outvars[0].aval.dtype == np.dtype(np.int32)

    # The public selector witness still exercises this lowering if a newer
    # native diagonal no longer emits it. Both native functions remain intact.
    for name, fn in (("native", native_diagonal), ("selector", _dispatch_diagonal)):
        reference = np.asarray(fn(inputs))
        np.testing.assert_array_equal(reference, expected, strict=True)
        assert reference.tobytes() == expected.tobytes()
        if policy is PrecisionPolicy.SOURCE:
            spec = jax.ShapeDtypeStruct(inputs.shape, inputs.dtype)
            native_trace = _fresh_trace(fn, spec)
            with precision_policy_scope(policy), _activate_plugin_worlds():
                source_trace = _fresh_trace(fn, spec)
            assert source_trace == native_trace
        path = tmp_path / f"{name}_diagonal.onnx"
        before = tuple(
            getattr(jax.config, setting)
            for setting in (
                "jax_enable_x64",
                "jax_default_matmul_precision",
                "jax_numpy_dtype_promotion",
                "jax_numpy_rank_promotion",
            )
        )
        if policy is PrecisionPolicy.LEGACY:
            to_onnx(fn, inputs=[inputs], opset=23, return_mode="file", output_path=path)
        else:
            ui._to_onnx(
                fn,
                [inputs],
                opset=23,
                return_mode="file",
                output_path=path,
                precision_policy=policy,
            )
        after = tuple(
            getattr(jax.config, setting)
            for setting in (
                "jax_enable_x64",
                "jax_default_matmul_precision",
                "jax_numpy_dtype_promotion",
                "jax_numpy_rank_promotion",
            )
        )
        assert after == before and inputs.tobytes() == input_bytes
        assert jnp.diagonal is native_diagonal and jnp.linalg.slogdet is native_slogdet
        assert np.asarray(fn(inputs)).tobytes() == reference.tobytes()
        # This shared sample runner uses one ORT thread under both settings.
        for optimization in ("disabled", "all"):
            report = validate_source_samples(
                path,
                samples=[
                    SourceSample((inputs,), [reference], SampleCriteria("bitwise"))
                ],
                optimization=optimization,
            )
            report.require_sample_acceptance()


def test_source_eye_preserves_native_default_width_and_explicit_canonicalization() -> (
    None
):
    original = jnp.eye
    for x64, args, kwargs in (
        (False, (3,), {}),
        (True, (3,), {}),
        (False, (2,), {"dtype": np.float64}),
        (False, (2, 4, 1), {"dtype": np.float32}),
    ):
        with jax.enable_x64(x64):
            reference = np.asarray(original(*args, **kwargs))

            def fn() -> Any:
                return jnp.eye(*args, **kwargs)

            native_trace = _fresh_trace(fn)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                observed = np.asarray(fn())
                source_trace = _fresh_trace(fn)
            assert source_trace == native_trace
            np.testing.assert_array_equal(observed, reference, strict=True)
            assert observed.tobytes() == reference.tobytes()
            model = ui._to_onnx(
                fn, [], opset=23, precision_policy=PrecisionPolicy.SOURCE
            )
            assert jnp.eye is original
            for optimization in ("disabled", "all"):
                report = validate_source_samples(
                    model,
                    samples=[SourceSample((), [reference], SampleCriteria("bitwise"))],
                    optimization=optimization,
                )
                report.require_sample_acceptance()


def test_source_eye_forwards_original_calls_and_errors_before_private_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    tokens = tuple(object() for _ in range(5))
    error = TypeError("original eye argument error")

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("unknown"):
            raise error
        return tokens[-1]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE eye entered a replacement primitive")

    storage = "__orig_impl__eye"
    monkeypatch.setitem(
        vars(JnpEyePlugin._PRIM), storage, vars(JnpEyePlugin._PRIM).get(storage)
    )
    monkeypatch.setitem(vars(JnpEyePlugin._PRIM), "bind", forbidden)
    spec = next(
        spec
        for spec in JnpEyePlugin.binding_specs()
        if isinstance(spec, MonkeyPatchSpec)
    )
    wrapper = spec.make_value(original)
    forms = (
        ((tokens[0],), {}),
        (tokens[:4], {"device": tokens[0]}),
        (
            (),
            {"N": tokens[0], "M": None, "k": tokens[1], "dtype": None, "device": None},
        ),
    )
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        for args, kwargs in forms:
            assert wrapper(*args, **kwargs) is tokens[-1]
        with pytest.raises(TypeError) as caught:
            wrapper(tokens[0], unknown=tokens[1])
    assert caught.value is error
    assert calls == [*forms, ((tokens[0],), {"unknown": tokens[1]})]
    for (_, actual), (_, expected) in zip(
        calls, [*forms, ((), {"unknown": tokens[1]})], strict=True
    ):
        assert actual.keys() == expected.keys()
        assert all(actual[key] is value for key, value in expected.items())


@pytest.mark.parametrize(
    "platforms",
    [
        (("cpu",), None),
        (("cuda", "rocm"), None),
        (("tpu",), None),
        (None, ("mosaic",)),
        (("mosaic",),),
        (("mosaic", "cpu"), None),
        [("mosaic",), None],
    ],
    ids=["cpu", "gpu", "tpu", "reordered", "no-default", "mixed", "list"],
)
def test_device_dispatch_and_foreign_selectors_reject_before_ir(platforms: Any) -> None:
    graph = _native_dispatch_graph()
    (native,) = _selectors(graph)
    eqn = native.replace(params={"platforms": platforms})
    with pytest.raises(NotImplementedError, match="platform_index.*Mosaic/default"):
        PlatformIndexPlugin().lower(None, eqn)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(_replace_equation(graph, native, eqn), 23)
    assert caught.value.primitive == "platform_index"
    assert caught.value.category == "profile-exclusion"


def test_malformed_selector_schema_rejects_before_ir() -> None:
    eqn = _selector()
    for params in ({}, {**eqn.params, "unknown": None}):
        with pytest.raises(NotImplementedError, match="platform_index.*Mosaic/default"):
            PlatformIndexPlugin().lower(None, eqn.replace(params=params))
    faults = [
        {"invars": [eqn.outvars[0]]},
        {"outvars": []},
        {"outvars": eqn.outvars * 2},
        {"effects": {"unexpected-effect"}},
    ]
    for shape, dtype, weak_type in (
        ((1,), np.int32, False),
        ((), np.int64, False),
        ((), np.int32, True),
    ):
        faults.append(
            {
                "outvars": [
                    SimpleNamespace(
                        aval=SimpleNamespace(
                            shape=shape, dtype=np.dtype(dtype), weak_type=weak_type
                        )
                    )
                ]
            }
        )
    for fault in faults:
        with pytest.raises(ValueError, match="platform_index requires"):
            PlatformIndexPlugin().lower(None, eqn.replace(**fault))


def test_source_selector_rejects_malformed_results_and_effects() -> None:
    graph = _native_dispatch_graph()
    validate_source_jaxpr(graph, 23)
    (selector,) = _selectors(graph)
    for params in ({}, {**selector.params, "unknown": None}):
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(
                _replace_equation(graph, selector, selector.replace(params=params)), 23
            )
        assert caught.value.primitive == "platform_index"
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(
            _replace_equation(
                graph, selector, selector.replace(effects={"unexpected-effect"})
            ),
            23,
        )
    assert caught.value.primitive == "platform_index"
    for shape, dtype, weak in (
        ((1,), np.int32, False),
        ((), np.int64, False),
        ((), np.int32, True),
    ):
        malformed = SimpleNamespace(
            aval=jax.ShapeDtypeStruct(shape, dtype, weak_type=weak)
        )
        equations = []
        for eqn in graph.eqns:
            if eqn is selector:
                equations.append(eqn.replace(outvars=[malformed]))
            else:
                equations.append(
                    eqn.replace(
                        invars=[
                            malformed if var is selector.outvars[0] else var
                            for var in eqn.invars
                        ]
                    )
                )
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(graph.replace(eqns=equations), 23)
        assert caught.value.primitive == "platform_index"
        assert caught.value.category == "lowering-defect"


def test_source_platform_cond_rejects_metadata_and_detached_selectors() -> None:
    graph = _native_dispatch_graph()
    validate_source_jaxpr(graph, 23)
    (selector,) = _selectors(graph)
    (conditional,) = (eqn for eqn in graph.eqns if eqn.primitive.name == "cond")
    params = conditional.params
    assert params["branches_platforms"] == selector.params["platforms"]
    for metadata in (
        {"branches": params["branches"]},
        {**params, "branches_platforms": (None, ("mosaic",))},
        {**params, "branches_platforms": (("cpu",), None)},
        {**params, "unknown": None},
        {**params, "sharding": None},
        {**params, "branches": params["branches"] + params["branches"][:1]},
        {**params, "branches": list(params["branches"])},
    ):
        malformed = _replace_equation(
            graph, conditional, conditional.replace(params=metadata)
        )
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(malformed, 23)
        assert caught.value.primitive == "cond"

    # A copied/formal value with identical avals has no producer provenance.
    foreign = fresh_var_like(selector.outvars[0])
    detached_cond = conditional.replace(invars=[foreign, *conditional.invars[1:]])
    detached = _replace_equation(graph, conditional, detached_cond).replace(
        invars=[*graph.invars, foreign]
    )
    with pytest.raises(SourcePrecisionError, match="selector lacks"):
        validate_source_jaxpr(detached, 23)

    template = jax.make_jaxpr(lambda index: jax.lax.add(index, np.int32(0)))(
        jax.ShapeDtypeStruct((), np.int32)
    ).jaxpr.eqns[0]
    assert template.primitive.name == "add"
    derived = fresh_var_like(selector.outvars[0])
    arithmetic = template.replace(
        invars=[selector.outvars[0], *template.invars[1:]], outvars=[derived]
    )
    derived_cond = conditional.replace(invars=[derived, *conditional.invars[1:]])
    equations = []
    for eqn in graph.eqns:
        equations.extend([arithmetic, derived_cond] if eqn is conditional else [eqn])
    with pytest.raises(SourcePrecisionError, match="selector lacks"):
        validate_source_jaxpr(graph.replace(eqns=equations), 23)


def test_source_platform_cond_still_checks_the_unselected_branch() -> None:
    graph = _native_dispatch_graph()
    validate_source_jaxpr(graph, 23)
    (conditional,) = (eqn for eqn in graph.eqns if eqn.primitive.name == "cond")
    branches = conditional.params["branches"]
    unselected = branches[0]
    body = getattr(unselected, "jaxpr", unselected)
    original = body.outvars[0]
    wrong_signature = body.replace(
        outvars=[
            SimpleNamespace(
                aval=jax.ShapeDtypeStruct(
                    original.aval.shape, original.aval.dtype, weak_type=True
                )
            ),
            *body.outvars[1:],
        ]
    )
    corrupted_body = body.replace(
        eqns=[
            body.eqns[0].replace(primitive=Primitive("unreviewed_mosaic_branch")),
            *body.eqns[1:],
        ]
    )
    for malformed in (wrong_signature, corrupted_body):
        branch = (
            malformed if body is unselected else unselected.replace(jaxpr=malformed)
        )
        replacement = conditional.replace(
            params={**conditional.params, "branches": (branch, *branches[1:])}
        )
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(
                _replace_equation(graph, conditional, replacement), 23
            )
        assert caught.value.primitive in {"cond", "unreviewed_mosaic_branch"}
