# tests/extra_tests/converter/test_source_nnx_linear_reshape_dispatch.py

"""Native Linear/reshape dispatch; static diagnostics do not establish numerics."""

from collections.abc import Iterator
from copy import copy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
from typing import Any

from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx.linear import LinearPlugin
from jax2onnx.plugins.jax.numpy.reshape import JnpReshapePlugin
from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter import source_nnx_linear_fixtures as native_fixtures
from tests.extra_tests.converter.source_nnx_linear_fixtures import (
    assert_original_linear,
    expected_original_linear_kernel,
)
from tests.extra_tests.converter.source_precision_support import SourceSettings
from tests.extra_tests.converter.source_precision_diagnostics import (
    DiagnosticValidator,
    typed_trace_manifest,
)


_LINEAR = nnx.Linear.__call__
_LINEAR_HAS_OUT_SHARDING = "out_sharding" in inspect.signature(_LINEAR).parameters
_RESHAPE = jnp.reshape
_KERNEL_SHA = "d443c758235aff0c56e9af2d5862ae9807f6ca8e96d9aaf5289450663e4d3ef9"
_BIAS_SHA = "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1"
_NAMED_MODEL: Any = None


@onnx_function(name="NativeLinearBias")
def _named_linear(x: Any) -> Any:
    return _NAMED_MODEL(x), _NAMED_MODEL.kernel[...], _NAMED_MODEL.bias[...]


@onnx_function(name="NativeLinearResult")
def _named_linear_result(x: Any) -> Any:
    return _NAMED_MODEL(x)


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
def _restore_settings_and_deny_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    previous, effective = _globals(), _settings()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "Dispatch diagnostics do not establish target execution or numerics"
        )

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)
        assert _settings() == effective and _globals() == previous
        assert nnx.Linear.__call__ is _LINEAR and jnp.reshape is _RESHAPE
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (np.dtype(v.aval.dtype).name, tuple(v.aval.shape), bool(v.aval.weak_type))
        for v in variables
    )


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.name, array.shape, array.tobytes()


def _sha(value: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(getattr(value, "weak_type", False))
    )


def _original_model(ordinal: int) -> Any:
    # These are the unmodified raw recipes, including their unresolved dtype.
    case = PLUGIN_REGISTRY["nnx.linear"].metadata["testcases"][ordinal]
    assert case["testcase"] == (
        "linear_symbolic_batch" if ordinal == 0 else "linear_no_bias"
    )
    recipe = case["callable"]
    assert recipe._ctor is nnx.Linear and recipe._args == (128, 64)
    assert recipe._dtype is None and recipe._instance is None
    assert set(recipe._kwargs) == (
        {"dtype", "rngs"} if ordinal == 0 else {"dtype", "rngs", "use_bias"}
    )
    assert recipe._kwargs["rngs"].metadata["rng_trace"].seed == 0
    model = recipe.instantiate()
    assert_original_linear(model, recipe, ordinal)
    if ordinal == 0:
        assert _sha(model.bias[...]) == _BIAS_SHA
    else:
        assert recipe._kwargs["use_bias"] is False and model.bias is None
    return model


@pytest.mark.parametrize("defect", ["seed", "shape", "dtype", "bias"])
def test_native_fixture_rejects_changed_original_recipe(defect: str) -> None:
    model = _original_model(0)
    original = PLUGIN_REGISTRY["nnx.linear"].metadata["testcases"][0]["callable"]
    recipe = copy(original)
    recipe._kwargs = dict(original._kwargs)
    if defect == "seed":
        rng = copy(recipe._kwargs["rngs"])
        rng._metadata = dict(rng.metadata)
        rng._metadata["rng_trace"] = replace(rng.metadata["rng_trace"], seed=1)
        recipe._kwargs["rngs"] = rng
    elif defect == "shape":
        recipe._args = (127, 64)
    elif defect == "dtype":
        recipe = recipe.with_dtype(jnp.float32)
    else:
        recipe._kwargs["use_bias"] = False
    with pytest.raises(AssertionError):
        assert_original_linear(model, recipe, 0)
    assert original._dtype is None and original._instance is None
    assert original._kwargs["rngs"].metadata["rng_trace"].seed == 0


def test_native_fixture_rejects_unknown_kernel_and_wrong_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _original_model(0)
    recipe = PLUGIN_REGISTRY["nnx.linear"].metadata["testcases"][0]["callable"]
    minimum = ("0.8.1", "0.8.1", "0.12.1")
    # Exercise the real native payload against an incompatible claimed binding.
    wrong = (
        ("0.8.1", "0.8.1", "0.12.2")
        if native_fixtures._native_versions() == minimum
        else minimum
    )
    with monkeypatch.context() as patch:
        patch.setattr(native_fixtures, "_native_versions", lambda: wrong)
        with pytest.raises(AssertionError, match="Native kernel differs"):
            assert_original_linear(
                model,
                recipe,
                0,
                settings=SourceSettings(False, None, "cpu", "standard", "allow"),
            )
    model.kernel[...] = model.kernel[...] + jnp.float32(1)
    with pytest.raises(AssertionError, match="Native kernel differs"):
        assert_original_linear(model, recipe, 0)


def test_minimum_native_fixture_rejects_unmeasured_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _original_model(0)
    recipe = PLUGIN_REGISTRY["nnx.linear"].metadata["testcases"][0]["callable"]
    monkeypatch.setattr(
        native_fixtures, "_native_versions", lambda: ("0.8.1", "0.8.1", "0.12.1")
    )
    with jax.default_matmul_precision("highest"):
        with pytest.raises(AssertionError, match="Unmeasured minimum settings"):
            assert_original_linear(model, recipe, 0)


def _nonzero_model() -> Any:
    model = _original_model(0)
    model.bias[...] = jnp.asarray(np.linspace(-0.5, 0.5, 64), np.float32)
    assert np.count_nonzero(np.asarray(model.bias[...])) == 64
    return model


def _same_outputs(actual: Any, expected: Any) -> None:
    left, right = jax.tree.leaves(actual), jax.tree.leaves(expected)
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert _payload(a) == _payload(b)


def _compare(fn: Any, value: Any, *, execute: bool = True) -> Any:
    original = jax.make_jaxpr(fn)(value)
    if execute:
        eager, compiled = fn(value), jax.jit(fn)(value)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual = jax.make_jaxpr(fn)(value)
        if execute:
            _same_outputs(fn(value), eager)
            _same_outputs(jax.jit(fn)(value), compiled)
    assert typed_trace_manifest(actual) == typed_trace_manifest(original)
    assert [_payload(v) for v in actual.consts] == [
        _payload(v) for v in original.consts
    ]
    return actual


def _export(
    fn: Any,
    value: Any,
    *,
    mode: str = "proto",
    path: Path | None = None,
    named: bool = False,
) -> tuple[Any, onnx.ModelProto, DiagnosticValidator, list[Any]]:
    original = jax.make_jaxpr(fn)(value)
    observer, seen = DiagnosticValidator(), []
    settings, hidden = _settings(), _globals()

    def validate(graph: Any, opset: int) -> None:
        if not seen:
            assert _signature(_body(graph).invars) == _signature(_body(original).invars)
            assert _signature(_body(graph).outvars) == _signature(
                _body(original).outvars
            )
            if not named:
                # Older JAX supplies a bare body: closed payloads are checked below.
                assert typed_trace_manifest(_body(graph)) == typed_trace_manifest(
                    _body(original)
                )
        seen.append(graph)
        observer(graph, opset)

    options: dict[str, Any] = dict(return_mode=mode, opset=23)
    if path is not None:
        options["output_path"] = path
    artifact = user_interface._to_onnx(
        fn,
        [_spec(value)],
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=validate,
        **options,
    )
    assert seen and (_settings(), _globals()) == (settings, hidden)
    proto = (
        ir.to_proto(artifact)
        if isinstance(artifact, ir.Model)
        else onnx.load(artifact) if isinstance(artifact, str) else artifact
    )
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert tuple((v.dtype, v.shape, v.weak_type) for v in report.traced_outputs) == (
        _signature(_body(original).outvars)
    )
    assert {
        c.category: c.status
        for c in report.checks
        if c.category in {"lowering", "runtime-execution", "numerical-acceptance"}
    } == {
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact, proto, observer, seen


@pytest.mark.parametrize("ordinal", [0, 3])
@pytest.mark.parametrize("x64", [False, True])
def test_original_recipes_symbolic_and_concrete_native_semantics(
    ordinal: int, x64: bool
) -> None:
    with jax.enable_x64(x64):
        model = _original_model(ordinal)
        symbolic = jax_export.symbolic_shape("B,128")
        for shape in (symbolic, (2, 128), (5, 128)):
            is_symbolic = not isinstance(shape[0], int)
            value = (
                jax.ShapeDtypeStruct(shape, np.float32)
                if is_symbolic
                else jnp.asarray(
                    np.linspace(-1, 1, int(shape[0]) * 128), np.float32
                ).reshape(shape)
            )
            trace = _compare(model, value, execute=not is_symbolic)
            equations = _body(trace).eqns
            assert [e.primitive.name for e in equations] == (
                ["dot_general", "reshape", "add"] if ordinal == 0 else ["dot_general"]
            )
            dot = equations[0]
            assert dot.params["dimension_numbers"] == (((1,), (0,)), ((), ()))
            assert dot.params["precision"] is None
            assert dot.params["preferred_element_type"] is None
            assert dot.params.get("out_sharding") is None
            assert _signature(_body(trace).invars) == (("float32", shape, False),)
            assert _signature(_body(trace).outvars) == (
                ("float32", (shape[0], 64), False),
            )
            expected_kernel = expected_original_linear_kernel(
                PLUGIN_REGISTRY["nnx.linear"].metadata["testcases"][ordinal][
                    "callable"
                ],
                ordinal,
            )
            assert [_sha(c) for c in trace.consts] == (
                [expected_kernel, _BIAS_SHA] if ordinal == 0 else [expected_kernel]
            )


@pytest.mark.parametrize("kind", ["dtype", "precision", "custom", "preferred", "weak"])
def test_real_module_dtype_precision_callbacks_and_actual_weakness(kind: str) -> None:
    with jax.enable_x64(True):
        model = _original_model(0)
        calls: list[Any] = []
        if kind == "dtype":
            model.dtype = jnp.float64
        if kind == "precision":
            model.precision = jax.lax.Precision.HIGHEST
        if kind in {"custom", "preferred"}:
            promote = model.promote_dtype
            if kind == "preferred":
                model.preferred_element_type = jnp.float64

            def custom_promote(args: Any, *, dtype: Any = None) -> Any:
                calls.append(("promote", args, dtype))
                return promote(args, dtype=dtype)

            def custom_dot(*args: Any, **kwargs: Any) -> Any:
                calls.append(("dot", args, kwargs.copy()))
                return jax.lax.dot_general(*args, **kwargs)

            model.promote_dtype, model.dot_general = custom_promote, custom_dot
        value = jax.ShapeDtypeStruct((2, 128), np.float32, weak_type=kind == "weak")
        trace = _compare(lambda x: model(inputs=x), value, execute=False)
        names = [e.primitive.name for e in _body(trace).eqns]
        dot = next(e for e in _body(trace).eqns if e.primitive.name == "dot_general")
        if kind == "dtype":
            assert names.count("convert_element_type") == 3
            assert _signature(_body(trace).outvars) == (("float64", (2, 64), False),)
        elif kind == "precision":
            assert dot.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2
            assert dot.params["preferred_element_type"] is None
        elif kind in {"custom", "preferred"}:
            assert [c[0] for c in calls] == ["promote", "dot", "promote", "dot"]
            assert all(c[2] is jnp.float32 for c in calls if c[0] == "promote")
            for _, operands, kwargs in (c for c in calls if c[0] == "dot"):
                assert [tuple(a.shape) for a in operands[:2]] == [(2, 128), (128, 64)]
                assert operands[2] == (((1,), (0,)), ((), ()))
                expected_kwargs = {"precision": None}
                if _LINEAR_HAS_OUT_SHARDING:
                    expected_kwargs["out_sharding"] = None
                if kind == "preferred":
                    expected_kwargs["preferred_element_type"] = jnp.float64
                assert kwargs == expected_kwargs
            assert dot.params["preferred_element_type"] == (
                np.dtype(np.float64) if kind == "preferred" else None
            )
        else:
            assert _signature(_body(trace).invars)[0][2] is True
            assert _signature(_body(trace).outvars) == (("float32", (2, 64), False),)


def test_native_linear_inputs_keyword_and_positional_placement() -> None:
    model = _original_model(0)
    value = jnp.ones((2, 128), np.float32)
    _compare(lambda x: model(inputs=x), value)
    for call in (
        lambda x: model(inputs=x, out_sharding=None),
        lambda x: model(x, None),
    ):
        if _LINEAR_HAS_OUT_SHARDING:
            _compare(call, value)
        else:
            # Unsupported native placement hints stay unsupported in SOURCE.
            for evaluate in (lambda: call(value), lambda: jax.make_jaxpr(call)(value)):
                with pytest.raises(TypeError) as native:
                    evaluate()
                with (
                    precision_policy_scope(PrecisionPolicy.SOURCE),
                    _activate_plugin_worlds(),
                ):
                    with pytest.raises(TypeError) as source:
                        evaluate()
                assert type(source.value) is type(native.value)
                assert str(source.value) == str(native.value)


def test_explicit_module_precision_is_traced_and_rejected_by_existing_lowerer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _original_model(0)
    model.precision = jax.lax.Precision.HIGHEST
    monkeypatch.setattr(
        __import__(__name__, fromlist=["_NAMED_MODEL"]), "_NAMED_MODEL", model
    )

    def call(x: Any) -> Any:
        return _named_linear_result(x)

    value = jnp.ones((2, 128), np.float32)
    original = jax.make_jaxpr(call)(value)
    dot = next(e for e in _body(original).eqns if e.primitive.name == "dot_general")
    assert dot.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2
    with pytest.raises(
        NotImplementedError,
        match=r"SOURCE dot_general: precision .*native all-FP64 Precision enum pairs",
    ):
        _export(call, value, named=True)


@pytest.mark.parametrize("kind", ["weak", "bias", "wide", "complex", "fortran"])
def test_reshape_native_types_inferred_size_order_and_copy(kind: str) -> None:
    with jax.enable_x64(kind == "wide"):
        value = {
            "weak": lambda: jnp.asarray(0.25),
            "bias": lambda: jnp.asarray(np.arange(64), np.float32),
            "wide": lambda: jnp.asarray(np.arange(6), np.float64),
            "complex": lambda: jnp.asarray([1 + 2j, 3 - 4j], np.complex64),
            "fortran": lambda: jnp.asarray(np.arange(6), np.float32).reshape(2, 3),
        }[kind]()
        shape = (1,) if kind == "weak" else (1, -1)

        def model(x: Any) -> Any:
            return jnp.reshape(
                x, shape=shape, order="F" if kind == "fortran" else "C", copy=True
            )

        trace = _compare(model, value)
        assert all(e.primitive.name != "jax.numpy.reshape" for e in _body(trace).eqns)
        _compare(lambda x: jnp.reshape(x, shape, copy=False), value)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (((3,),), {}),
        (((2,),), {"order": "invalid"}),
        ((), {}),
        (((2,),), {"shape": (2,)}),
        (((2,),), {"unknown": True}),
    ],
)
def test_native_reshape_invalid_arguments_match_without_normalization(
    args: Any, kwargs: Any
) -> None:
    value = jnp.asarray([1, 2], np.float32)
    with pytest.raises(Exception) as expected:
        _RESHAPE(value, *args, **kwargs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(expected.value)) as actual:
            jnp.reshape(value, *args, **kwargs)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("name", ["linear", "reshape"])
def test_dispatch_captures_actual_original_and_forwards_identity_once(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    token, argument, option, result = object(), object(), object(), object()
    failure = ValueError("native failure identity")

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return result

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE must call its captured original without custom binding")

    plugin = LinearPlugin if name == "linear" else JnpReshapePlugin
    slot = "_ORIGINAL_LINEAR_CALL" if name == "linear" else "__orig_impl___reshape"
    owner = LinearPlugin if name == "linear" else JnpReshapePlugin._PRIM
    monkeypatch.setattr(owner, slot, getattr(owner, slot, None), raising=False)
    monkeypatch.setattr(plugin._PRIM, "bind", forbidden)
    spec = next(s for s in plugin.binding_specs() if isinstance(s, MonkeyPatchSpec))
    wrapper = spec.make_value(original)
    monkeypatch.setattr(owner, slot, forbidden)
    args = (token,) if name == "linear" else ()
    key = "inputs" if name == "linear" else "a"
    kwargs = {key: argument, "out_sharding" if name == "linear" else "shape": option}
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        assert wrapper(*args, **kwargs) is result
        with pytest.raises(ValueError) as caught:
            wrapper(*args, fail=True)
    assert caught.value is failure
    assert calls == [(args, kwargs), (args, {"fail": True})]
    assert calls[0][1][key] is argument


@pytest.mark.parametrize("operation", ["linear", "reshape"])
@pytest.mark.parametrize("transform", ["grad", "vmap"])
def test_native_derivative_and_batching_paths(operation: str, transform: str) -> None:
    model = _nonzero_model()
    fn = model if operation == "linear" else lambda x: jnp.reshape(x, (1, -1))
    value = jnp.asarray(np.linspace(-1, 1, 128), np.float32)
    if transform == "grad":

        def total(x: Any) -> Any:
            output = fn(x)
            return jax.lax.reduce_sum(output, tuple(range(output.ndim)))

        transformed = jax.grad(total)
    else:
        transformed = jax.vmap(fn)
        value = jnp.stack([value, -value])
    _compare(transformed, value)


def _assert_linear_artifact(proto: onnx.ModelProto, model: Any, batch: int) -> None:
    nodes = proto.graph.node
    assert [n.op_type for n in nodes].count("MatMul") == 1
    assert not any(n.op_type in {"Gemm", "Cast", "ConstantOfShape"} for n in nodes)
    has_bias = model.bias is not None
    assert sum(n.op_type == "Add" for n in nodes) == int(has_bias)
    for boundary, shape in (
        (proto.graph.input[0], (batch, 128)),
        (proto.graph.output[0], (batch, 64)),
    ):
        assert boundary.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert tuple(d.dim_value for d in boundary.type.tensor_type.shape.dim) == shape
    initializers = {
        t.name: onnx.numpy_helper.to_array(t) for t in proto.graph.initializer
    }
    matmul = next(n for n in nodes if n.op_type == "MatMul")
    assert _payload(initializers[matmul.input[1]]) == _payload(model.kernel[...])
    if has_bias:
        add = next(n for n in nodes if n.op_type == "Add")
        assert add.input[0] == matmul.output[0]
        reshapes = [n for n in nodes if n.op_type == "Reshape"]
        if reshapes:
            assert len(reshapes) == 1 and reshapes[0].output[0] == add.input[1]
            bias = initializers[reshapes[0].input[0]]
        else:
            bias = initializers[add.input[1]]
        assert bias.dtype == np.float32 and bias.size == 64
        assert bias.tobytes() == np.asarray(model.bias[...]).tobytes()
    else:
        assert [n.op_type for n in nodes] == ["MatMul"]
        assert len(initializers) == 1


@pytest.mark.parametrize("kind", ["bias", "no-bias", "nonzero"])
@pytest.mark.parametrize("batch", [2, 5])
def test_real_static_diagnostic_artifacts_preserve_parameters_and_arithmetic(
    kind: str, batch: int
) -> None:
    model = (
        _nonzero_model()
        if kind == "nonzero"
        else _original_model(3 if kind == "no-bias" else 0)
    )
    frozen = [_payload(model.kernel[...])]
    if model.bias is not None:
        frozen.append(_payload(model.bias[...]))
    value = jnp.asarray(np.linspace(-1, 1, batch * 128), np.float32).reshape(batch, 128)
    _compare(model, value)
    _, proto, observer, seen = _export(model, value)
    _assert_linear_artifact(proto, model, batch)
    assert len(seen) == len(observer.scopes) >= 2
    assert [_payload(model.kernel[...])] + (
        [_payload(model.bias[...])] if model.bias is not None else []
    ) == frozen


@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
def test_static_artifact_roundtrips_and_returned_constant_aliases(
    mode: str, tmp_path: Path
) -> None:
    model = _nonzero_model()
    kernel, bias = model.kernel[...], model.bias[...]
    frozen = [_payload(kernel), _payload(bias)]

    def aliases(x: Any) -> Any:
        return model(x), kernel, bias

    value = jnp.ones((2, 128), np.float32)
    _compare(aliases, value)
    artifact, proto, _, _ = _export(
        aliases,
        value,
        mode=mode,
        path=tmp_path / "linear.onnx" if mode == "file" else None,
    )
    _assert_linear_artifact(proto, model, 2)
    payloads = [onnx.numpy_helper.to_array(t) for t in proto.graph.initializer]
    assert all(any(_payload(v) == p for v in payloads) for p in frozen)
    by_name = {t.name: onnx.numpy_helper.to_array(t) for t in proto.graph.initializer}
    assert [_payload(by_name[o.name]) for o in proto.graph.output[1:]] == frozen
    assert [_payload(kernel), _payload(bias)] == frozen
    if isinstance(artifact, ir.Model):
        for node in artifact.graph:
            assert node.graph is artifact.graph
            assert all(v.producer().graph is artifact.graph for v in node.outputs)


def test_real_named_linear_body_bias_reshape_constants_and_call_site_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _nonzero_model()
    monkeypatch.setattr(
        __import__(__name__, fromlist=["_NAMED_MODEL"]), "_NAMED_MODEL", model
    )
    kernel, bias = model.kernel[...], model.bias[...]
    frozen = [_payload(kernel), _payload(bias)]

    def call(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            return _named_linear(x)

    value = jnp.ones((2, 128), np.float32)
    original = jax.make_jaxpr(call)(value)
    artifact, proto, observer, seen = _export(call, value, mode="ir", named=True)
    assert isinstance(artifact, ir.Model) and len(proto.functions) == 1
    function = proto.functions[0]
    assert function.name.startswith("NativeLinearBias")
    assert [n.op_type for n in function.node if n.op_type != "Constant"] == [
        "MatMul",
        "Reshape",
        "Add",
    ]
    child = next(
        s for s in observer.scopes if any(o[0] == "dot_general" for o in s.operations)
    )
    assert child.inputs == (("float32", (2, 128), False),)
    assert child.outputs == _signature(_body(original).outvars)
    assert [o[0] for o in child.operations] == ["dot_general", "reshape", "add"]
    child_graphs = [
        g for g in seen if any(e.primitive.name == "dot_general" for e in _body(g).eqns)
    ]
    assert child_graphs
    for graph in child_graphs:
        assert typed_trace_manifest(_body(graph)) == typed_trace_manifest(
            _body(original)
        )
        dot = _body(graph).eqns[0]
        assert (
            dot.params["precision"] is None
            and dot.params["preferred_element_type"] is None
        )
    ir_function = next(iter(artifact.functions.values()))
    assert not ir_function.graph.initializers
    for node in ir_function.graph:
        assert node.graph is ir_function.graph
        for operand in node.inputs:
            assert (
                operand in ir_function.graph.inputs
                or operand.producer().graph is ir_function.graph
            )
    payloads = [
        onnx.numpy_helper.to_array(a.t)
        for n in function.node
        for a in n.attribute
        if a.type == onnx.AttributeProto.TENSOR
    ]
    assert all(any(_payload(v) == p for v in payloads) for p in frozen)
    assert [_payload(kernel), _payload(bias)] == frozen


@pytest.mark.parametrize("reverse", [False, True])
def test_cache_orders_two_instances_captured_references_and_nested_restoration(
    reverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [_original_model(0), _nonzero_model()]
    value = jnp.ones((2, 128), np.float32)
    sequence = ("native", "legacy", "source", "native")
    for model in models:
        monkeypatch.setattr(
            __import__(__name__, fromlist=["_NAMED_MODEL"]), "_NAMED_MODEL", model
        )

        def call(x: Any) -> Any:
            return _named_linear_result(x)

        reference = jax.make_jaxpr(call)(value)
        eager, compiled = call(value), jax.jit(call)(value)
        for phase in reversed(sequence) if reverse else sequence:
            if phase == "native":
                assert typed_trace_manifest(
                    jax.make_jaxpr(call)(value)
                ) == typed_trace_manifest(reference)
                _same_outputs(call(value), eager)
                _same_outputs(jax.jit(call)(value), compiled)
            elif phase == "legacy":
                artifact = to_onnx(call, [_spec(value)])
                onnx.checker.check_model(artifact, full_check=True)
                assert any(
                    n.op_type == "Gemm" for f in artifact.functions for n in f.node
                )
            else:
                _, proto, _, _ = _export(call, value, named=True)
                payloads = [
                    onnx.numpy_helper.to_array(a.t)
                    for function in proto.functions
                    for node in function.node
                    for a in node.attribute
                    if a.type == onnx.AttributeProto.TENSOR
                ]
                assert any(_payload(v) == _payload(model.bias[...]) for v in payloads)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            patched = nnx.Linear.__call__, jnp.reshape
            with _activate_plugin_worlds():
                _same_outputs(_LINEAR(model, inputs=value), eager)
                _same_outputs(_RESHAPE(value, (1, -1)), value.reshape(1, -1))
            assert (nnx.Linear.__call__, jnp.reshape) == patched
            with pytest.raises(ValueError, match="scope failure"):
                with _activate_plugin_worlds():
                    raise ValueError("scope failure")
            assert (nnx.Linear.__call__, jnp.reshape) == patched


@pytest.mark.parametrize(
    "dtype_policy,rank_policy", [("standard", "allow"), ("strict", "raise")]
)
def test_local_contexts_restore_effective_and_global_settings(
    dtype_policy: str, rank_policy: str
) -> None:
    model = _original_model(0)
    hidden = _globals()
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion(dtype_policy),
        jax.numpy_rank_promotion(rank_policy),
    ):
        before = _settings()
        _compare(model, jnp.ones((2, 128), np.float32))
        _compare(lambda x: jnp.reshape(x, shape=(1, -1)), jnp.ones((64,), np.float64))
        assert _settings() == before and _globals() == hidden
