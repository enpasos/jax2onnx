# tests/extra_tests/converter/test_source_jnp_trig_dispatch.py

"""Native trig binding preserves source semantics; artifacts are not an audit."""

from collections.abc import Callable, Iterator
import hashlib
from pathlib import Path
from typing import Any

import jax
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
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl
from jax2onnx.plugins.jax.numpy.cos import JnpCosPlugin
from jax2onnx.plugins.jax.numpy.sin import JnpSinPlugin
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_diagnostics import (
    DiagnosticValidator,
    typed_trace_manifest,
)


_ORIGINALS = {name: getattr(jnp, name) for name in ("sin", "cos")}
_PLUGINS = {"sin": JnpSinPlugin, "cos": JnpCosPlugin}
_TRIG_CONSTANT = np.asarray([[0.125, -0.25]], np.float32)


@onnx_function(name="NativeTrigConstant")
def _named_trig(x: Any) -> Any:
    shifted = x + _TRIG_CONSTANT
    return jnp.sin(shifted), jnp.cos(shifted), _TRIG_CONSTANT


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
        pytest.fail("Native trig binding tests do not establish target numerics")

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
        assert all(getattr(jnp, name) is fn for name, fn in _ORIGINALS.items())
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (np.dtype(var.aval.dtype).name, tuple(var.aval.shape), bool(var.aval.weak_type))
        for var in variables
    )


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(getattr(value, "weak_type", False))
    )


def _operation(name: str) -> Callable[..., Any]:
    def operation(value: Any) -> Any:
        return getattr(jnp, name)(value)

    return operation


def _native(fn: Any, value: Any) -> tuple[Any, list[np.ndarray], list[np.ndarray]]:
    assert all(getattr(jnp, name) is orig for name, orig in _ORIGINALS.items())
    trace = jax.make_jaxpr(fn)(value)
    eager = [np.asarray(v) for v in jax.tree.leaves(fn(value))]
    compiled = [np.asarray(v) for v in jax.tree.leaves(jax.jit(fn)(value))]
    assert [(v.dtype, v.shape) for v in eager] == [(v.dtype, v.shape) for v in compiled]
    return trace, eager, compiled


def _same_outputs(actual: Any, expected: list[np.ndarray]) -> None:
    arrays = [np.asarray(v) for v in jax.tree.leaves(actual)]
    assert len(arrays) == len(expected)
    for left, right in zip(arrays, expected, strict=True):
        assert (left.dtype, left.shape, left.tobytes()) == (
            right.dtype,
            right.shape,
            right.tobytes(),
        )


def _export(
    fn: Any,
    value: Any,
    *,
    mode: str = "proto",
    path: Path | None = None,
    named: bool = False,
) -> tuple[Any, onnx.ModelProto, DiagnosticValidator]:
    original, _, _ = _native(fn, value)
    observer = DiagnosticValidator()
    seen = []
    settings, hidden = _settings(), _globals()

    def validate(graph: Any, opset: int) -> None:
        if not seen:
            assert _signature(_body(graph).invars) == _signature(_body(original).invars)
            assert _signature(_body(graph).outvars) == _signature(
                _body(original).outvars
            )
            if not named:
                # A validator may receive a bare Jaxpr on the older stack.
                # Closed constant payloads are compared separately below.
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
    if isinstance(artifact, ir.Model):
        proto = ir.to_proto(artifact)
    elif isinstance(artifact, str):
        proto = onnx.load(artifact)
    else:
        proto = artifact
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert tuple(
        (v.dtype, v.shape, v.weak_type) for v in report.traced_outputs
    ) == _signature(_body(original).outvars)
    assert {
        c.category: c.status
        for c in report.checks
        if c.category in {"lowering", "runtime-execution", "numerical-acceptance"}
    } == {
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact, proto, observer


@pytest.mark.parametrize("name", _ORIGINALS)
@pytest.mark.parametrize(
    "kind",
    [
        "python-weak",
        "jax-weak",
        "scalar",
        "vector",
        "wide",
        "integer",
        "half",
        "bf16",
        "complex",
    ],
)
def test_complete_native_typed_trace_and_output_weakness(name: str, kind: str) -> None:
    with jax.enable_x64(kind == "wide"):
        values = {
            "python-weak": lambda: 0.25,
            "jax-weak": lambda: jnp.asarray(0.25),
            "scalar": lambda: jnp.asarray(0.25, np.float32),
            "vector": lambda: jnp.asarray([0.25, -0.5], np.float32),
            "wide": lambda: jnp.asarray([[0.25, -0.5], [1.0, 2.0]], np.float64),
            "integer": lambda: jnp.asarray([0, 1, 2], np.int32),
            "half": lambda: jnp.asarray([0.25, -0.5], np.float16),
            "bf16": lambda: jnp.asarray([0.25, -0.5], jnp.bfloat16),
            "complex": lambda: jnp.asarray([0.25 + 0.5j], np.complex64),
        }
        value = values[kind]()
        fn = _operation(name)
        original, eager, compiled = _native(fn, value)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = jax.make_jaxpr(fn)(value)
            _same_outputs(fn(value), eager)
            _same_outputs(jax.jit(fn)(value), compiled)
        assert typed_trace_manifest(actual) == typed_trace_manifest(original)
        assert _signature(_body(actual).outvars) == _signature(_body(original).outvars)
        assert any(e.primitive.name == name for e in _body(actual).eqns)
        assert not any(
            e.primitive.name == f"jax.numpy.{name}" for e in _body(actual).eqns
        )
        assert not actual.effects
        assert all(
            e.params == {"accuracy": None} and not e.effects
            for e in _body(actual).eqns
            if e.primitive.name == name
        )


@pytest.mark.parametrize("name", _ORIGINALS)
def test_actual_plugin_binder_captures_and_forwards_once_without_retry(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _PLUGINS[name]
    primitive = plugin._PRIM
    monkeypatch.setattr(primitive, f"__orig_impl___{name}", None, raising=False)
    calls = []
    token, option = object(), {"unchanged": object()}
    failure = ValueError("original unary failure")

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return option

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("An opted-in SOURCE call must not retry the custom primitive")

    monkeypatch.setattr(primitive, "bind", forbidden)
    spec = next(s for s in plugin.binding_specs() if isinstance(s, MonkeyPatchSpec))
    wrapper = spec.make_value(original)
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        assert wrapper(token, keyword=option) is option
        with pytest.raises(ValueError) as caught:
            wrapper(token, fail=True)
    assert caught.value is failure
    assert calls == [((token,), {"keyword": option}), ((token,), {"fail": True})]
    assert get_orig_impl(primitive, name) is original


@pytest.mark.parametrize("name", _ORIGINALS)
@pytest.mark.parametrize("dtype_policy", ["standard", "strict"])
@pytest.mark.parametrize("rank_policy", ["allow", "raise"])
def test_native_policies_keyword_errors_and_nested_restoration(
    name: str, dtype_policy: str, rank_policy: str
) -> None:
    fn = _operation(name)
    value = jnp.asarray([[0.25, -0.5]], np.float32)
    hidden = _globals()
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion(dtype_policy),
        jax.numpy_rank_promotion(rank_policy),
    ):
        before = _settings()
        original, _, _ = _native(fn, value)
        with pytest.raises(TypeError) as expected:
            _ORIGINALS[name](value, unknown_keyword=True)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            outer = getattr(jnp, name)
            assert get_orig_impl(_PLUGINS[name]._PRIM, name) is _ORIGINALS[name]
            with _activate_plugin_worlds():
                actual = jax.make_jaxpr(fn)(value)
                with pytest.raises(type(expected.value)) as caught:
                    getattr(jnp, name)(value, unknown_keyword=True)
            assert getattr(jnp, name) is outer
        assert str(caught.value) == str(expected.value)
        assert typed_trace_manifest(actual) == typed_trace_manifest(original)
        assert _settings() == before and _globals() == hidden


@pytest.mark.parametrize("name", _ORIGINALS)
@pytest.mark.parametrize("transform", ["grad", "jvp", "vmap"])
def test_native_derivative_and_batching_paths(name: str, transform: str) -> None:
    operation = _operation(name)
    value = jnp.asarray(0.25, np.float32)
    if transform == "grad":
        fn = jax.grad(operation)
    elif transform == "jvp":
        tangent = jnp.asarray(1, np.float32)

        def fn(x: Any) -> Any:
            return jax.jvp(operation, (x,), (tangent,))

    else:
        value = jnp.asarray([[0.25, -0.5], [1.0, 2.0]], np.float32)
        fn = jax.vmap(operation)
    original, eager, compiled = _native(fn, value)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual = jax.make_jaxpr(fn)(value)
        _same_outputs(fn(value), eager)
        _same_outputs(jax.jit(fn)(value), compiled)
    assert typed_trace_manifest(actual) == typed_trace_manifest(original)
    _export(fn, value)


@pytest.mark.parametrize("name", _ORIGINALS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
def test_diagnostic_artifact_forms_keep_existing_real_lowerings(
    name: str, dtype: Any, mode: str, tmp_path: Path
) -> None:
    with jax.enable_x64(dtype is np.float64):
        value = jnp.asarray([[0.25, -0.5, 1.0], [2.0, 3.0, 4.0]], dtype)
        artifact, proto, _ = _export(
            _operation(name),
            value,
            mode=mode,
            path=tmp_path / "trig.onnx" if mode == "file" else None,
        )
        expected = (
            ["Mul", "Sin", "Mul", "Mul", "Sub"]
            if name == "cos" and dtype is np.float64
            else [name.title()]
        )
        assert [node.op_type for node in proto.graph.node] == expected
        assert not proto.functions
        enum = (
            onnx.TensorProto.DOUBLE if dtype is np.float64 else onnx.TensorProto.FLOAT
        )
        for boundary in (*proto.graph.input, *proto.graph.output):
            assert boundary.type.tensor_type.elem_type == enum
            assert [d.dim_value for d in boundary.type.tensor_type.shape.dim] == [2, 3]
        if isinstance(artifact, ir.Model):
            assert [node.op_type for node in artifact.graph] == expected
            for node in artifact.graph:
                assert node.graph is artifact.graph
                for result in node.outputs:
                    assert tuple(result.shape) == (
                        2,
                        3,
                    ) and result.dtype == ir.DataType(enum)
        if name == "cos" and dtype is np.float64:
            constants = [onnx.numpy_helper.to_array(t) for t in proto.graph.initializer]
            assert sorted(float(v) for v in constants) == [0.5, 1.0, 2.0]
            assert all(v.dtype == np.float64 for v in constants)


@pytest.mark.parametrize("name", _ORIGINALS)
def test_explicit_cast_closed_constant_and_direct_alias_are_preserved(
    name: str,
) -> None:
    with jax.enable_x64(True):
        constant = np.asarray([[0.125, -0.25]], np.float64)
        frozen = constant.tobytes()

        def model(x: Any) -> Any:
            return getattr(jnp, name)(x.astype(jnp.float64) + constant), constant

        value = jnp.asarray([[0.25, -0.5]], np.float32)
        original, _, _ = _native(model, value)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = jax.make_jaxpr(model)(value)
        assert typed_trace_manifest(actual) == typed_trace_manifest(original)
        assert len(actual.consts) == len(original.consts) == 1
        assert np.asarray(actual.consts[0]).tobytes() == frozen
        artifact, proto, _ = _export(model, value, mode="ir")
        assert isinstance(artifact, ir.Model)
        casts = [n for n in proto.graph.node if n.op_type == "Cast"]
        assert len(casts) == 1 and casts[0].attribute[0].i == onnx.TensorProto.DOUBLE
        retained = [onnx.numpy_helper.to_array(t) for t in proto.graph.initializer]
        assert (
            sum(
                a.shape == constant.shape
                and a.dtype == constant.dtype
                and a.tobytes() == frozen
                for a in retained
            )
            == 1
        )
        assert constant.tobytes() == frozen


@pytest.mark.parametrize("name", _ORIGINALS)
def test_native_integer_conversion_stays_visible_in_actual_artifact(name: str) -> None:
    value = jnp.asarray([0, 1, 2], np.int32)
    _, proto, _ = _export(_operation(name), value)
    assert [n.op_type for n in proto.graph.node] == ["Cast", name.title()]
    cast = proto.graph.node[0]
    assert [(a.name, a.i) for a in cast.attribute] == [("to", onnx.TensorProto.FLOAT)]
    assert proto.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert proto.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_named_function_owns_typed_constant_and_nested_policy_boundaries() -> None:
    constant = _TRIG_CONSTANT
    frozen = hashlib.sha256(constant.tobytes()).hexdigest()

    def model(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            return _named_trig(x)

    value = jnp.asarray([[0.25, -0.5]], np.float32)
    artifact, proto, observer = _export(model, value, mode="ir", named=True)
    assert isinstance(artifact, ir.Model) and len(proto.functions) == 1
    assert len(observer.scopes) >= 2
    child = next(s for s in observer.scopes if any(o[0] == "sin" for o in s.operations))
    assert child.inputs == (("float32", (1, 2), False),)
    assert child.outputs == (("float32", (1, 2), False),) * 3
    assert [o[0] for o in child.operations] == ["add", "sin", "cos"]
    function = proto.functions[0]
    assert function.name.startswith("NativeTrigConstant")
    assert [n.op_type for n in function.node if n.op_type != "Constant"] == [
        "Add",
        "Sin",
        "Cos",
    ]
    assert len(artifact.functions) == 1
    ir_function = next(iter(artifact.functions.values()))
    assert not ir_function.graph.initializers
    for node in ir_function.graph:
        assert node.graph is ir_function.graph
        for operand in node.inputs:
            assert operand in ir_function.graph.inputs or (
                operand.producer() is not None
                and operand.producer().graph is ir_function.graph
            )
        for result in node.outputs:
            assert result.dtype == ir.DataType.FLOAT and tuple(result.shape) == (1, 2)
    payloads = [
        onnx.numpy_helper.to_array(a.t)
        for n in function.node
        for a in n.attribute
        if a.type == onnx.AttributeProto.TENSOR
    ]
    assert any(
        v.dtype == np.float32
        and v.shape == constant.shape
        and hashlib.sha256(v.tobytes()).hexdigest() == frozen
        for v in payloads
    )
    assert hashlib.sha256(constant.tobytes()).hexdigest() == frozen
    assert all(
        o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for o in proto.graph.output
    )


@pytest.mark.parametrize("name", _ORIGINALS)
@pytest.mark.parametrize("reverse", [False, True])
def test_native_legacy_source_cache_orders_without_clearing(
    name: str, reverse: bool
) -> None:
    value = jnp.asarray([0.25, -0.5], np.float32)
    fn = _operation(name)
    original, eager, compiled = _native(fn, value)
    sequence = ("native", "legacy", "source", "native")
    for phase in reversed(sequence) if reverse else sequence:
        if phase == "native":
            trace, after_eager, after_compiled = _native(fn, value)
            assert typed_trace_manifest(trace) == typed_trace_manifest(original)
            _same_outputs(after_eager, eager)
            _same_outputs(after_compiled, compiled)
        elif phase == "legacy":
            with (
                precision_policy_scope(PrecisionPolicy.LEGACY),
                _activate_plugin_worlds(),
            ):
                trace = jax.make_jaxpr(fn)(value)
            assert [e.primitive.name for e in _body(trace).eqns] == [
                f"jax.numpy.{name}"
            ]
            artifact = to_onnx(fn, [_spec(value)])
            onnx.checker.check_model(artifact, full_check=True)
        else:
            _export(fn, value)


@pytest.mark.parametrize("name", _ORIGINALS)
def test_native_trig_remains_excluded_from_default_source_profile(name: str) -> None:
    hidden = _globals()
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        before = _settings()
        artifact = user_interface._to_onnx(
            _operation(name),
            [jax.ShapeDtypeStruct((2,), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
        assert isinstance(artifact, onnx.ModelProto)
        onnx.checker.check_model(artifact, full_check=True)
        assert any(node.op_type == name.title() for node in artifact.graph.node)
        report = get_precision_report(artifact)
        assert report is not None and report.profile == "source-static-v1"
        assert _settings() == before and _globals() == hidden
        with pytest.raises(SourcePrecisionError) as caught:
            user_interface._to_onnx(
                jax.lax.erf,
                [jax.ShapeDtypeStruct((2,), np.float32)],
                precision_policy=PrecisionPolicy.SOURCE,
            )
        assert _settings() == before and _globals() == hidden
    assert caught.value.category == "profile-exclusion"
    assert caught.value.primitive == "erf"
