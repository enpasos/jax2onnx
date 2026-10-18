# tests/extra_tests/converter/test_source_pointwise_mul_exp.py

"""Native Exp and finite Mul composition through the audited export facade."""

from collections.abc import Callable, Iterator
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface as ui
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
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy.exp import JnpExpPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


# Criteria are fixed before any target execution. Low precision receives only
# structural and exactly representable controls, never a general tolerance.
_BITWISE = SampleCriteria("bitwise")
_NAN_CLASS = SampleCriteria("exact", equal_nan=True)
_FINITE = {
    "float32": SampleCriteria("tolerant", rtol=1e-5, atol=1e-5),
    "float64": SampleCriteria("tolerant", rtol=1e-7, atol=1e-7),
}
_ORIGINALS = {name: getattr(jnp, name) for name in ("exp", "multiply", "add", "where")}


@onnx_function(name="SourcePointwiseExpMul")
def _named(x: Any, scale: Any) -> Any:
    return jnp.exp(x) * scale


@onnx_function(name="NestedSourcePointwiseExpMul")
def _nested(x: Any, scale: Any) -> Any:
    return _named(x, scale)


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
def _restore_state() -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)
        assert all(getattr(jnp, name) is fn for name, fn in _ORIGINALS.items())
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _specs(shapes: Any, dtype: Any = np.float32, *, weak: bool = False) -> list[Any]:
    (batch,) = export.symbolic_shape("B")
    return [
        jax.ShapeDtypeStruct(
            tuple(batch if dim == "B" else dim for dim in shape), dtype, weak_type=weak
        )
        for shape in shapes
    ]


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (
            np.dtype(var.aval.dtype).name,
            normalized_shape_dimensions(var.aval.shape),
            bool(var.aval.weak_type),
        )
        for var in variables
    )


def _reference(fn: Callable[..., Any], values: Any) -> list[np.ndarray]:
    return [np.asarray(value) for value in jax.tree.leaves(fn(*values))]


def _artifact(
    fn: Callable[..., Any],
    specs: list[Any],
    *,
    feeds: tuple[Any, ...] = (),
    mode: str = "proto",
    opset: int = 23,
    path: Path | None = None,
    named: bool = False,
) -> tuple[Any, onnx.ModelProto, Any, list[Any]]:
    assert all(getattr(jnp, name) is original for name, original in _ORIGINALS.items())
    native = jax.make_jaxpr(fn)(*specs)
    references = [(_reference(fn, v), _reference(jax.jit(fn), v)) for v in feeds]
    before = _globals(), _settings()
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(fn)(*specs)
        assert _signature(_body(activated).invars) == _signature(_body(native).invars)
        assert _signature(_body(activated).outvars) == _signature(_body(native).outvars)
        if named:
            # Named bindings intentionally replace native equations with a
            # Function call and have no JAX MLIR rule. Their actual child IR
            # formals/body and independent native eager/JIT references are
            # checked by the named composition regression below.
            assert any(
                e.primitive.name.startswith("onnx_fn::") for e in _body(activated).eqns
            )
        else:
            assert typed_trace_manifest(activated) == typed_trace_manifest(native)
            for values, (eager, compiled) in zip(feeds, references, strict=True):
                for actual, expected in zip(_reference(fn, values), eager, strict=True):
                    assert (
                        actual.dtype == expected.dtype
                        and actual.shape == expected.shape
                    )
                    assert actual.tobytes() == expected.tobytes()
                for actual, expected in zip(
                    _reference(jax.jit(fn), values), compiled, strict=True
                ):
                    assert (
                        actual.dtype == expected.dtype
                        and actual.shape == expected.shape
                    )
                    assert actual.tobytes() == expected.tobytes()
    options: dict[str, Any] = {"return_mode": mode, "opset": opset}
    if path is not None:
        options["output_path"] = path
    try:
        artifact = ui._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, **options
        )
    finally:
        assert (_globals(), _settings()) == before
    proto = (
        ir.to_proto(artifact)
        if isinstance(artifact, ir.Model)
        else onnx.load(artifact) if isinstance(artifact, str) else artifact
    )
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert tuple(
        (v.dtype, v.shape, v.weak_type) for v in report.traced_outputs
    ) == _signature(_body(native).outvars)
    return artifact, proto, report, references


def _accept(proto: Any, feeds: Any, references: Any, criteria: SampleCriteria) -> None:
    for optimization in ("disabled", "all"):
        for reference_index in (0, 1):
            result = validate_source_samples(
                proto,
                samples=[
                    SourceSample(values, expected[reference_index], criteria)
                    for values, expected in zip(feeds, references, strict=True)
                ],
                optimization=optimization,
            )
            result.require_sample_acceptance()
            assert result.status is CheckStatus.VALIDATED


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.float64])
@pytest.mark.parametrize("weak", [False, True])
def test_native_exp_preserves_complete_typed_trace_and_weakness(
    x64: bool, dtype: Any, weak: bool
) -> None:
    with jax.enable_x64(x64):
        specs = _specs([()], dtype, weak=weak)
        if dtype is np.float64 and not x64:
            before = _globals(), _settings()
            with pytest.raises(SourcePrecisionError) as caught:
                ui._to_onnx(jnp.exp, specs, precision_policy=PrecisionPolicy.SOURCE)
            assert caught.value.category == "configuration-conflict"
            assert (_globals(), _settings()) == before
            return

        def model(x: Any) -> Any:
            return jnp.exp(x)

        _, proto, report, _ = _artifact(model, specs)
        assert [node.op_type for node in proto.graph.node] == ["Exp"]
        assert report.traced_outputs[0].weak_type is weak
        assert report.traced_outputs[0].dtype == np.dtype(dtype).name


def test_actual_exp_binding_forwards_arguments_and_original_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive = JnpExpPlugin._PRIM
    monkeypatch.setattr(primitive, "__orig_impl___exp", None, raising=False)
    sentinel, argument, option = object(), object(), {"identity": object()}
    failure, calls = TypeError("original Exp keyword failure"), []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if "fail" in kwargs:
            raise failure
        return sentinel

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE Exp must not retry the custom primitive")

    monkeypatch.setattr(primitive, "bind", forbidden)
    wrapper = next(
        spec
        for spec in JnpExpPlugin.binding_specs()
        if isinstance(spec, MonkeyPatchSpec)
    ).make_value(original)
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        assert wrapper(argument, option=option) is sentinel
        with pytest.raises(TypeError) as caught:
            wrapper(argument, fail=option)
    assert caught.value is failure
    assert calls == [((argument,), {"option": option}), ((argument,), {"fail": option})]


@pytest.mark.parametrize("keyword", ["out", "where", "unknown"])
def test_native_exp_keyword_errors_match_the_unpatched_original(keyword: str) -> None:
    value = jnp.asarray([0.0], np.float32)
    with pytest.raises(Exception) as native:
        _ORIGINALS["exp"](value, **{keyword: value})
    before = _globals(), _settings()
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(native.value)) as actual:
            jnp.exp(value, **{keyword: value})
    assert str(actual.value) == str(native.value)
    assert (_globals(), _settings()) == before


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.complex64])
def test_native_promotion_remains_traceable_without_admitting_integer_or_complex_boundaries(
    x64: bool, dtype: Any
) -> None:
    with jax.enable_x64(x64):
        values = (jnp.asarray([0, 1], dtype),)
        specs = _specs([(2,)], dtype)

        def model(x: Any) -> Any:
            return jnp.exp(x)

        native = jax.make_jaxpr(model)(*specs)
        eager, compiled = _reference(model, values), _reference(jax.jit(model), values)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            assert typed_trace_manifest(
                jax.make_jaxpr(model)(*specs)
            ) == typed_trace_manifest(native)
            assert _reference(model, values)[0].tobytes() == eager[0].tobytes()
            assert (
                _reference(jax.jit(model), values)[0].tobytes() == compiled[0].tobytes()
            )
        before = _globals(), _settings()
        with pytest.raises(SourcePrecisionError) as caught:
            ui._to_onnx(model, specs, precision_policy=PrecisionPolicy.SOURCE)
        assert caught.value.category == "profile-exclusion"
        assert (_globals(), _settings()) == before


@pytest.mark.parametrize("primitive", ["exp", "mul"])
@pytest.mark.parametrize("present", [False, True])
def test_native_default_parameters_accept_absent_and_none(
    primitive: str, present: bool
) -> None:
    fn = jax.lax.exp if primitive == "exp" else jax.lax.mul
    native = _body(
        jax.make_jaxpr(fn)(*_specs([("B", 3)] * (1 if primitive == "exp" else 2)))
    )
    original = native.eqns[0]
    params = dict(original.params)
    key = "accuracy" if primitive == "exp" else "out_dtype"
    if present:
        params[key] = None
    else:
        params.pop(key, None)
    equation = SimpleNamespace(
        primitive=original.primitive,
        invars=original.invars,
        outvars=original.outvars,
        params=params,
        effects=(),
    )
    graph = SimpleNamespace(
        invars=native.invars,
        constvars=native.constvars,
        outvars=native.outvars,
        eqns=[equation],
        effects=(),
    )
    validate_source_jaxpr(graph, 23)


@pytest.mark.parametrize("left_weak", [False, True])
@pytest.mark.parametrize("right_weak", [False, True])
def test_symbolic_mul_result_uses_native_operand_weakness_conjunction(
    left_weak: bool, right_weak: bool
) -> None:
    specs = _specs([("B", 3), (1, 3)])
    specs = [
        jax.ShapeDtypeStruct(spec.shape, spec.dtype, weak_type=weak)
        for spec, weak in zip(specs, (left_weak, right_weak), strict=True)
    ]
    _, proto, report, _ = _artifact(lambda x, y: jax.lax.mul(x, y), specs)
    assert [node.op_type for node in proto.graph.node] == ["Mul"]
    assert report.traced_outputs[0].weak_type is (left_weak and right_weak)


@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.float64])
def test_actual_static_exp_mul_artifacts_keep_source_ports_and_casts(
    opset: int, mode: str, dtype: Any, tmp_path: Path
) -> None:
    with jax.enable_x64(True):

        def model(x: Any, scale: Any) -> Any:
            return jnp.exp(x) * scale

        _, proto, report, _ = _artifact(
            model,
            _specs([(2, 3), ()], dtype),
            mode=mode,
            opset=opset,
            path=tmp_path / "pointwise.onnx" if mode == "file" else None,
        )
        assert [node.op_type for node in proto.graph.node] == ["Exp", "Mul"]
        assert all(v.exported.dtype == np.dtype(dtype).name for v in report.inputs)
        assert report.exported_outputs[0].shape == (2, 3)
        assert report.exported_outputs[0].dtype == np.dtype(dtype).name
        assert proto.opset_import[0].version == opset

        def cast_model(x: Any) -> Any:
            return jnp.exp(x.astype(jnp.float64)) * np.float64(2)

        _, cast, cast_report, _ = _artifact(cast_model, _specs([(2,)], np.float32))
        assert [node.op_type for node in cast.graph.node] == ["Cast", "Exp", "Mul"]
        assert cast_report.exported_outputs[0].dtype == "float64"


@pytest.mark.parametrize("shape", [(), (0,), (2, 0)])
def test_exp_keeps_scalar_and_empty_static_shapes(shape: tuple[int, ...]) -> None:
    _, proto, report, _ = _artifact(lambda x: jnp.exp(x), _specs([shape]))
    assert [node.op_type for node in proto.graph.node] == ["Exp"]
    assert report.exported_outputs[0].shape == shape


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("kind", ["same-B", "left-1", "right-1"])
@pytest.mark.parametrize("opset", [23, 26])
def test_one_exp_mul_artifact_runs_at_two_batch_sizes_with_equal_tails(
    rank: int, kind: str, opset: int
) -> None:
    tail = (3,) if rank == 2 else (2, 3)
    shapes = [("B", *tail), ("B", *tail)]
    if kind != "same-B":
        shapes[0 if kind == "left-1" else 1] = (1, *tail)
    feeds = []
    for batch in (2, 5):
        concrete = [
            tuple(batch if dim == "B" else dim for dim in shape) for shape in shapes
        ]
        feeds.append(
            (
                np.linspace(
                    -0.5, 0.5, int(np.prod(concrete[0])), dtype=np.float32
                ).reshape(concrete[0]),
                np.full(concrete[1], 2, np.float32),
            )
        )

    def model(x: Any, scale: Any) -> Any:
        return jnp.exp(x) * scale

    _, proto, report, references = _artifact(
        model, _specs(shapes), feeds=tuple(feeds), opset=opset
    )
    assert [node.op_type for node in proto.graph.node] == ["Exp", "Mul"]
    assert report.exported_outputs[0].shape == ("B", *tail)
    assert all(
        expected[0][0].shape == (b, *tail)
        for b, expected in zip((2, 5), references, strict=True)
    )
    _accept(proto, feeds, references, _FINITE["float32"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_finite_exp_and_static_broadcast_mul_meet_frozen_criteria(dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = (
            np.asarray([[-0.5, 0, 0.5], [0.25, -0.25, 1]], dtype),
            np.asarray([1, 2, 4], dtype),
        )

        def model(x: Any, scale: Any) -> Any:
            return jnp.exp(x) * scale

        _, proto, _, references = _artifact(
            model, _specs([v.shape for v in values], dtype), feeds=(values,)
        )
        native = _body(jax.make_jaxpr(model)(*_specs([v.shape for v in values], dtype)))
        broadcasts = [e for e in native.eqns if e.primitive.name == "broadcast_in_dim"]
        assert len(broadcasts) == 1 and broadcasts[0].params["shape"] == (1, 3)
        names = [node.op_type for node in proto.graph.node]
        assert "Cast" not in names and names.count("Expand") == len(broadcasts)
        _accept(proto, (values,), references, _FINITE[np.dtype(dtype).name])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_exact_exp_specials_and_sign_sensitive_mul_controls(dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = (
            np.asarray([-0.0, 0.0, -np.inf, np.inf], dtype),
            np.asarray([-2, 2, -2, 2], dtype),
        )

        def model(x: Any, scale: Any) -> Any:
            product = jnp.exp(x) * scale
            return jnp.exp(x), product, np.asarray(1, dtype) / product

        _, proto, _, references = _artifact(
            model, _specs([(4,), (4,)], dtype), feeds=(values,)
        )
        assert (
            references[0][0][0].tobytes()
            == np.asarray([1, 1, 0, np.inf], dtype).tobytes()
        )
        assert (
            references[0][0][1].tobytes()
            == np.asarray([-2, 2, -0.0, np.inf], dtype).tobytes()
        )
        _accept(proto, (values,), references, _BITWISE)


def test_captured_nan_zero_bytes_and_nan_class_are_separate_promises() -> None:
    captured = np.asarray([0x7FC12345, 0x80000000, 0], np.uint32).view(np.float32)
    original = captured.tobytes()

    def model(x: Any) -> Any:
        return jnp.exp(x), captured

    values = (np.asarray([np.nan, -0.0, 0.0], np.float32),)
    _, proto, _, references = _artifact(model, _specs([(3,)]), feeds=(values,))
    arrays = [onnx.numpy_helper.to_array(t) for t in proto.graph.initializer]
    assert any(array.tobytes() == original for array in arrays)
    assert captured.tobytes() == original
    _accept(proto, (values,), references, _NAN_CLASS)
    assert captured.tobytes() == original


def test_bfloat16_specials_are_exact_source_and_schema_controls() -> None:
    values = (np.asarray([-0.0, 0.0, -np.inf, np.inf], jnp.bfloat16),)
    _, proto, report, references = _artifact(
        lambda x: jnp.exp(x), _specs([(4,)], jnp.bfloat16), feeds=(values,)
    )
    expected = np.asarray([1, 1, 0, np.inf], jnp.bfloat16)
    assert all(ref[0].tobytes() == expected.tobytes() for ref in references[0])
    assert report.exported_outputs[0].dtype == "bfloat16"
    assert [node.op_type for node in proto.graph.node] == ["Exp"]
    assert (
        next(c.status for c in report.checks if c.category == "runtime-execution")
        is CheckStatus.UNVERIFIED
    )


@pytest.mark.parametrize("transform", ["vmap", "grad"])
def test_native_exp_transforms_reach_actual_audited_artifacts(transform: str) -> None:
    def model(x: Any) -> Any:
        if transform == "vmap":
            return jax.vmap(jnp.exp)(x)
        return jax.grad(lambda y: jnp.sum(jnp.exp(y)))(x)

    values = (np.asarray([[-0.5, 0], [0.5, 1]], np.float32),)
    _, proto, _, references = _artifact(model, _specs([(2, 2)]), feeds=(values,))
    assert sum(node.op_type == "Exp" for node in proto.graph.node) == 1
    assert not {"Cast", "Expand"} & {node.op_type for node in proto.graph.node}
    _accept(proto, (values,), references, _FINITE["float32"])


@pytest.mark.parametrize("nested", [False, True])
def test_named_composition_keeps_actual_formals_aliases_captures_and_cache_state(
    nested: bool,
) -> None:
    captured = np.asarray([[2, -2, 4]], np.float32)
    frozen = captured.tobytes()

    def model(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            result = (_nested if nested else _named)(x, captured)
            return result, x, captured

    specs = _specs([("B", 3)])
    feeds = tuple((np.zeros((batch, 3), np.float32),) for batch in (2, 5))
    for phase in ("native", "legacy", "source", "native", "source"):
        before = _globals(), _settings()
        if phase == "native":
            for values in feeds:
                _reference(model, values)
                _reference(jax.jit(model), values)
        elif phase == "legacy":
            try:
                to_onnx(model, specs, return_mode="ir")
            finally:
                for name, value in before[0].items():
                    jax.config.update(name, value)
        else:
            artifact, proto, report, references = _artifact(
                model, specs, feeds=feeds, mode="ir", named=True
            )
            assert isinstance(artifact, ir.Model)
            child = next(
                f.graph
                for f in artifact.functions.values()
                if any(node.op_type == "Exp" for node in f.graph)
            )
            assert len(child.inputs) == 2 and not child.initializers
            assert normalized_shape_dimensions(child.inputs[0].shape) == ("B", 3)
            assert normalized_shape_dimensions(child.inputs[1].shape) == (1, 3)
            assert normalized_shape_dimensions(child.outputs[0].shape) == ("B", 3)
            assert all(
                value.dtype is ir.DataType.FLOAT
                for value in (*child.inputs, *child.outputs)
            )
            assert [node.op_type for node in child] == ["Exp", "Mul"]
            assert all(node.graph is child for node in child)
            assert all(
                value.producer().graph is child
                for node in child
                for value in node.outputs
            )
            assert report.exported_outputs[1].shape == ("B", 3)
            assert report.exported_outputs[2].shape == (1, 3)
            assert any(
                onnx.numpy_helper.to_array(t).tobytes() == frozen
                for t in proto.graph.initializer
            )
            _accept(proto, feeds, references, _BITWISE)
        assert (_globals(), _settings()) == before
        assert captured.tobytes() == frozen


@pytest.mark.parametrize("primitive", ["exp", "mul"])
@pytest.mark.parametrize(
    "defect",
    [
        "arity",
        "outputs",
        "dtype",
        "shape",
        "weak",
        "scope",
        "unknown",
        "placement",
        "parameter",
        "effect",
    ],
)
def test_typed_pointwise_corruption_is_rejected_before_lowering(
    primitive: str, defect: str
) -> None:
    specs = _specs([("B", 3)] * (1 if primitive == "exp" else 2))
    fn = jax.lax.exp if primitive == "exp" else jax.lax.mul
    native = _body(jax.make_jaxpr(fn)(*specs))
    original = native.eqns[0]
    inputs = [SimpleNamespace(aval=copy(v.aval)) for v in original.invars]
    formals = inputs.copy()
    output = SimpleNamespace(aval=copy(original.outvars[0].aval))
    outputs, params, effects = [output], dict(original.params), ()
    if defect == "arity":
        inputs = []
    elif defect == "outputs":
        outputs = [output, output]
    elif defect == "dtype":
        output.aval = jax.ShapeDtypeStruct(specs[0].shape, np.float16)
    elif defect == "shape":
        output.aval = jax.ShapeDtypeStruct((specs[0].shape[0], 4), np.float32)
    elif defect == "weak":
        output.aval = jax.ShapeDtypeStruct(specs[0].shape, np.float32, weak_type=True)
    elif defect == "scope":
        (foreign,) = export.symbolic_shape("B")
        output.aval = jax.ShapeDtypeStruct((foreign, 3), np.float32)
    elif defect == "unknown":
        params["unreviewed"] = None
    elif defect == "placement":
        params["out_sharding"] = object()
    elif defect == "parameter":
        params["accuracy" if primitive == "exp" else "out_dtype"] = np.float32
    else:
        effects = ("effect",)
    equation = SimpleNamespace(
        primitive=original.primitive,
        invars=inputs,
        outvars=outputs,
        params=params,
        effects=effects,
    )
    graph = SimpleNamespace(
        invars=formals, constvars=(), outvars=outputs, eqns=[equation], effects=effects
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == (
        "lowering-defect"
        if defect in {"arity", "outputs", "dtype", "shape", "weak"}
        else "profile-exclusion"
    )


@pytest.mark.parametrize(
    "kind",
    ["scalar", "tail", "rank", "affine", "constrained", "foreign", "other-symbolic-op"],
)
def test_symbolic_pointwise_admission_does_not_expand_other_shape_families(
    kind: str,
) -> None:
    (batch,) = export.symbolic_shape("B")
    left = (batch, 3)
    right = (batch, 3)
    if kind == "scalar":
        right = ()
    elif kind == "tail":
        right = (batch, 1)
    elif kind == "rank":
        right = (1, batch, 3)
    elif kind == "affine":
        left = right = (batch + 1, 3)
    elif kind == "constrained":
        (constrained,) = export.symbolic_shape("B", constraints=("B >= 2",))
        left = right = (constrained, 3)
    elif kind == "foreign":
        (foreign,) = export.symbolic_shape("B")
        right = (foreign, 3)
    specs = [
        jax.ShapeDtypeStruct(left, np.float32),
        jax.ShapeDtypeStruct(right, np.float32),
    ]
    inputs = [SimpleNamespace(aval=spec) for spec in specs]
    output = SimpleNamespace(aval=specs[0])
    name = "neg" if kind == "other-symbolic-op" else "mul"
    eqn = SimpleNamespace(
        primitive=SimpleNamespace(name=name),
        invars=inputs[:1] if name == "neg" else inputs,
        outvars=[output],
        params={},
        effects=(),
    )
    graph = SimpleNamespace(
        invars=inputs, constvars=(), outvars=[output], eqns=[eqn], effects=()
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == "profile-exclusion"


@pytest.mark.parametrize("kind", ["B1-B5", "static-1", "reference-B", "reference-tail"])
def test_symbolic_samples_reject_bad_bindings_before_target_session(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shapes = [("B", 3), (1, 3) if kind == "static-1" else ("B", 3)]
    _, proto, _, _ = _artifact(lambda x, scale: jnp.exp(x) * scale, _specs(shapes))
    values = [
        np.zeros((5, 3), np.float32),
        np.ones((1 if kind == "static-1" else 5, 3), np.float32),
    ]
    expected = np.ones((5, 3), np.float32)
    if kind == "B1-B5":
        values[0] = values[0][:1]
    elif kind == "static-1":
        values[1] = np.ones((5, 3), np.float32)
    elif kind == "reference-B":
        expected = expected[:2]
    else:
        expected = expected[:, :2]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid source samples must fail before a target session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(
        proto,
        samples=[SourceSample(values, [expected], _BITWISE)],
        optimization="disabled",
    )
    assert result.status is CheckStatus.FAILED
    assert any(
        check.category == "sample-input" and check.status is CheckStatus.FAILED
        for sample in result.samples
        for check in sample.checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def test_strict_target_mismatch_remains_failed_and_raising() -> None:
    values = (np.asarray([0], np.float32),)
    _, proto, _, _ = _artifact(lambda x: jnp.exp(x), _specs([(1,)]))
    result = validate_source_samples(
        proto,
        samples=[SourceSample(values, [np.asarray([2], np.float32)], _BITWISE)],
        optimization="disabled",
    )
    assert result.status is CheckStatus.FAILED
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def test_failed_local_scope_restores_all_five_hidden_and_effective_settings() -> None:
    def unsupported(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            return jnp.sin(jnp.exp(x))

    before = _globals(), _settings()
    with pytest.raises(SourcePrecisionError) as caught:
        _artifact(unsupported, _specs([("B", 3)]))
    assert caught.value.category == "profile-exclusion"
    assert (_globals(), _settings()) == before
