# tests/extra_tests/converter/test_source_complex_add.py

"""Native complex Add, declared pair storage and finite exact dyadic witnesses."""

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import ShapedArray, Var
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    _validate_complex_add,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.plugins.jax.lax.add import lower_add
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    _array_evidence,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.test_source_predicate_families import (
    _globals,
    _restore_context as _restore_context,
    _settings,
)


_CONFIGURATIONS = [("complex64", False), ("complex64", True), ("complex128", True)]
_BITWISE = SampleCriteria("bitwise")
_SHAPES = [
    ((), ()),
    ((3,), (3,)),
    ((2,), (2,)),
    ((0, 3), (1, 3)),
    ((2, 1), (1, 3)),
    ((1, 3), (2, 1)),
    ((), (2, 3)),
    ((2, 3), ()),
]


def _physical(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "c":
        return array.copy(order="C")
    base = np.float32 if array.dtype == np.dtype("complex64") else np.float64
    return np.ascontiguousarray(array).reshape(-1).view(base).reshape((*array.shape, 2))


def _payload(dtype: str, shape: tuple[int, ...], *, right: bool = False) -> np.ndarray:
    # Both component sums are exact dyadics. C128 carries bits FP32 cannot store.
    step = 2.0 ** (-10 if dtype == "complex64" else -35)
    real = (
        [-0.5 + step, 4 + step, -1.25 - step]
        if right
        else [1 + step, -2 - step, 3.25 + step]
    )
    imag = (
        [-1.5 + step, 2.25 + step, 0.5 + step]
        if right
        else [4.5 + step, -6.25 - step, 8 + step]
    )
    value = np.empty(shape, dtype=dtype)
    value.real = np.resize(real, value.size).reshape(shape)
    value.imag = np.resize(imag, value.size).reshape(shape)
    return value


def _dyadic_add(left: Any, right: Any) -> np.ndarray:
    """Freeze exact component sums independently of native/target observations."""
    a, b = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    assert a.dtype == b.dtype and a.dtype.kind == "c"
    x, y = _physical(a), _physical(b)
    result = np.empty(x.shape, dtype=x.dtype)
    for index in np.ndindex(x.shape):
        total = Fraction(float(x[index])) + Fraction(float(y[index]))
        result[index] = (
            -0.0
            if not total and np.signbit(x[index]) and np.signbit(y[index])
            else float(total)
        )
        assert Fraction(float(result[index])) == total
    return result


def _constant(value: np.ndarray) -> tuple[Any, ...]:
    return "constant", value.dtype.str, value.shape, value.tobytes(order="C")


def _signature(value: onnx.ValueInfoProto) -> tuple[Any, ...]:
    tensor = value.type.tensor_type
    assert tensor.HasField("shape")
    assert all(dim.HasField("dim_value") for dim in tensor.shape.dim)
    return tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim)


def _artifact_contract(
    proto: onnx.ModelProto,
    inputs: tuple[np.ndarray, ...],
    outputs: tuple[np.ndarray, ...],
    roots: tuple[Any, ...],
) -> None:
    """Follow actual SSA roots through Identity and owned Function interfaces."""
    for ports, values in ((proto.graph.input, inputs), (proto.graph.output, outputs)):
        assert len(ports) == len(values)
        for port, value in zip(ports, values, strict=True):
            assert _signature(port) == (
                onnx.helper.np_dtype_to_tensor_dtype(value.dtype),
                value.shape,
            )
    functions = {(f.domain, f.name, f.overload): f for f in proto.functions}
    assert len(functions) == len(proto.functions)
    additions = []

    def scope(graph: Any, arguments: list[Any], *, function: bool = False) -> list[Any]:
        names = list(graph.input) if function else [v.name for v in graph.input]
        assert len(names) == len(arguments)
        environment = dict(zip(names, arguments, strict=True))
        infos = {v.name: v for v in graph.value_info}
        for value in getattr(graph, "initializer", ()):
            array = onnx.numpy_helper.to_array(value)
            assert value.name not in environment
            environment[value.name] = (_constant(array), array.dtype, array.shape)
        for node in graph.node:
            assert all(name not in environment for name in node.output)
            values = [environment[name] for name in node.input]
            key = node.domain, node.op_type, node.overload
            if key in functions:
                assert not node.attribute
                result = scope(functions[key], values, function=True)
            else:
                assert node.domain == "" and len(node.output) == 1
                if node.op_type == "Constant":
                    assert not values and len(node.attribute) == 1
                    assert node.attribute[0].name == "value"
                    array = onnx.numpy_helper.to_array(node.attribute[0].t)
                    result = [(_constant(array), array.dtype, array.shape)]
                elif node.op_type == "Identity":
                    assert len(values) == 1 and not node.attribute
                    result = values
                else:
                    assert node.op_type == "Add" and len(values) == 2
                    assert not node.attribute and values[0][1] == values[1][1]
                    shape = np.broadcast_shapes(values[0][2], values[1][2])
                    token = "add", values[0][0], values[1][0]
                    additions.append(token)
                    result = [(token, values[0][1], shape)]
            assert len(result) == len(node.output)
            for name, (_, dtype, shape) in zip(node.output, result, strict=True):
                if name in infos:
                    assert _signature(infos[name]) == (
                        onnx.helper.np_dtype_to_tensor_dtype(dtype),
                        shape,
                    )
            environment.update(zip(node.output, result, strict=True))
        names = list(graph.output) if function else [v.name for v in graph.output]
        return [environment[name] for name in names]

    actual = scope(
        proto.graph,
        [
            (("input", index), value.dtype, value.shape)
            for index, value in enumerate(inputs)
        ],
    )
    assert tuple(item[0] for item in actual) == roots
    assert tuple((item[1], item[2]) for item in actual) == tuple(
        (value.dtype, value.shape) for value in outputs
    )

    def required(token: Any) -> set[Any]:
        return (
            {token} | required(token[1]) | required(token[2])
            if token[0] == "add"
            else set()
        )

    assert set(additions) == set().union(*(required(root) for root in roots))


def _observe(
    fn: Any,
    values: tuple[Any, ...],
    expected: tuple[np.ndarray, ...],
    roots: tuple[Any, ...],
    directory: Path,
    *,
    opset: int,
    callback: Any = None,
) -> onnx.ModelProto:
    directory.mkdir(parents=True, exist_ok=True)
    # The predicate and its exact expected payload precede every target session.
    (directory / "criterion.json").write_text(
        json.dumps(asdict(_BITWISE), sort_keys=True)
    )
    np.savez(
        directory / "expected-before-target.npz",
        **{f"out{i}": v for i, v in enumerate(expected)},
    )
    snapshots = tuple(np.asarray(value).tobytes() for value in values)
    physical_inputs = tuple(_physical(value) for value in values)
    np.savez(
        directory / "original-inputs.npz", **{f"in{i}": v for i, v in enumerate(values)}
    )
    before = _globals(), _settings(), active_precision_policy()
    errors: dict[str, str] = {}
    native: dict[str, tuple[np.ndarray, ...]] = {}

    def fresh(*arguments: Any) -> Any:
        return fn(*arguments)

    for role in ("eager", "jit"):
        try:
            result = fn(*values) if role == "eager" else jax.jit(fresh)(*values)
            native[role] = tuple(np.asarray(v).copy() for v in jax.tree.leaves(result))
        except Exception as exc:
            errors[role] = f"{type(exc).__name__}: {exc}"
            native[role] = ()
        np.savez(
            directory / f"native-{role}.npz",
            **{f"out{i}": v for i, v in enumerate(native[role])},
        )
    specs = [
        jax.ShapeDtypeStruct(
            jax.typeof(v).shape, jax.typeof(v).dtype, weak_type=jax.typeof(v).weak_type
        )
        for v in values
    ]
    trace = None
    try:
        trace = jax.make_jaxpr(fresh)(*specs)
        (directory / "native-trace.txt").write_text(str(trace))
    except Exception as exc:
        errors["trace"] = f"{type(exc).__name__}: {exc}"
    proto = None
    try:
        proto = ui._to_onnx(
            fn, specs, opset=opset, precision_policy=PrecisionPolicy.SOURCE
        )
        (directory / "source.onnx").write_bytes(proto.SerializeToString())
    except Exception as exc:
        errors["export"] = f"{type(exc).__name__}: {exc}"
    report = None
    if proto is not None:
        try:
            report = get_precision_report(proto)
        except Exception as exc:
            errors["export-report"] = f"{type(exc).__name__}: {exc}"
    targets: dict[str, tuple[np.ndarray, ...]] = {}
    evidence = []
    for mode in ("disabled", "all"):
        try:
            if proto is None:
                raise RuntimeError("Target unavailable: original export failed")
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.optimized_model_filepath = str(
                directory / f"{mode}-runtime-graph.onnx"
            )
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                if mode == "disabled"
                else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            session.disable_fallback()
            feeds = dict(
                zip(
                    (port.name for port in session.get_inputs()),
                    physical_inputs,
                    strict=True,
                )
            )
            targets[mode] = tuple(
                np.asarray(v).copy() for v in session.run(None, feeds)
            )
        except Exception as exc:
            errors[mode] = f"{type(exc).__name__}: {exc}"
            targets[mode] = ()
        np.savez(
            directory / f"{mode}-actual.npz",
            **{f"out{i}": v for i, v in enumerate(targets[mode])},
        )
    if proto is not None:
        for mode in ("disabled", "all"):
            for role in ("eager", "jit"):
                try:
                    result = validate_source_samples(
                        proto,
                        samples=[
                            SourceSample(
                                physical_inputs,
                                tuple(_physical(v) for v in native[role]),
                                _BITWISE,
                            )
                        ],
                        optimization=mode,
                    )
                    evidence.append((mode, role, result))
                except Exception as exc:
                    errors[f"sample-{mode}-{role}"] = f"{type(exc).__name__}: {exc}"
    callback_result = None
    if proto is not None and callback is not None:
        try:
            callback_result = callback(proto)
        except Exception as exc:
            errors["raw-callback"] = f"{type(exc).__name__}: {exc}"
    (directory / "observations-before-acceptance.json").write_text(
        json.dumps(
            {
                "errors": errors,
                "raw_callback": callback_result,
                "artifact_sha256": (
                    None
                    if proto is None
                    else hashlib.sha256(proto.SerializeToString()).hexdigest()
                ),
                "samples": [
                    {
                        "optimization": mode,
                        "reference": role,
                        "status": r.status,
                        "target": None if r.target is None else asdict(r.target),
                        "checks": [asdict(c) for c in r.checks],
                        "samples": [asdict(s) for s in r.samples],
                    }
                    for mode, role, r in evidence
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    # Every native role, CPU mode and sample report is retained before guards.
    for _, _, result in evidence:
        result.require_sample_acceptance()
    assert not errors, errors
    assert proto is not None and trace is not None and report is not None
    onnx.checker.check_model(proto, full_check=True)
    assert {o.version for o in proto.opset_import if o.domain == ""} == {opset}
    assert report.profile == "source-static-v1"
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert len(evidence) == 4
    for mode, role, result in evidence:
        assert (
            result.artifact_sha256
            == hashlib.sha256(proto.SerializeToString()).hexdigest()
        )
        assert result.status is CheckStatus.VALIDATED and result.target is not None
        assert result.target.providers == ("CPUExecutionProvider",)
        assert result.target.runtime_version == ort.__version__
        assert result.target.optimization == (
            "ORT_DISABLE_ALL" if mode == "disabled" else "ORT_ENABLE_ALL"
        )
        sample = result.samples[0]
        assert sample.inputs == tuple(
            _array_evidence(str(i), v) for i, v in enumerate(physical_inputs)
        )
        assert sample.references == tuple(
            _array_evidence(str(i), _physical(v)) for i, v in enumerate(native[role])
        )
        assert sample.outputs == tuple(
            _array_evidence(str(i), v) for i, v in enumerate(targets[mode])
        )
        assert sample.criteria == _BITWISE
    for returned in (*native.values(), *targets.values()):
        physical = tuple(_physical(v) for v in returned)
        assert len(physical) == len(expected)
        for actual, wanted in zip(physical, expected, strict=True):
            assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
            assert actual.tobytes() == wanted.tobytes()
    for entries, variables in (
        ([v.traced for v in report.inputs], trace.jaxpr.invars),
        (report.traced_outputs, trace.jaxpr.outvars),
    ):
        assert [(v.dtype, v.shape, v.weak_type) for v in entries] == [
            (np.dtype(v.aval.dtype).name, tuple(v.aval.shape), v.aval.weak_type)
            for v in variables
        ]
    for entry, variable in zip(report.inputs, trace.jaxpr.invars, strict=True):
        logical = (
            np.dtype(variable.aval.dtype).name,
            tuple(variable.aval.shape),
            variable.aval.weak_type,
        )
        assert all(
            (v.dtype, v.shape, v.weak_type) == logical
            for v in (entry.supplied, entry.normalized, entry.traced)
        )
        declaration = representation_for_aval(variable.aval)
        assert (entry.exported.dtype, entry.exported.shape) == (
            declaration.physical_dtype.name,
            declaration.physical_shape,
        )
    _artifact_contract(proto, physical_inputs, expected, roots)
    assert get_precision_report(proto) == report
    assert (_globals(), _settings(), active_precision_policy()) == before
    assert tuple(np.asarray(value).tobytes() for value in values) == snapshots
    if callback is not None:
        assert callback_result is True
    return proto


def assert_complex_add_source(
    fn: Any,
    values: tuple[Any, ...],
    directory: Path,
    *,
    operands: tuple[int, int] = (0, 1),
    opset: int = 23,
) -> onnx.ModelProto:
    """Also serve the retained old x+x counterpart, without importing its module."""
    a, b = operands
    return _observe(
        fn,
        values,
        (_dyadic_add(values[a], values[b]),),
        (("add", ("input", a), ("input", b)),),
        directory,
        opset=opset,
    )


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize(
    "shapes",
    _SHAPES,
    ids=[
        "scalar",
        "vector",
        "logical-two",
        "empty-broadcast",
        "broadcast-left",
        "broadcast-right",
        "scalar-array",
        "array-scalar",
    ],
)
def test_native_complex_add_keeps_declared_width_pair_axis_and_broadcast(
    dtype: str, x64: bool, opset: int, shapes: Any, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        left, right = _payload(dtype, shapes[0]), _payload(dtype, shapes[1], right=True)
        assert_complex_add_source(jax.lax.add, (left, right), tmp_path, opset=opset)


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize(
    "weak", [(False, False), (False, True), (True, False), (True, True)]
)
def test_native_complex_add_preserves_actual_operand_and_result_weakness(
    dtype: str, x64: bool, weak: tuple[bool, bool], tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        values = tuple(
            jax.lax.convert_element_type_p.bind(
                jnp.asarray(_payload(dtype, (2,), right=bool(i))),
                new_dtype=np.dtype(dtype),
                weak_type=flag,
                sharding=None,
            )
            for i, flag in enumerate(weak)
        )
        proto = assert_complex_add_source(jax.lax.add, values, tmp_path)
        report = get_precision_report(proto)
        assert [v.traced.weak_type for v in report.inputs] == list(weak)
        assert report.traced_outputs[0].weak_type is all(weak)


@onnx_function(name="SourceComplexAdd")
def _named_add(left: Any, right: Any) -> Any:
    return jax.lax.add(left, right)


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("named", [False, True], ids=["capture", "named-jit"])
def test_complex_add_capture_scopes_real_sibling_and_repeated_outputs(
    dtype: str, x64: bool, opset: int, named: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        left, capture = _payload(dtype, (2,)), _payload(dtype, (2,), right=True)
        base = np.float32 if dtype == "complex64" else np.float64
        ordinary = np.asarray([[2, 5], [7, 11]], base)
        frozen = capture.tobytes()

        def model(x: Any, real_pair: Any) -> Any:
            result = (
                jax.jit(_named_add)(x, capture) if named else jax.lax.add(x, capture)
            )
            return result, result, x, capture, real_pair + real_pair, real_pair

        total = _dyadic_add(left, capture)
        add = "add", ("input", 0), _constant(_physical(capture))
        expected = (
            total,
            total,
            _physical(left),
            _physical(capture),
            ordinary * 2,
            ordinary,
        )
        roots = (
            add,
            add,
            ("input", 0),
            _constant(_physical(capture)),
            ("add", ("input", 1), ("input", 1)),
            ("input", 1),
        )
        proto = _observe(
            model, (left, ordinary), expected, roots, tmp_path, opset=opset
        )
        assert bool(proto.functions) is named
        assert capture.tobytes() == frozen


@pytest.mark.parametrize("x64", [False, True])
def test_original_raw_complex64_add_row_keeps_recipe_callback_and_missing_criterion(
    x64: bool, tmp_path: Path
) -> None:
    metadata = source.raw_source_case("plugin::add", 2)
    original, callback = metadata["callable"], metadata["post_check_onnx_graph"]
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::add", 2)
    )[int(x64)]
    values = tuple(metadata["input_values"])
    snapshots = tuple(v.tobytes() for v in values)
    with source.source_configuration(scenario):
        expected = _dyadic_add(*values)
        _observe(
            original,
            values,
            (expected,),
            (("add", ("input", 0), ("input", 1)),),
            tmp_path,
            opset=23,
            callback=callback,
        )
    assert metadata["testcase"] == "add_complex64"
    assert metadata.get("opset_version", 23) == 23
    assert metadata["expected_output_dtypes"] == [np.float32]
    assert snapshots == (
        np.asarray([1 + 2j, 3 - 4j], np.complex64).tobytes(),
        np.asarray([-1.5 + 0.5j, 2 + 1.25j], np.complex64).tobytes(),
    )
    assert scenario.criteria is None
    assert tuple(v.tobytes() for v in values) == snapshots
    assert source.raw_source_case("plugin::add", 2)["callable"] is original
    assert source.raw_source_case("plugin::add", 2)["post_check_onnx_graph"] is callback


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "defect",
    [
        "identity",
        "parameters",
        "input-arity",
        "output-arity",
        "mixed-width",
        "result-width",
        "real-operand",
        "weakness",
        "result-shape",
        "unequal-ranks",
        "incompatible-shapes",
        "symbolic",
    ],
)
def test_complex_add_owning_validator_rejects_forged_logical_contract(
    dtype: str, defect: str
) -> None:
    with jax.enable_x64(True):
        equation = jax.make_jaxpr(jax.lax.add)(
            *[jax.ShapeDtypeStruct((2,), dtype)] * 2
        ).jaxpr.eqns[0]
        changed = SimpleNamespace(
            primitive=equation.primitive,
            params=dict(equation.params),
            invars=list(equation.invars),
            outvars=list(equation.outvars),
        )
        other = "complex128" if dtype == "complex64" else "complex64"
        if defect == "identity":
            changed.primitive = SimpleNamespace(name="add")
        elif defect == "parameters":
            changed.params["out_sharding"] = None
        elif defect == "input-arity":
            changed.invars.pop()
        elif defect == "output-arity":
            changed.outvars.append(changed.outvars[0])
        elif defect == "mixed-width":
            changed.invars[1] = Var(ShapedArray((2,), other))
        elif defect == "result-width":
            changed.outvars[0] = Var(ShapedArray((2,), other))
        elif defect == "real-operand":
            changed.invars[1] = Var(ShapedArray((2,), np.float32))
        elif defect == "weakness":
            changed.outvars[0] = Var(ShapedArray((2,), dtype, weak_type=True))
        elif defect == "result-shape":
            changed.outvars[0] = Var(ShapedArray((3,), dtype))
        elif defect == "unequal-ranks":
            changed.invars[1] = Var(ShapedArray((1, 2), dtype))
        elif defect == "incompatible-shapes":
            changed.invars[1] = Var(ShapedArray((3,), dtype))
        else:
            changed.invars[0] = SimpleNamespace(
                aval=SimpleNamespace(
                    dtype=np.dtype(dtype), shape=("B",), weak_type=False
                )
            )
        site = _Site("complex-add-negative", 23, "add")
        with pytest.raises(SourcePrecisionError):
            _validate_complex_add(
                changed,
                [_tensor(v, site) for v in changed.invars],
                [_tensor(v, site) for v in changed.outvars],
                site,
            )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize(
    "defect",
    [
        "width",
        "missing-shape",
        "missing-pair",
        "extra-pair",
        "pair-one",
        "pair-three",
        "extent",
    ],
)
def test_complex_add_rejects_physical_drift_before_allocation_or_emission(
    dtype: str, operand: int, defect: str
) -> None:
    with jax.enable_x64(True):
        equation = jax.make_jaxpr(jax.lax.add)(
            *[jax.ShapeDtypeStruct((2,), dtype)] * 2
        ).jaxpr.eqns[0]
        base = ir.DataType.FLOAT if dtype == "complex64" else ir.DataType.DOUBLE
        values = [ir.val(f"operand{i}", dtype=base, shape=(2, 2)) for i in range(2)]
        if defect == "width":
            values[operand].type = ir.TensorType(
                ir.DataType.DOUBLE if base is ir.DataType.FLOAT else ir.DataType.FLOAT
            )
        else:
            values[operand].shape = {
                "missing-shape": None,
                "missing-pair": ir.Shape((2,)),
                "extra-pair": ir.Shape((2, 2, 2)),
                "pair-one": ir.Shape((2, 1)),
                "pair-three": ir.Shape((2, 3)),
                "extent": ir.Shape((3, 2)),
            }[defect]
        snapshot = [
            (v.type, None if v.shape is None else tuple(v.shape)) for v in values
        ]
        bindings = dict(zip(map(id, equation.invars), values, strict=True))

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail(
                "Invalid complex storage reached allocation or builder emission"
            )

        context = SimpleNamespace(
            precision_policy=PrecisionPolicy.SOURCE,
            builder=SimpleNamespace(opset=23, Add=forbidden, op=forbidden),
            require_value_for_var=lambda v: bindings[id(v)],
            allocate_value_for_var=forbidden,
            bind_value_for_var=forbidden,
            fresh_name=forbidden,
        )
        with pytest.raises(TypeError, match="SOURCE complex Add physical storage"):
            lower_add(context, equation)
        assert snapshot == [
            (v.type, None if v.shape is None else tuple(v.shape)) for v in values
        ]


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_actual_complex_add_artifact_rejects_casts_wrong_roots_and_channel_corruptions(
    dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        values = (_payload(dtype, (2,)), _payload(dtype, (2,), right=True))
        proto = assert_complex_add_source(jax.lax.add, values, tmp_path)
        inputs, outputs = tuple(map(_physical, values)), (_dyadic_add(*values),)
        roots = (("add", ("input", 0), ("input", 1)),)
        original = proto.SerializeToString()
        for defect in (
            "extra-cast",
            "wrong-ssa",
            "swapped-channels",
            "duplicated-channels",
            "bypass",
        ):
            damaged = onnx.load_model_from_string(original)
            graph = damaged.graph
            addition = next(n for n in graph.node if n.op_type == "Add")
            if defect == "wrong-ssa":
                addition.input[1] = addition.input[0]
            elif defect == "bypass":
                graph.output[0].name = graph.input[0].name
            else:
                source_name, replacement = addition.input[0], "corrupt-operand"
                if defect == "extra-cast":
                    node = onnx.helper.make_node(
                        "Cast",
                        [source_name],
                        [replacement],
                        to=graph.input[0].type.tensor_type.elem_type,
                    )
                else:
                    indices = np.asarray(
                        [1, 0] if defect == "swapped-channels" else [0, 0], np.int64
                    )
                    graph.initializer.append(
                        onnx.numpy_helper.from_array(indices, name="corrupt-channels")
                    )
                    node = onnx.helper.make_node(
                        "Gather",
                        [source_name, "corrupt-channels"],
                        [replacement],
                        axis=-1,
                    )
                addition.input[0] = replacement
                graph.node.insert(0, node)
            (tmp_path / f"{defect}.onnx").write_bytes(damaged.SerializeToString())
            with pytest.raises(AssertionError):
                _artifact_contract(damaged, inputs, outputs, roots)
        assert proto.SerializeToString() == original


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("opset", [16, 18, 21, 24, 27])
def test_unselected_complex_add_opsets_remain_explicit(dtype: str, opset: int) -> None:
    with jax.enable_x64(True):
        native = jax.make_jaxpr(jax.lax.add)(*[jax.ShapeDtypeStruct((2,), dtype)] * 2)
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(native.jaxpr, opset)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_native_unequal_positive_ranks_fail_without_numpy_style_broadcasting(
    dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        values = (_payload(dtype, (2,)), _payload(dtype, (1, 2), right=True))
        specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]

        def original(left: Any, right: Any) -> Any:
            return jax.lax.add(left, right)

        def fresh(left: Any, right: Any) -> Any:
            return original(left, right)

        attempts = (
            ("eager", lambda: original(*values)),
            ("fresh-jit", lambda: jax.jit(fresh)(*values)),
            ("trace", lambda: jax.make_jaxpr(original)(*specs)),
            (
                "source",
                lambda: ui._to_onnx(
                    original, specs, opset=23, precision_policy=PrecisionPolicy.SOURCE
                ),
            ),
        )
        observations = {}
        for role, invoke in attempts:
            try:
                invoke()
                observations[role] = {"status": "returned"}
            except Exception as exc:
                observations[role] = {
                    "status": "raised",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        (tmp_path / "native-rank-failures-before-assertions.json").write_text(
            json.dumps(observations, indent=2, sort_keys=True)
        )
        assert all(item["status"] == "raised" for item in observations.values())
        assert all(
            observations[role]["type"] == "TypeError"
            for role in ("eager", "fresh-jit", "trace")
        )
        assert observations["source"]["type"] in {"TypeError", "SourcePrecisionError"}


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("operation", ["sub", "mul", "div", "neg"])
def test_other_complex_arithmetic_keeps_its_separate_source_obligation(
    dtype: str, operation: str
) -> None:
    with jax.enable_x64(True):
        function = getattr(jax.lax, operation)
        specs = [jax.ShapeDtypeStruct((2,), dtype)] * (1 if operation == "neg" else 2)
        native = jax.make_jaxpr(function)(*specs)
        with pytest.raises(SourcePrecisionError) as error:
            validate_source_jaxpr(native.jaxpr, 23)
        assert error.value.primitive == operation
