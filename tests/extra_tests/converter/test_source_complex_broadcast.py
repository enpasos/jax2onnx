# tests/extra_tests/converter/test_source_complex_broadcast.py

"""Static SOURCE complex broadcasts preserve declared channels and payloads."""

import json
from pathlib import Path
import traceback
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
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, ShapedArray, Var
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    _validate_complex_broadcast_in_dim,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax.broadcast_in_dim import BroadcastInDimPlugin
from tests.extra_tests.converter.test_source_complex_add import _physical, _signature


_CONFIGURATIONS = [("complex64", False), ("complex64", True), ("complex128", True)]


@onnx_function
class _ComplexBroadcastCapture:
    def __init__(self, capture: np.ndarray) -> None:
        self.capture = capture

    def __call__(self, x: Any) -> Any:
        return x, jnp.broadcast_to(self.capture, (2, 3))


def _payload(dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.empty(shape, dtype=dtype)
    # C128 carries independently declared bits that FP32 cannot retain.
    step = 2.0 ** (-20 if dtype == "complex64" else -45)
    array.real = np.resize([1.25 + step, -0.0, np.inf], array.size).reshape(shape)
    array.imag = np.resize([-2.5 - step, 0.0, np.nan], array.size).reshape(shape)
    return array


def _assert_equal(actual: Any, expected: Any) -> None:
    left, right = _physical(actual), _physical(expected)
    assert left.dtype == right.dtype and left.shape == right.shape
    assert np.array_equal(np.isnan(left), np.isnan(right))
    mask = ~np.isnan(right)
    assert left[mask].tobytes() == right[mask].tobytes()


def _export_original(
    original: Any,
    values: tuple[Any, ...],
    expected: tuple[np.ndarray, ...],
    directory: Path,
    opset: int,
) -> onnx.ModelProto:
    frozen = tuple(np.asarray(value).tobytes() for value in values)
    # Pure copying has an independent NumPy indexing/broadcast oracle. No
    # complex multiplication or component arithmetic supplies this reference.
    np.savez(
        directory / "expected-before-target.npz",
        **{str(i): value for i, value in enumerate(expected)},
    )
    np.savez(
        directory / "original-inputs.npz",
        **{str(i): np.asarray(value) for i, value in enumerate(values)},
    )
    compiled = jax.jit(original)
    native: dict[str, tuple[np.ndarray, ...]] = {}
    errors: dict[str, str] = {}
    signatures: dict[str, Any] = {}
    for role in ("eager", "jit-first", "jit-warm"):
        try:
            result = original(*values) if role == "eager" else compiled(*values)
            jax.block_until_ready(result)
            leaves, tree = jax.tree_util.tree_flatten(result)
            native[role] = tuple(np.asarray(value).copy() for value in leaves)
            np.savez(
                directory / f"native-{role}.npz",
                **{str(i): value for i, value in enumerate(native[role])},
            )
            signatures[role] = {
                "tree": str(tree),
                "leaves": [
                    {
                        "dtype": np.dtype(value.dtype).str,
                        "shape": tuple(value.shape),
                        "weak_type": bool(jax.typeof(value).weak_type),
                    }
                    for value in leaves
                ],
            }
        except Exception as exc:
            errors[role] = f"{type(exc).__name__}: {exc}"
            (directory / f"native-{role}-error.txt").write_text(traceback.format_exc())
    specs = [
        jax.ShapeDtypeStruct(
            value.shape, value.dtype, weak_type=jax.typeof(value).weak_type
        )
        for value in values
    ]
    trace = None
    try:
        trace = jax.make_jaxpr(original)(*specs)
        (directory / "native-trace.txt").write_text(str(trace))
    except Exception as exc:
        errors["trace"] = f"{type(exc).__name__}: {exc}"
        (directory / "native-trace-error.txt").write_text(traceback.format_exc())
    (directory / "native-recording.json").write_text(
        json.dumps(
            {
                "errors": errors,
                "signatures": signatures,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "jax_version": jax.__version__,
                "ort_version": ort.__version__,
                "broadcast_equations": [
                    {
                        "dimensions": list(equation.params["broadcast_dimensions"]),
                        "inputs": [list(var.aval.shape) for var in equation.invars],
                        "outputs": [list(var.aval.shape) for var in equation.outvars],
                    }
                    for equation in (
                        _broadcast_equations(trace) if trace is not None else []
                    )
                ],
            },
            indent=2,
        )
    )
    # Save all original roles and the actual native trace before the first
    # independent-oracle assertion, so one mismatch cannot hide later roles.
    assert not errors, errors
    assert trace is not None
    for role in ("eager", "jit-first", "jit-warm"):
        assert len(native[role]) == len(expected)
        for actual, wanted in zip(native[role], expected, strict=True):
            _assert_equal(actual, wanted)
    validate_source_jaxpr(trace, opset)
    proto = ui._to_onnx(
        original, specs, opset=opset, precision_policy=PrecisionPolicy.SOURCE
    )
    path = directory / "source.onnx"
    path.write_bytes(proto.SerializeToString())
    proto = onnx.load(path)
    onnx.checker.check_model(proto, full_check=True)
    for ports, arrays in ((proto.graph.input, values), (proto.graph.output, expected)):
        for port, value in zip(ports, arrays, strict=True):
            physical = _physical(value)
            assert _signature(port) == (
                onnx.helper.np_dtype_to_tensor_dtype(physical.dtype),
                physical.shape,
            )
    for name, mode in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = mode
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        assert session.get_providers() == ["CPUExecutionProvider"]
        outputs = session.run(
            None,
            {
                port.name: _physical(value)
                for port, value in zip(session.get_inputs(), values, strict=True)
            },
        )
        np.savez(
            directory / f"target-{name}.npz",
            **{str(i): value for i, value in enumerate(outputs)},
        )
        for actual, wanted in zip(outputs, expected, strict=True):
            _assert_equal(actual, wanted)
    assert tuple(np.asarray(value).tobytes() for value in values) == frozen
    return proto


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize(
    "kind",
    [
        "scalar",
        "vector",
        "singleton",
        "mapped",
        "empty",
        "permuted",
        "square-permuted",
        "transpose",
    ],
)
def test_original_static_complex_broadcast(
    dtype: str, x64: bool, opset: int, kind: str, tmp_path: Path
) -> None:
    if kind in {"permuted", "square-permuted", "transpose"}:
        shape = (2, 2) if kind == "square-permuted" else (2, 3)
        value = _payload(dtype, shape)
        target = (
            (3, 2, 4)
            if kind == "permuted"
            else (2, 2) if kind == "square-permuted" else (3, 2)
        )
        assert value.tobytes() != value.T.tobytes()

        def original(x: Any) -> Any:
            return jax.lax.broadcast_in_dim(x, target, (1, 0))

        if kind == "square-permuted":
            # The official native equal-shape forwarding rule drops this public
            # equation. Retained eager/first/warm evidence confirms identity;
            # the original callable and its (1,0) parameter remain unchanged.
            expected = value.copy()
        elif kind == "permuted":
            expected = np.broadcast_to(value.T[:, :, None], target)
        else:
            expected = value.T
    elif kind == "mapped":
        value = _payload(dtype, (2,))

        def original(x: Any) -> Any:
            return jax.lax.broadcast_in_dim(x, (3, 2, 4), (1,))

        expected = np.broadcast_to(value.reshape((1, 2, 1)), (3, 2, 4))
    else:
        shape = {"scalar": (), "vector": (3,), "singleton": (1, 3), "empty": (1, 3)}[
            kind
        ]
        target = (0, 3) if kind == "empty" else (2, 3)
        value = _payload(dtype, shape)

        def original(x: Any) -> Any:
            return jnp.broadcast_to(x, target)

        expected = np.broadcast_to(value, target)
    before = bool(jax.config.jax_enable_x64)
    with jax.enable_x64(x64):
        _export_original(original, (value,), (expected,), tmp_path, opset)
    recording = json.loads((tmp_path / "native-recording.json").read_text())
    if kind == "transpose":
        assert recording["broadcast_equations"] == [
            {
                "dimensions": [1, 0],
                "inputs": [[2, 3]],
                "outputs": [[3, 2]],
            }
        ]
    elif kind == "square-permuted":
        assert recording["broadcast_equations"] == []
    assert bool(jax.config.jax_enable_x64) is before


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("named", [False, True])
def test_original_noncontiguous_capture_and_real_pair_sibling(
    dtype: str, x64: bool, named: bool, tmp_path: Path
) -> None:
    capture = _payload(dtype, (6,))[::2]
    assert not capture.flags.c_contiguous
    frozen = capture.tobytes()
    capture_strides = capture.strides

    def original(x: Any) -> Any:
        return x, jnp.broadcast_to(capture, (2, 3))

    owner = _ComplexBroadcastCapture(capture) if named else None
    fn = owner if owner is not None else original
    if owner is not None:
        assert owner.capture is capture and owner.capture.strides == capture_strides
    ordinary = np.array([1.25, -0.0], dtype=np.float32)
    with jax.enable_x64(x64):
        proto = _export_original(
            fn, (ordinary,), (ordinary, np.broadcast_to(capture, (2, 3))), tmp_path, 23
        )
    assert bool(proto.functions) is named
    assert capture.tobytes() == frozen
    assert capture.strides == capture_strides
    if owner is not None:
        assert owner.capture is capture and owner.capture.strides == capture_strides


def _broadcast_equations(trace: Any) -> list[Any]:
    pending = [trace]
    result = []
    while pending:
        current = pending.pop()
        body = current.jaxpr if isinstance(current, ClosedJaxpr) else current
        for equation in body.eqns:
            if equation.primitive is jax.lax.broadcast_in_dim_p:
                result.append(equation)
            pending.extend(
                value
                for value in equation.params.values()
                if isinstance(value, (Jaxpr, ClosedJaxpr))
            )
    return result


@pytest.mark.parametrize("weak", [False, True])
def test_actual_native_complex_broadcast_weakness(weak: bool, tmp_path: Path) -> None:
    with jax.enable_x64(False):
        value = (
            jnp.asarray(1.25 + 2.5j)
            if weak
            else jnp.asarray(np.asarray(1.25 + 2.5j, dtype=np.complex64))
        )
        assert jax.typeof(value).weak_type is weak

        def original(x: Any) -> Any:
            return jnp.broadcast_to(x, (2, 3))

        trace = jax.make_jaxpr(original)(value)
        assert trace.jaxpr.outvars[0].aval.weak_type is weak
        _export_original(
            original,
            (value,),
            (np.broadcast_to(np.asarray(value), (2, 3)),),
            tmp_path,
            23,
        )


@pytest.mark.parametrize("captured", [False, True])
def test_original_python_mul_rank_promotion_keeps_its_broadcast(captured: bool) -> None:
    capture = _payload("complex64", (6,))[::2]
    if captured:

        def original(x: Any) -> Any:
            return x * capture

        specs = [jax.ShapeDtypeStruct((2, 3), np.complex64)]
    else:

        def original(x: Any, y: Any) -> Any:
            return x * y

        specs = [
            jax.ShapeDtypeStruct((3,), np.complex64),
            jax.ShapeDtypeStruct((2, 3), np.complex64),
        ]
    frozen = capture.tobytes()
    trace = jax.make_jaxpr(original)(*specs)
    equations = _broadcast_equations(trace)
    assert equations
    for equation in equations:
        site = _Site("original Python Mul rank promotion", 23, "broadcast_in_dim")
        _validate_complex_broadcast_in_dim(
            equation,
            [_tensor(v, site) for v in equation.invars],
            [_tensor(v, site) for v in equation.outvars],
            site,
        )
    assert capture.tobytes() == frozen
    # These are broadcast-admission controls. The original multiplication is
    # preserved and still needs its independently reviewed numerical algorithm.


@pytest.mark.parametrize(
    "defect",
    [
        "identity",
        "effects",
        "unknown",
        "placement",
        "shape",
        "mapping",
        "width",
        "weakness",
    ],
)
def test_complex_broadcast_rejects_forged_contract(defect: str) -> None:
    equation = jax.make_jaxpr(lambda x: jax.lax.broadcast_in_dim(x, (2, 3), (1,)))(
        jax.ShapeDtypeStruct((3,), np.complex64)
    ).jaxpr.eqns[0]
    changed = SimpleNamespace(
        primitive=equation.primitive,
        effects=equation.effects,
        params=dict(equation.params),
        invars=list(equation.invars),
        outvars=list(equation.outvars),
    )
    if defect == "identity":
        changed.primitive = SimpleNamespace(name="broadcast_in_dim")
    elif defect == "effects":
        changed.effects = {"unexpected"}
    elif defect == "unknown":
        changed.params["unknown"] = None
    elif defect == "placement":
        changed.params["sharding"] = object()
    elif defect == "shape":
        changed.params["shape"] = (2, 4)
    elif defect == "mapping":
        changed.params["broadcast_dimensions"] = (0,)
    elif defect == "width":
        changed.outvars[0] = Var(ShapedArray((2, 3), np.float32))
    else:
        changed.outvars[0] = Var(ShapedArray((2, 3), np.complex64, weak_type=True))
    site = _Site("complex broadcast rejection", 23, "broadcast_in_dim")
    with pytest.raises(SourcePrecisionError):
        _validate_complex_broadcast_in_dim(
            changed,
            [_tensor(v, site) for v in changed.invars],
            [_tensor(v, site) for v in changed.outvars],
            site,
        )


@pytest.mark.parametrize("defect", ["width", "missing-pair", "extra-pair", "unknown"])
def test_complex_broadcast_rejects_physical_drift_before_emission(defect: str) -> None:
    equation = jax.make_jaxpr(lambda x: jax.lax.broadcast_in_dim(x, (2, 3), (1,)))(
        jax.ShapeDtypeStruct((3,), np.complex64)
    ).jaxpr.eqns[0]
    value = ir.val("source", dtype=ir.DataType.FLOAT, shape=(3, 2))
    if defect == "width":
        value.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        value.shape = {
            "missing-pair": ir.Shape((3,)),
            "extra-pair": ir.Shape((3, 2, 2)),
            "unknown": None,
        }[defect]
    before = value.type, value.shape

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid pair reached allocation or emission")

    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=lambda var: value,
        fresh_name=forbidden,
        allocate_value_for_var=forbidden,
        bind_value_for_var=forbidden,
    )
    with pytest.raises(TypeError, match="complex broadcast physical storage"):
        BroadcastInDimPlugin().lower(context, equation)
    assert (value.type, value.shape) == before
