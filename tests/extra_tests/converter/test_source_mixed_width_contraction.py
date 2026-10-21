# tests/extra_tests/converter/test_source_mixed_width_contraction.py

"""Finite mixed FP32/FP64 native dots retain narrow ports and local FP64 work.

These coordinate controls add their own fixed bitwise criterion. Original
Eqx/model recipes, callbacks and absent criteria stay in their existing owners.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
from jax import export
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx._compat.jax import Var
from jax2onnx.converter.source_profile import (
    _Site,
    _tensor,
    _validate_dot,
    SourcePrecisionError,
)
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _body,
    _observe,
    _oracle,
    _restore_settings as _restore_settings,
    _save,
)

_MATRIX = (((1,), (0,)), ((), ()))


@dataclass(frozen=True)
class _Geometry:
    name: str
    left: tuple[Any, ...]
    right: tuple[Any, ...]
    dimensions: Any
    left_dtype: str
    preferred: str | None = None
    input_order: tuple[int, ...] | None = None
    output_order: tuple[int, ...] | None = None
    precision: str | None = None


_CASES = [
    _Geometry("matrix-32-64-none", (2, 3), (3, 4), _MATRIX, "float32"),
    _Geometry("matrix-64-32-none", (2, 3), (3, 4), _MATRIX, "float64"),
    _Geometry("matrix-32-64-preferred", (2, 3), (3, 4), _MATRIX, "float32", "float64"),
    _Geometry("matrix-64-32-preferred", (2, 3), (3, 4), _MATRIX, "float64", "float64"),
    _Geometry(
        "ordered-multi-contract",
        (2, 3, 4),
        (4, 3, 5),
        (((2, 1), (0, 1)), ((), ())),
        "float32",
    ),
    _Geometry("free-left-32-64", ("B", 3), (3, 4), _MATRIX, "float32"),
    _Geometry("free-left-64-32", ("B", 3), (3, 4), _MATRIX, "float64", "float64"),
    _Geometry(
        "weight-first-32-64",
        (4, 3),
        ("B", 3),
        (((1,), (1,)), ((), ())),
        "float32",
        output_order=(1, 0),
    ),
    _Geometry(
        "weight-first-64-32",
        (4, 3),
        ("B", 3),
        (((1,), (1,)), ((), ())),
        "float64",
        "float64",
        output_order=(1, 0),
    ),
    _Geometry(
        "paired-B-HIGH",
        ("B", 2, 3),
        ("B", 3, 4),
        (((2,), (1,)), ((0,), (0,))),
        "float32",
        "float64",
        precision="HIGH",
    ),
    _Geometry(
        "nonleading-intermediate-HIGHEST",
        ("B", 2, 3),
        (3, 4),
        (((2,), (0,)), ((), ())),
        "float64",
        input_order=(1, 0, 2),
        output_order=(1, 0, 2),
        precision="HIGHEST",
    ),
    _Geometry("K-zero", (2, 0), (0, 3), _MATRIX, "float32"),
    _Geometry("M-zero", (0, 3), (3, 4), _MATRIX, "float64"),
    _Geometry("N-zero", (2, 3), (3, 0), _MATRIX, "float32", "float64"),
    _Geometry("outer-signed-zero", (2,), (3,), (((), ()), ((), ())), "float64"),
]


def _shape(shape: Any, batch: Any) -> tuple[Any, ...]:
    return tuple(batch if dim == "B" else dim for dim in shape)


def _payload(dtype: str, shape: Any, side: int) -> np.ndarray:
    indices = np.arange(np.prod(shape, dtype=np.int64)).reshape(shape)
    values = ((indices * (5 if side == 0 else 3) + side) % 13 - 6) / 8
    if dtype == "float64":
        # The low bits expose accidental FP64-to-FP32 narrowing. All products
        # and short coordinate sums here remain exactly representable in FP64.
        values = values + (indices % 3) * 2.0**-35
    return values.astype(dtype)


def _signature(variables: Any) -> Any:
    return tuple(
        (
            np.dtype(v.aval.dtype).name,
            normalized_shape_dimensions(v.aval.shape),
            v.aval.weak_type,
        )
        for v in variables
    )


def _output_shape(left: Any, right: Any, dimensions: Any) -> tuple[Any, ...]:
    (lc, rc), (lb, rb) = dimensions
    return (
        *(left[a] for a in lb),
        *(left[a] for a in range(len(left)) if a not in (*lc, *lb)),
        *(right[a] for a in range(len(right)) if a not in (*rc, *rb)),
    )


def _runtime_payloads(directory: Path) -> Any:
    constructor = ort.InferenceSession
    counter = 0

    def session(*args: Any, **kwargs: Any) -> Any:
        nonlocal counter
        actual = constructor(*args, **kwargs)
        index = counter
        counter += 1
        run, run_index = actual.run, 0

        def recording_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal run_index
            outputs = run(*args, **kwargs)
            _save(directory / f"actual-session-{index}-run-{run_index}.npz", outputs)
            run_index += 1
            return outputs

        actual.run = recording_run
        return actual

    return patch.object(ort, "InferenceSession", session)


def _artifact_route(model: Any, directory: Path, *, outer: bool, empty: bool) -> None:
    # Analyze the serialized/reloaded artifact, not provisional builder values.
    payload = model.SerializeToString()
    reloaded = onnx.load_model_from_string(payload)
    onnx.checker.check_model(reloaded, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(reloaded, strict_mode=True).graph
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in (*inferred.input, *inferred.value_info, *inferred.output)
    }
    assert sorted(types[v.name] for v in inferred.input) == sorted(
        (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE)
    )
    assert types[inferred.output[0].name] == onnx.TensorProto.DOUBLE
    for source, alias in zip(inferred.input, inferred.output[1:], strict=True):
        assert types[source.name] == types[alias.name]
    casts = [node for node in inferred.node if node.op_type == "Cast"]
    arithmetic = [node for node in inferred.node if node.op_type in {"MatMul", "Mul"}]
    if empty:
        assert not casts and not arithmetic
        (zero,) = [node for node in inferred.node if node.op_type == "ConstantOfShape"]
        assert types[zero.output[0]] == onnx.TensorProto.DOUBLE
        assert any(
            a.name == "value" and a.t.data_type == onnx.TensorProto.DOUBLE
            for a in zero.attribute
        )
        return
    (cast,) = casts
    assert cast.input[0] in {v.name for v in inferred.input}
    assert types[cast.input[0]] == onnx.TensorProto.FLOAT
    assert types[cast.output[0]] == onnx.TensorProto.DOUBLE
    assert any(
        a.name == "to" and a.i == onnx.TensorProto.DOUBLE for a in cast.attribute
    )
    (operation,) = arithmetic
    assert operation.op_type == ("Mul" if outer else "MatMul")
    assert all(
        types[name] == onnx.TensorProto.DOUBLE
        for name in (*operation.input, *operation.output)
    )
    producers = {name: node for node in inferred.node for name in node.output}
    sources = []
    for name in operation.input:
        seen = set()
        while name in producers and producers[name].op_type in {
            "Transpose",
            "Reshape",
            "Identity",
        }:
            assert name not in seen and types[name] == onnx.TensorProto.DOUBLE
            seen.add(name)
            name = producers[name].input[0]
        sources.append(name)
    assert cast.output[0] in sources
    wide_input = next(
        v.name for v in inferred.input if types[v.name] == onnx.TensorProto.DOUBLE
    )
    assert wide_input in sources
    (directory / "serialized-local-cast.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "cast_input": cast.input[0],
                "cast_output": cast.output[0],
                "operation": operation.op_type,
                "operand_origins": sources,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("case", _CASES, ids=[case.name for case in _CASES])
def test_finite_mixed_native_geometries_preserve_original_ports_and_local_cast(
    case: _Geometry, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        right_dtype = "float64" if case.left_dtype == "float32" else "float32"
        dtypes = case.left_dtype, right_dtype
        (batch,) = export.symbolic_shape("B")
        specs = [
            jax.ShapeDtypeStruct(_shape(shape, batch), dtype)
            for shape, dtype in zip((case.left, case.right), dtypes, strict=True)
        ]
        preferred = None if case.preferred is None else np.dtype(case.preferred).type
        precision = (
            None
            if case.precision is None
            else getattr(jax.lax.Precision, case.precision)
        )

        def fn(left: Any, right: Any) -> Any:
            inner = (
                jax.lax.transpose(left, case.input_order) if case.input_order else left
            )
            result = jax.lax.dot_general(
                inner,
                right,
                case.dimensions,
                precision=precision,
                preferred_element_type=preferred,
            )
            if case.output_order:
                result = jax.lax.transpose(result, case.output_order)
            return result, left, right

        symbolic = "B" in (*case.left, *case.right)
        feeds, expected = [], []
        for size in (2, 5) if symbolic else (2,):
            values = tuple(
                _payload(dtype, _shape(shape, size), side)
                for side, (shape, dtype) in enumerate(
                    zip((case.left, case.right), dtypes, strict=True)
                )
            )
            if case.name == "outer-signed-zero":
                values = np.asarray([-0.0, 0.5], np.float64), np.asarray(
                    [1.0, -2.0, 0.0], np.float32
                )
            inner = (
                np.transpose(values[0], case.input_order)
                if case.input_order
                else values[0]
            )
            wanted = _oracle(inner, values[1], case.dimensions, np.float64)
            if case.output_order:
                wanted = np.transpose(wanted, case.output_order).copy()
            # lax.dot_general receives actual JAX arrays in both native roles.
            feeds.append(tuple(jax.device_put(value) for value in values))
            expected.append((wanted, *values))
        inner_shape = (
            tuple(case.left[a] for a in case.input_order)
            if case.input_order
            else case.left
        )
        raw_shape = _output_shape(inner_shape, case.right, case.dimensions)
        output_shape = (
            tuple(raw_shape[a] for a in case.output_order)
            if case.output_order
            else raw_shape
        )
        with _runtime_payloads(tmp_path):
            model, trace, _ = _observe(
                fn,
                specs,
                feeds,
                expected,
                tmp_path,
                output_shapes=[output_shape, case.left, case.right],
            )
        (dot,) = [e for e in _body(trace).eqns if e.primitive is jax.lax.dot_general_p]
        assert _signature(dot.invars) == (
            (case.left_dtype, inner_shape, False),
            (right_dtype, case.right, False),
        )
        assert _signature(dot.outvars) == (("float64", raw_shape, False),)
        assert dot.params["dimension_numbers"] == case.dimensions
        assert dot.params["preferred_element_type"] == preferred
        assert dot.params["precision"] == (
            None if precision is None else (precision, precision)
        )
        _artifact_route(
            model,
            tmp_path,
            outer=not case.dimensions[0][0],
            empty=0 in raw_shape
            or any(inner_shape[a] == 0 for a in case.dimensions[0][0]),
        )
        if case.name == "outer-signed-zero":
            assert np.signbit(expected[0][0][0, 0]) and not np.signbit(
                expected[0][0][0, 1]
            )


def test_composed_symbolic_mixed_zero_contraction_keeps_positive_zero_identity(
    tmp_path: Path,
) -> None:
    with jax.enable_x64(True):
        empty_weight = jax.device_put(np.empty((3, 0), np.float64))
        continuation = jax.device_put(np.empty((0, 4), np.float32))

        def fn(x: Any) -> Any:
            empty = jax.lax.dot_general(x, empty_weight, _MATRIX)
            result = jax.lax.dot_general(empty, continuation, _MATRIX)
            return result, empty, x

        (batch,) = export.symbolic_shape("B")
        feeds = [(jax.device_put(_payload("float32", (b, 3), 0)),) for b in (2, 5)]
        expected = [
            (np.zeros((b, 4), np.float64), np.empty((b, 0), np.float64), np.asarray(x))
            for b, (x,) in zip((2, 5), feeds, strict=True)
        ]
        with _runtime_payloads(tmp_path):
            model, trace, _ = _observe(
                fn,
                [jax.ShapeDtypeStruct((batch, 3), np.float32)],
                feeds,
                expected,
                tmp_path,
                output_shapes=[("B", 4), ("B", 0), ("B", 3)],
            )
        dots = [e for e in _body(trace).eqns if e.primitive is jax.lax.dot_general_p]
        assert len(dots) == 2
        assert [
            tuple(np.dtype(v.aval.dtype).name for v in (*e.invars, *e.outvars))
            for e in dots
        ] == [("float32", "float64", "float64"), ("float64", "float32", "float64")]
        assert sum(node.op_type == "ConstantOfShape" for node in model.graph.node) == 2
        assert not any(
            node.op_type in {"MatMul", "Mul", "Cast"} for node in model.graph.node
        )
        for wanted, _, _ in expected:
            assert not np.signbit(wanted).any()


def _equation() -> Any:
    native = jax.make_jaxpr(lambda a, b: jax.lax.dot_general(a, b, _MATRIX))(
        jax.ShapeDtypeStruct((2, 3), np.float32),
        jax.ShapeDtypeStruct((3, 4), np.float64),
    )
    (equation,) = _body(native).eqns
    return SimpleNamespace(
        primitive=equation.primitive,
        params=dict(equation.params),
        invars=list(equation.invars),
        outvars=list(equation.outvars),
        effects=equation.effects,
        ctx=equation.ctx,
    )


def _validate(eqn: Any) -> None:
    site = _Site("mixed-contraction-control", 23, "dot_general")
    _validate_dot(
        eqn,
        [_tensor(v, site) for v in eqn.invars],
        [_tensor(v, site) for v in eqn.outvars],
        site,
    )


def _physical(eqn: Any) -> tuple[Any, Any]:
    return (
        [
            ir.val(
                f"input{i}",
                dtype=ir.DataType.FLOAT if i == 0 else ir.DataType.DOUBLE,
                shape=v.aval.shape,
            )
            for i, v in enumerate(eqn.invars)
        ],
        ir.val("output", dtype=ir.DataType.DOUBLE, shape=eqn.outvars[0].aval.shape),
    )


def _lower(eqn: Any, values: Any, output: Any, *, forbidden: bool = False) -> Any:
    emissions, allocations, bindings = [], [], []

    def op(name: str, inputs: Any, attributes: Any = None, *, output: Any) -> Any:
        assert not forbidden, "invalid physical metadata reached emission"
        emissions.append((name, tuple(inputs), output))
        return output

    def allocate(*args: Any, **kwargs: Any) -> Any:
        allocations.append(True)
        return output

    context = SimpleNamespace(
        builder=SimpleNamespace(opset=23, op=op),
        require_value_for_var=lambda var: values[eqn.invars.index(var)],
        allocate_value_for_var=allocate,
        bind_value_for_var=lambda var, value: bindings.append((var, value)),
        fresh_name=lambda name: f"{name}-{len(emissions)}",
    )
    DotGeneralPlugin()._lower_source(context, eqn)
    return emissions, allocations, bindings


@pytest.mark.parametrize("port", ["left", "right", "output"])
@pytest.mark.parametrize("fault", ["dtype", "shape", "missing"])
def test_mixed_local_cast_never_repairs_wrong_physical_ports(
    port: str, fault: str
) -> None:
    with jax.enable_x64(True):
        eqn = _equation()
        _validate(eqn)
        values, output = _physical(eqn)
        emissions, allocations, bindings = _lower(eqn, values, output)
        assert [entry[0] for entry in emissions] == ["Cast", "MatMul"]
        assert len(allocations) == len(bindings) == 1
        assert emissions[0][1] == (values[0],)
        values, output = _physical(eqn)
        target = (
            values[0] if port == "left" else values[1] if port == "right" else output
        )
        if fault == "dtype":
            target.type = ir.TensorType(
                ir.DataType.DOUBLE if port == "left" else ir.DataType.FLOAT
            )
        else:
            target.shape = (
                None
                if fault == "missing"
                else ir.Shape(tuple(reversed(tuple(target.shape))))
            )
        before = [(v.type, v.shape) for v in (*values, output)]
        with pytest.raises(ValueError, match="physical matrix dtype/shape"):
            _lower(eqn, values, output, forbidden=True)
        assert [(v.type, v.shape) for v in (*values, output)] == before


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "shape",
        "weak",
        "context",
        "algorithm",
        "same-f32-preferred-f64",
        "bf16",
        "integer",
    ],
)
def test_native_mixed_contract_corruptions_reject_after_valid_baseline(
    fault: str,
) -> None:
    with jax.enable_x64(True):
        eqn = _equation()
        _validate(eqn)
        values, output = _physical(eqn)
        assert [entry[0] for entry in _lower(eqn, values, output)[0]] == [
            "Cast",
            "MatMul",
        ]
        original = tuple(
            (v, v.aval, v.aval.shape, v.aval.dtype, v.aval.weak_type)
            for v in (*eqn.invars, *eqn.outvars)
        )
        if fault in {"dtype", "shape", "weak"}:
            change = (
                {"dtype": np.float32}
                if fault == "dtype"
                else {"shape": (4, 2)} if fault == "shape" else {"weak_type": True}
            )
            eqn.outvars = [Var(eqn.outvars[0].aval.update(**change))]
        elif fault == "context":
            eqn.ctx = SimpleNamespace(manager=object())
        elif fault == "algorithm":
            eqn.params["precision"] = jax.lax.DotAlgorithmPreset.F64_F64_F64
        elif fault == "same-f32-preferred-f64":
            eqn.invars[1] = Var(eqn.invars[1].aval.update(dtype=np.float32))
            eqn.params["preferred_element_type"] = np.float64
        elif fault == "bf16":
            import jax.numpy as jnp

            eqn.invars[0] = Var(eqn.invars[0].aval.update(dtype=jnp.bfloat16))
        else:
            eqn.invars[0] = Var(eqn.invars[0].aval.update(dtype=np.int32))
        with pytest.raises(SourcePrecisionError) as caught:
            _validate(eqn)
        assert caught.value.category == (
            "lowering-defect"
            if fault in {"dtype", "shape", "weak", "context"}
            else "profile-exclusion"
        )
        for var, aval, shape, dtype, weak in original:
            assert var.aval is aval and (aval.shape, aval.dtype, aval.weak_type) == (
                shape,
                dtype,
                weak,
            )


def test_frozen_native_mixed_result_rule_cannot_be_replaced_by_mutable_abstract_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with jax.enable_x64(True):
        eqn = _equation()
        _validate(eqn)
        calls = []

        def replacement(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return eqn.outvars[0].aval, frozenset()

        with monkeypatch.context() as temporary:
            temporary.setitem(vars(eqn.primitive), "abstract_eval", replacement)
            _validate(eqn)
            eqn.outvars = [Var(eqn.outvars[0].aval.update(weak_type=True))]
            with pytest.raises(SourcePrecisionError):
                _validate(eqn)
            assert not calls
