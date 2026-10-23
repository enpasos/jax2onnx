# tests/extra_tests/converter/test_source_mixed_preferred_fp32.py

"""Explicit native mixed-width FP32 work preserves original storage and ports.

The finite bitwise controls here do not change any original model recipe or its
criteria. Their coordinate oracle narrows before products, as native CPU JAX
requires, and deliberately distinguishes a final cast of FP64 computation.
"""

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx._compat.jax import Var
from jax2onnx.converter.source_profile import SourcePrecisionError
from tests.extra_tests.converter.test_source_mixed_width_contraction import (
    _lower,
    _runtime_payloads,
    _signature,
    _validate,
)
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _body,
    _observe,
    _oracle,
    _restore_settings as _restore_settings,
)

_MATRIX = (((1,), (0,)), ((), ()))
_CASES = [
    ("matrix-32-64", (2, 3), (3, 4), _MATRIX, "float32"),
    ("matrix-64-32", (2, 3), (3, 4), _MATRIX, "float64"),
    (
        "ordered-contract",
        (2, 3, 4),
        (4, 3, 5),
        (((2, 1), (0, 1)), ((), ())),
        "float64",
    ),
    ("free-B-left", ("B", 3), (3, 4), _MATRIX, "float32"),
    ("free-B-right", (4, 3), ("B", 3), (((1,), (1,)), ((), ())), "float64"),
    (
        "paired-B",
        ("B", 2, 3),
        ("B", 3, 4),
        (((2,), (1,)), ((0,), (0,))),
        "float64",
    ),
    ("K-zero", (2, 0), (0, 3), _MATRIX, "float32"),
    ("M-zero", (0, 3), (3, 4), _MATRIX, "float64"),
    ("N-zero", (2, 3), (3, 0), _MATRIX, "float32"),
]


def _dims(shape: Any, batch: Any) -> tuple[Any, ...]:
    return tuple(batch if d == "B" else d for d in shape)


def _payload(shape: Any, dtype: str) -> np.ndarray:
    grid = np.arange(np.prod(shape, dtype=np.int64)).reshape(shape)
    value = ((grid * 3 + 1) % 9 - 4) / 8
    return (value + (2.0**-35 if dtype == "float64" else 0)).astype(dtype)


def _dots(closed: Any) -> list[Any]:
    found = []
    for eqn in _body(closed).eqns:
        if eqn.primitive is jax.lax.dot_general_p:
            found.append(eqn)
        elif eqn.primitive.name in {"jit", "pjit"}:
            found.extend(_dots(eqn.params["jaxpr"]))
    return found


def _serialized_route(model: Any, *, empty: bool, capture: bool = False) -> None:
    saved = onnx.load_model_from_string(model.SerializeToString())
    onnx.checker.check_model(saved, full_check=True)
    graph = onnx.shape_inference.infer_shapes(saved, strict_mode=True).graph
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in (*graph.input, *graph.value_info, *graph.output)
    }
    types.update({v.name: v.data_type for v in graph.initializer})
    assert types[graph.output[0].name] == onnx.TensorProto.FLOAT
    casts = [n for n in graph.node if n.op_type == "Cast"]
    dots = [n for n in graph.node if n.op_type == "MatMul"]
    if empty:
        assert not casts and not dots
        (zero,) = [n for n in graph.node if n.op_type == "ConstantOfShape"]
        assert types[zero.output[0]] == onnx.TensorProto.FLOAT
        assert any(
            a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT
            for a in zero.attribute
        )
        return
    (cast,), (dot,) = casts, dots
    assert types[cast.input[0]] == onnx.TensorProto.DOUBLE
    assert types[cast.output[0]] == onnx.TensorProto.FLOAT
    assert any(a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in cast.attribute)
    assert all(
        types[name] == onnx.TensorProto.FLOAT for name in (*dot.input, *dot.output)
    )
    producers = {name: node for node in graph.node for name in node.output}
    origins = []
    for name in dot.input:
        while name in producers and producers[name].op_type in {
            "Reshape",
            "Transpose",
            "Identity",
        }:
            name = producers[name].input[0]
        origins.append(name)
    assert cast.output[0] in origins
    assert not any(n.input[0] == dot.output[0] for n in casts)
    if capture:
        assert cast.input[0] in {v.name for v in graph.initializer}
    else:
        assert cast.input[0] in {v.name for v in graph.input}


@pytest.mark.parametrize(
    "name,left,right,dimensions,left_dtype", _CASES, ids=[v[0] for v in _CASES]
)
def test_native_explicit_fp32_mixed_geometries(
    name: str, left: Any, right: Any, dimensions: Any, left_dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        right_dtype = "float32" if left_dtype == "float64" else "float64"
        (batch,) = export.symbolic_shape("B")
        specs = [
            jax.ShapeDtypeStruct(_dims(s, batch), d)
            for s, d in ((left, left_dtype), (right, right_dtype))
        ]

        def fn(a: Any, b: Any) -> Any:
            return (
                jax.lax.dot_general(
                    a, b, dimensions, preferred_element_type=np.float32
                ),
                a,
                b,
            )

        feeds, wanted = [], []
        output_shape = None
        for size in (2, 5) if "B" in (*left, *right) else (2,):
            a, b = _payload(_dims(left, size), left_dtype), _payload(
                _dims(right, size), right_dtype
            )
            target = _oracle(
                a.astype(np.float32), b.astype(np.float32), dimensions, np.float32
            )
            feeds.append((jax.device_put(a), jax.device_put(b)))
            wanted.append((target, a, b))
            if output_shape is None:
                (lc, rc), (lb, rb) = dimensions
                output_shape = (
                    *(left[i] for i in lb),
                    *(d for i, d in enumerate(left) if i not in (*lc, *lb)),
                    *(d for i, d in enumerate(right) if i not in (*rc, *rb)),
                )
        with _runtime_payloads(tmp_path):
            model, native, _ = _observe(
                fn,
                specs,
                feeds,
                wanted,
                tmp_path,
                output_shapes=[output_shape, left, right],
            )
        (dot,) = _dots(native)
        assert _signature(dot.invars) == (
            (left_dtype, left, False),
            (right_dtype, right, False),
        )
        assert _signature(dot.outvars) == (("float32", output_shape, False),)
        assert np.dtype(dot.params["preferred_element_type"]) == np.dtype(np.float32)
        assert dot.params["dimension_numbers"] == dimensions
        assert dot.params["precision"] is None
        _serialized_route(model, empty=name in {"K-zero", "M-zero", "N-zero"})
        assert [v.type.tensor_type.elem_type for v in model.graph.output[1:]] == [
            v.type.tensor_type.elem_type for v in model.graph.input
        ]


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("wide_side", ["left", "right"])
def test_captured_fp64_storage_with_native_fp32_input_and_jit_scope(
    x64: bool, wide_side: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        storage = np.asarray([[1 + 2.0**-24], [-1 - 2.0**-24]], np.float64)
        if wide_side == "right":
            storage = storage.T.copy()
        weight = jax.device_put(storage)
    original = np.asarray(weight).tobytes()
    with jax.enable_x64(x64):

        @jax.jit
        def inner(x: Any) -> Any:
            return jax.lax.dot_general(
                weight if wide_side == "left" else x,
                x if wide_side == "left" else weight,
                _MATRIX,
                preferred_element_type=np.float32,
            )

        def fn(x: Any) -> Any:
            return inner(x), weight, weight, x

        feed = np.asarray([[1 + 2.0**-23, 2.0]], np.float32)
        if wide_side == "right":
            feed = feed.T.copy()
        expected = _oracle(
            storage.astype(np.float32) if wide_side == "left" else feed,
            feed if wide_side == "left" else storage.astype(np.float32),
            _MATRIX,
            np.float32,
        )
        with _runtime_payloads(tmp_path):
            model, trace, _ = _observe(
                fn,
                [jax.ShapeDtypeStruct(feed.shape, np.float32)],
                [(jax.device_put(feed),)],
                [(expected, storage, storage, feed)],
                tmp_path,
                output_shapes=[(2, 2), storage.shape, storage.shape, feed.shape],
            )
        (dot,) = _dots(trace)
        assert [np.dtype(v.aval.dtype).name for v in (*dot.invars, *dot.outvars)] == (
            ["float64", "float32", "float32"]
            if wide_side == "left"
            else ["float32", "float64", "float32"]
        )
        assert len(model.graph.input) == 1
        _serialized_route(model, empty=False, capture=True)
        aliases = model.graph.output[1:3]
        assert all(
            v.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE for v in aliases
        )
        assert np.asarray(weight).dtype == np.dtype(np.float64)
        assert np.asarray(weight).tobytes() == original
        # Independent rational product: one cast before this K=1 product differs
        # by one FP32 ULP from computing in FP64 and casting only the result.
        exact = Fraction(1) + Fraction(1, 2**24)
        data = Fraction(1) + Fraction(1, 2**23)
        before = np.float32(float(Fraction(1) * data))
        after = np.float32(float(exact * data))
        assert before.tobytes() != after.tobytes()
        assert expected[0, 0].tobytes() == before.tobytes()
        (tmp_path / "rounding-witness.json").write_text(
            json.dumps(
                {
                    "narrow_before_product": float(before),
                    "wide_product_then_cast": float(after),
                    "weight_dtype": str(np.asarray(weight).dtype),
                    "caller_x64": x64,
                },
                indent=2,
            )
        )


def _equation(*, empty: bool = False) -> Any:
    native = jax.make_jaxpr(
        lambda a, b: jax.lax.dot_general(
            a, b, _MATRIX, preferred_element_type=np.float32
        )
    )(
        jax.ShapeDtypeStruct((2, 0) if empty else (2, 3), np.float64),
        jax.ShapeDtypeStruct((0, 4) if empty else (3, 4), np.float32),
    )
    (eqn,) = _body(native).eqns
    return SimpleNamespace(
        primitive=eqn.primitive,
        params=dict(eqn.params),
        invars=list(eqn.invars),
        outvars=list(eqn.outvars),
        effects=eqn.effects,
        ctx=eqn.ctx,
    )


def _physical(eqn: Any) -> tuple[Any, Any]:
    return (
        [
            ir.val(
                f"input{i}",
                dtype=ir.DataType.DOUBLE if i == 0 else ir.DataType.FLOAT,
                shape=v.aval.shape,
            )
            for i, v in enumerate(eqn.invars)
        ],
        ir.val("output", dtype=ir.DataType.FLOAT, shape=eqn.outvars[0].aval.shape),
    )


@pytest.mark.parametrize("port", ["left", "right", "output"])
@pytest.mark.parametrize("fault", ["dtype", "shape", "missing", "missing-dtype"])
@pytest.mark.parametrize("empty", [False, True])
def test_narrowing_never_repairs_physical_metadata(
    port: str, fault: str, empty: bool
) -> None:
    with jax.enable_x64(True):
        eqn = _equation(empty=empty)
        _validate(eqn)
        values, output = _physical(eqn)
        emissions, allocations, bindings = _lower(eqn, values, output)
        assert [v[0] for v in emissions] == (
            ["ConstantOfShape"] if empty else ["Cast", "MatMul"]
        )
        if not empty:
            assert emissions[0][1] == (values[0],)
        assert len(allocations) == len(bindings) == 1
        assert bindings[0] == (eqn.outvars[0], output)
        assert emissions[0][2].dtype == ir.DataType.FLOAT
        target = (
            values[0] if port == "left" else values[1] if port == "right" else output
        )
        if fault == "dtype":
            target.type = ir.TensorType(
                ir.DataType.FLOAT if port == "left" else ir.DataType.DOUBLE
            )
        elif fault == "missing-dtype":
            target.type = None
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
        "HIGH",
        "HIGHEST",
        "algorithm",
        "same-width-f64",
    ],
)
def test_native_metadata_and_fp32_precision_gates_remain_strict(fault: str) -> None:
    with jax.enable_x64(True):
        eqn = _equation()
        _validate(eqn)
        original = tuple((v, v.aval) for v in (*eqn.invars, *eqn.outvars))
        if fault in {"dtype", "shape", "weak"}:
            update = (
                {"dtype": np.float64}
                if fault == "dtype"
                else {"shape": (4, 2)} if fault == "shape" else {"weak_type": True}
            )
            eqn.outvars = [Var(eqn.outvars[0].aval.update(**update))]
        elif fault == "context":
            eqn.ctx = SimpleNamespace(manager=object())
        elif fault in {"HIGH", "HIGHEST"}:
            precision = getattr(jax.lax.Precision, fault)
            eqn.params["precision"] = (precision, precision)
        elif fault == "algorithm":
            eqn.params["precision"] = jax.lax.DotAlgorithmPreset.F32_F32_F32
        else:
            eqn.invars[1] = Var(eqn.invars[1].aval.update(dtype=np.float64))
        with pytest.raises(SourcePrecisionError) as caught:
            _validate(eqn)
        assert caught.value.category == (
            "lowering-defect"
            if fault in {"dtype", "shape", "weak", "context"}
            else "profile-exclusion"
        )
        with pytest.raises((SourcePrecisionError, NotImplementedError)):
            _lower(eqn, *_physical(eqn), forbidden=True)
        assert all(v.aval is aval for v, aval in original)


def test_frozen_fp32_native_result_rule_ignores_mutable_abstract_eval(
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
