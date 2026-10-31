# tests/extra_tests/converter/test_source_complex_select.py

"""Original complex selectors copy both component words in the native domain."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx._compat.jax import ShapedArray, Var
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    _validate_complex_select,
)
from jax2onnx.plugins.jax.lax.select_n import _lower_source_complex_select
from tests.extra_tests.converter.test_source_complex_broadcast import _export_original
from tests.extra_tests.converter.test_source_complex_conj import _payload


_CONFIGURATIONS = [("complex64", False), ("complex64", True), ("complex128", True)]
_SHAPES = [(), (0,), (0, 3), (3, 4), (3, 2)]
_INDEX_CONFIGURATIONS = [
    (dtype, x64, index_dtype)
    for dtype, x64 in _CONFIGURATIONS
    for index_dtype in (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
    if x64 or index_dtype not in ("int64", "uint64")
]


def _cases(
    dtype: str, shape: tuple[int, ...], count: int = 4
) -> tuple[np.ndarray, ...]:
    words = _payload(dtype, (32,))
    size = int(np.prod(shape)) if shape else 1
    return tuple(
        np.resize(np.roll(words, i * 3 + 1), size).reshape(shape).copy()
        for i in range(count)
    )


def _selector(shape: tuple[int, ...], count: int, dtype: Any) -> np.ndarray:
    size = int(np.prod(shape)) if shape else 1
    return (np.arange(size, dtype=np.int64) % count).astype(dtype).reshape(shape)


def _selected_copy(which: np.ndarray, cases: tuple[np.ndarray, ...]) -> np.ndarray:
    """Independent coordinate copying; no arithmetic on complex components."""
    shape = cases[0].shape
    result = np.empty(shape, dtype=cases[0].dtype)
    for coordinate in np.ndindex(shape):
        index = int(which[coordinate] if which.shape else which)
        assert 0 <= index < len(cases)
        result[coordinate] = cases[index][coordinate]
    return result


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("shape", _SHAPES)
def test_original_boolean_select_select_n_and_where(
    dtype: str, x64: bool, opset: int, shape: tuple[int, ...], tmp_path: Path
) -> None:
    a, b = _cases(dtype, shape, 2)
    mask = _selector(shape, 2, np.bool_)
    no, yes = np.asarray(False), np.asarray(True)

    def original(p: Any, false: Any, true: Any, x: Any, y: Any) -> Any:
        return (
            jax.lax.select(p, x, y),
            jax.lax.select_n(p, x, y),
            jax.lax.select_n(false, x, y),
            jax.lax.select_n(true, x, y),
            jnp.where(p, y, x),
            jax.lax.select_n(false, x),
            p,
        )

    expected = (
        _selected_copy(mask, (b, a)),
        _selected_copy(mask, (a, b)),
        a.copy(),
        b.copy(),
        _selected_copy(mask, (a, b)),
        a.copy(),
        mask.copy(),
    )
    with jax.enable_x64(x64):
        _export_original(original, (mask, no, yes, a, b), expected, tmp_path, opset)
    assert "select_n" in (tmp_path / "native-trace.txt").read_text()


@pytest.mark.parametrize("dtype,x64,index_dtype", _INDEX_CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("shape", [(), (0, 3), (3, 4), (3, 2)])
def test_original_integer_one_two_three_four_case_selection(
    dtype: str,
    x64: bool,
    index_dtype: str,
    opset: int,
    shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    cases = _cases(dtype, shape)
    indices = tuple(
        _selector(shape, count, np.dtype(index_dtype)) for count in (1, 2, 3, 4)
    )
    scalar = np.asarray(2, dtype=index_dtype)

    def original(
        one: Any,
        two: Any,
        three: Any,
        four: Any,
        scalar_three: Any,
        a: Any,
        b: Any,
        c: Any,
        d: Any,
    ) -> Any:
        return (
            jax.lax.select_n(one, a),
            jax.lax.select_n(two, a, b),
            jax.lax.select_n(three, a, b, c),
            jax.lax.select_n(four, a, b, c, d),
            jax.lax.select_n(scalar_three, a, b, c),
            one,
            two,
            three,
            four,
            scalar_three,
        )

    expected = tuple(
        _selected_copy(index, cases[:count])
        for count, index in zip((1, 2, 3, 4), indices, strict=True)
    ) + (
        _selected_copy(scalar, cases[:3]),
        *tuple(index.copy() for index in indices),
        scalar.copy(),
    )
    with jax.enable_x64(x64):
        _export_original(
            original, (*indices, scalar, *cases), expected, tmp_path, opset
        )
    trace = (tmp_path / "native-trace.txt").read_text()
    assert "select_n" in trace
    assert np.load(tmp_path / "original-inputs.npz")["3"].dtype == np.dtype(index_dtype)


@onnx_function
class _ComplexSelectCapture:
    def __init__(self, cases: tuple[np.ndarray, ...]) -> None:
        self.cases = cases

    def __call__(self, which: Any, real: Any, integer: Any) -> Any:
        selected = jax.lax.select_n(which, *self.cases)
        return selected, real, integer, self.cases[0], selected


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("named", [False, True])
def test_noncontiguous_capture_named_scope_and_real_index_siblings(
    dtype: str, x64: bool, opset: int, named: bool, tmp_path: Path
) -> None:
    cases = tuple(value[:, ::2] for value in _cases(dtype, (3, 8)))
    strides = tuple(value.strides for value in cases)
    frozen = tuple(value.tobytes() for value in cases)
    which = _selector((3, 4), 4, np.int32)
    real = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    integer = np.asarray([0, 1, 2, 3], np.uint32)
    owner = _ComplexSelectCapture(cases) if named else None

    def original(index: Any, r: Any, i: Any) -> Any:
        if owner is not None:
            return owner(index, r, i)
        selected = jax.lax.select_n(index, *cases)
        return selected, r, i, cases[0], selected

    expected = _selected_copy(which, cases)
    with jax.enable_x64(x64):
        model = _export_original(
            original,
            (which, real, integer),
            (expected, real.copy(), integer.copy(), cases[0].copy(), expected.copy()),
            tmp_path,
            opset,
        )
    assert len(model.functions) == int(named)
    assert tuple(value.tobytes() for value in cases) == frozen
    assert tuple(value.strides for value in cases) == strides
    if owner is not None:
        assert owner.cases is cases


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("weak_selector", [False, True])
@pytest.mark.parametrize(
    "weak_arms", [(True, True), (True, False), (False, True), (False, False)]
)
def test_native_weakness_depends_on_cases_not_integer_selector(
    x64: bool, weak_selector: bool, weak_arms: tuple[bool, bool], tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        weak = jnp.asarray(1.25 - 2.5j)
        strong = jnp.asarray(np.asarray(weak))
        cases = tuple(
            jnp.broadcast_to(weak if flag else strong, (2, 3)) for flag in weak_arms
        )
        selector = jnp.asarray(1) if weak_selector else jnp.asarray(np.int32(1))

        def original(which: Any, a: Any, b: Any) -> Any:
            return jax.lax.select_n(which, a, b)

        expected = np.asarray(cases[1]).copy()
        _export_original(original, (selector, *cases), (expected,), tmp_path, 23)
    record = json.loads((tmp_path / "native-recording.json").read_text())
    assert all(
        value["leaves"][0]["weak_type"] is all(weak_arms)
        for value in record["signatures"].values()
    )


def _equation(dtype: str = "complex64", cases: int = 4) -> Any:
    def original(which: Any, *arms: Any) -> Any:
        return jax.lax.select_n(which, *arms)

    with jax.enable_x64(dtype == "complex128"):
        specs = [
            jax.ShapeDtypeStruct((2, 3), np.int32),
            *[jax.ShapeDtypeStruct((2, 3), np.dtype(dtype)) for _ in range(cases)],
        ]
        trace = jax.make_jaxpr(original)(*specs)
    return next(e for e in trace.jaxpr.eqns if e.primitive is jax.lax.select_n_p)


def _changed(equation: Any) -> Any:
    return SimpleNamespace(
        primitive=equation.primitive,
        effects=equation.effects,
        params=dict(equation.params),
        invars=list(equation.invars),
        outvars=list(equation.outvars),
    )


def _validate(equation: Any, opset: int = 23) -> None:
    site = _Site("complex select test", opset, "select_n")
    _validate_complex_select(
        equation,
        [_tensor(v, site) for v in equation.invars],
        [_tensor(v, site) for v in equation.outvars],
        site,
    )


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    pytest.fail("invalid complex select reached allocation/emission")


@pytest.mark.parametrize(
    "defect",
    [
        "identity",
        "effects",
        "container",
        "parameter",
        "no-cases",
        "result-arity",
        "width",
        "result-width",
        "real-cases",
        "shape",
        "selector-shape",
        "selector-type",
        "selector-key",
        "bool-three",
        "weakness",
    ],
)
def test_strict_native_complex_select_contract(defect: str) -> None:
    changed = _changed(_equation())
    if defect == "identity":
        changed.primitive = SimpleNamespace(name="select_n")
    elif defect == "effects":
        changed.effects = {"unexpected"}
    elif defect == "container":
        changed.params = []
    elif defect == "parameter":
        changed.params["out_sharding"] = None
    elif defect == "no-cases":
        changed.invars = changed.invars[:1]
    elif defect == "result-arity":
        changed.outvars = []
    elif defect == "width":
        changed.invars[2] = Var(
            changed.invars[2].aval.update(dtype=np.dtype(np.complex128))
        )
    elif defect == "result-width":
        changed.outvars[0] = Var(
            changed.outvars[0].aval.update(dtype=np.dtype(np.float32))
        )
    elif defect == "real-cases":
        changed.invars[1:] = [
            Var(var.aval.update(dtype=np.dtype(np.float32)))
            for var in changed.invars[1:]
        ]
    elif defect == "shape":
        changed.invars[-1] = Var(changed.invars[-1].aval.update(shape=(3, 2)))
    elif defect == "selector-shape":
        changed.invars[0] = Var(changed.invars[0].aval.update(shape=(3,)))
    elif defect == "selector-type":
        changed.invars[0] = Var(
            changed.invars[0].aval.update(dtype=np.dtype(np.float32))
        )
    elif defect == "selector-key":
        changed.invars[0] = Var(jax.typeof(jax.random.key(17)))
    elif defect == "bool-three":
        changed.invars[0] = Var(changed.invars[0].aval.update(dtype=np.dtype(np.bool_)))
    else:
        changed.outvars[0] = Var(changed.outvars[0].aval.update(weak_type=True))
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        fresh_name=_forbidden,
        allocate_value_for_var=_forbidden,
        require_value_for_var=_forbidden,
    )
    with pytest.raises(SourcePrecisionError):
        _lower_source_complex_select(ctx, changed)


def _bindings(equation: Any) -> dict[int, ir.Value]:
    result = {}
    for index, var in enumerate(equation.invars):
        dtype = np.dtype(var.aval.dtype)
        complex_data = dtype.kind == "c"
        kind = (
            ir.DataType.FLOAT
            if dtype == np.dtype(np.complex64)
            else ir.DataType.DOUBLE if complex_data else ir.DataType.INT32
        )
        shape = (*var.aval.shape, 2) if complex_data else var.aval.shape
        result[id(var)] = ir.val(f"input_{index}", dtype=kind, shape=shape)
    return result


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("position", range(5))
@pytest.mark.parametrize("defect", ["dtype", "shape", "unknown"])
def test_every_selector_and_case_descriptor_precedes_allocation(
    dtype: str, position: int, defect: str
) -> None:
    equation = _equation(dtype)
    values = _bindings(equation)
    value = values[id(equation.invars[position])]
    if defect == "dtype":
        value.type = ir.TensorType(ir.DataType.BOOL)
    elif defect == "shape":
        value.shape = ir.Shape((12345,))
    else:
        value.shape = None
    before = [(v.type, v.shape) for v in values.values()]
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=lambda v: values[id(v)],
        fresh_name=_forbidden,
        allocate_value_for_var=_forbidden,
    )
    with pytest.raises(TypeError):
        _lower_source_complex_select(ctx, equation)
    assert [(v.type, v.shape) for v in values.values()] == before


def test_output_descriptor_precedes_first_emission() -> None:
    equation = _equation()
    values = _bindings(equation)
    bad = ir.val("bad", dtype=ir.DataType.FLOAT, shape=(1,))
    allocations = []

    def allocate(*args: Any, **kwargs: Any) -> Any:
        allocations.append(True)
        return bad

    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23, op=_forbidden),
        require_value_for_var=lambda v: values[id(v)],
        fresh_name=lambda prefix: prefix,
        allocate_value_for_var=allocate,
    )
    with pytest.raises(TypeError):
        _lower_source_complex_select(ctx, equation)
    assert allocations == [True]
    assert bad.shape == ir.Shape((1,)) and bad.dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("large,rejected", [(1 << 61, True), (1 << 60, False)])
def test_stacked_int64_geometry_is_proved_without_allocating_arrays(
    large: int, rejected: bool
) -> None:
    equation = _changed(_equation(cases=2))
    equation.invars[0] = Var(ShapedArray((), np.dtype(np.int32)))
    equation.invars[1:] = [
        Var(ShapedArray((large,), np.dtype(np.complex64))) for _ in range(2)
    ]
    equation.outvars = [Var(ShapedArray((large,), np.dtype(np.complex64)))]
    if rejected:
        ctx = SimpleNamespace(
            precision_policy=PrecisionPolicy.SOURCE,
            builder=SimpleNamespace(opset=23),
            fresh_name=_forbidden,
            allocate_value_for_var=_forbidden,
            require_value_for_var=_forbidden,
        )
        with pytest.raises(
            SourcePrecisionError, match="stacked physical element count exceeds INT64"
        ):
            _lower_source_complex_select(ctx, equation)
    else:
        _validate(equation)


@pytest.mark.parametrize("opset", [21, 24])
def test_complex_select_keeps_unselected_opsets_explicit(opset: int) -> None:
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(), opset)


@pytest.mark.parametrize("kind", ["real-tail-two", "legacy"])
def test_separate_real_and_legacy_owners_bypass_private_complex_lowerer(
    kind: str,
) -> None:
    equation = _equation()
    policy = PrecisionPolicy.LEGACY if kind == "legacy" else PrecisionPolicy.SOURCE
    if kind == "real-tail-two":
        equation = _changed(equation)
        equation.invars[1:] = [
            Var(ShapedArray((2, 3, 2), np.dtype(np.float32))) for _ in range(4)
        ]
        equation.outvars[0] = Var(ShapedArray((2, 3, 2), np.dtype(np.float32)))
    assert not _lower_source_complex_select(
        SimpleNamespace(precision_policy=policy), equation
    )
