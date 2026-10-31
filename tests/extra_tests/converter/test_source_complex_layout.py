# tests/extra_tests/converter/test_source_complex_layout.py

"""Original static complex tensor layouts preserve the complete physical pair."""

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
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, ShapedArray, Var
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    validate_source_jaxpr,
)
from tests.extra_tests.converter.test_source_complex_broadcast import _export_original
from tests.extra_tests.converter.test_source_complex_conj import _payload


_OPERATIONS = ("copy", "reshape", "transpose", "squeeze", "concatenate", "rev")
_CONFIGURATIONS = [("complex64", False), ("complex64", True), ("complex128", True)]
_KINDS = ("regular", "identity", "empty", "mapped")


def _primitives() -> dict[str, Any]:
    return {
        "copy": jax.lax.copy_p,
        "reshape": jax.lax.reshape_p,
        "transpose": jax.lax.transpose_p,
        "squeeze": jax.lax.squeeze_p,
        "concatenate": jax.lax.concatenate_p,
        "rev": jax.lax.rev_p,
    }


def _case(
    operation: str, kind: str, dtype: str
) -> tuple[Any, tuple[Any, ...], np.ndarray]:
    if operation == "copy":
        shape = {"regular": (4, 4), "identity": (), "empty": (0, 2), "mapped": (8, 4)}[
            kind
        ]
        value = _payload(dtype, shape)
        if kind == "mapped":
            value = value[::2]
        return jnp.copy, (value,), value.copy()
    if operation == "reshape":
        value = _payload(dtype, (2, 0, 4) if kind == "empty" else (4, 4))
        target = {
            "regular": (2, 8),
            "identity": (4, 4),
            "empty": (0, 8),
            "mapped": (2, 8),
        }[kind]
        if kind == "mapped":

            def original(x: Any) -> Any:
                return jax.lax.reshape(x, target, dimensions=(1, 0))

            expected = np.transpose(value, (1, 0)).reshape(target)
        else:

            def original(x: Any) -> Any:
                return jnp.reshape(x, target)

            expected = np.reshape(value, target)
        return original, (value,), expected.copy()
    if operation == "transpose":
        shape = {
            "regular": (2, 8),
            "identity": (4, 4),
            "empty": (2, 0, 4),
            "mapped": (4, 4),
        }[kind]
        axes = (
            (2, 0, 1) if kind == "empty" else (0, 1) if kind == "identity" else (1, 0)
        )
        value = _payload(dtype, shape)

        def original(x: Any) -> Any:
            return jax.lax.transpose(x, axes)

        if kind == "mapped":
            assert value.tobytes() != value.T.tobytes()
        return original, (value,), np.transpose(value, axes).copy()
    if operation == "squeeze":
        shape = {
            "regular": (1, 4, 1, 4),
            "identity": (1, 4, 1, 4),
            "empty": (1, 0, 1, 4),
            "mapped": (1, 1, 4, 4),
        }[kind]
        axes = () if kind == "identity" else (1,) if kind == "mapped" else (0, 2)
        value = _payload(dtype, shape)

        def original(x: Any) -> Any:
            return jax.lax.squeeze(x, axes)

        return original, (value,), np.squeeze(value, axis=axes).copy()
    if operation == "concatenate":
        if kind == "identity":
            values, axis = (_payload(dtype, (4, 4)),), 1
        elif kind == "empty":
            values, axis = (_payload(dtype, (4, 0)), _payload(dtype, (4, 4))), 1
        elif kind == "mapped":
            values, axis = (
                tuple(_payload(dtype, shape) for shape in ((2, 2), (2, 1), (2, 5))),
                1,
            )
        else:
            values, axis = (_payload(dtype, (2, 4)), _payload(dtype, (2, 4))), 0

        def original(*xs: Any) -> Any:
            return jnp.concatenate(xs, axis=axis)

        return original, values, np.concatenate(values, axis=axis).copy()
    if operation == "rev":
        shape = () if kind == "identity" else (2, 0, 4) if kind == "empty" else (4, 4)
        axes = (
            ()
            if kind == "identity"
            else (0, 1) if kind in {"empty", "mapped"} else (0,)
        )
        value = _payload(dtype, shape)

        def original(x: Any) -> Any:
            return jax.lax.rev(x, axes)

        return original, (value,), np.flip(value, axis=axes).copy()
    raise AssertionError(operation)


def _layout_equations(trace: Any) -> list[Any]:
    pending = [trace]
    seen: set[int] = set()
    result = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        body = current.jaxpr if isinstance(current, ClosedJaxpr) else current
        for equation in body.eqns:
            if equation.primitive in _primitives().values():
                result.append(equation)
            pending.extend(
                value
                for value in equation.params.values()
                if isinstance(value, (Jaxpr, ClosedJaxpr))
            )
    return result


def _record_layout_trace(
    original: Any, values: tuple[Any, ...], directory: Path
) -> list[Any]:
    # The unchanged recorder already saved all original roles and its actual
    # trace before oracle/target comparisons. This structural trace uses the
    # identical original callable/specs/context and is bound to that first trace.
    specs = [
        jax.ShapeDtypeStruct(
            value.shape, value.dtype, weak_type=jax.typeof(value).weak_type
        )
        for value in values
    ]
    trace = jax.make_jaxpr(original)(*specs)
    assert str(trace) == (directory / "native-trace.txt").read_text()
    equations = _layout_equations(trace)
    (directory / "layout-equations.json").write_text(
        json.dumps(
            [
                {
                    "primitive": equation.primitive.name,
                    "params": repr(equation.params),
                    "inputs": [list(var.aval.shape) for var in equation.invars],
                    "outputs": [list(var.aval.shape) for var in equation.outvars],
                }
                for equation in equations
            ],
            indent=2,
        )
    )
    return equations


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("kind", _KINDS)
def test_original_static_complex_layout_family(
    dtype: str, x64: bool, opset: int, operation: str, kind: str, tmp_path: Path
) -> None:
    original, values, expected = _case(operation, kind, dtype)
    strides = tuple(value.strides for value in values)
    previous = bool(jax.config.jax_enable_x64)
    with jax.enable_x64(x64):
        _export_original(original, values, (expected,), tmp_path, opset)
        equations = _record_layout_trace(original, values, tmp_path)
    matches = [e for e in equations if e.primitive is _primitives()[operation]]
    (tmp_path / "native-layout-selection.json").write_text(
        json.dumps(
            {
                "requested": operation,
                "kind": kind,
                "equations": len(matches),
                "native_forwarded": not matches,
            },
            indent=2,
        )
    )
    if operation == "copy" or kind in {"regular", "mapped"}:
        assert matches, "nontrivial original witness did not reach its native primitive"
    assert tuple(value.strides for value in values) == strides
    assert bool(jax.config.jax_enable_x64) is previous


def _complete_family(value: Any) -> tuple[Any, ...]:
    copied = jnp.array(value, copy=True)
    reshaped = jax.lax.reshape(copied, (2, 8), dimensions=(1, 0))
    transposed = jax.lax.transpose(reshaped, (1, 0))
    squeezed = jax.lax.squeeze(jnp.reshape(transposed, (1, 8, 2, 1)), (0, 3))
    concatenated = jax.lax.concatenate((squeezed, squeezed), 1)
    reversed_value = jax.lax.rev(concatenated, (0, 1))
    return copied, reshaped, transposed, squeezed, concatenated, reversed_value


def _complete_oracle(value: np.ndarray) -> tuple[np.ndarray, ...]:
    copied = value.copy()
    reshaped = np.transpose(copied, (1, 0)).reshape((2, 8))
    transposed = np.transpose(reshaped, (1, 0))
    squeezed = np.squeeze(np.reshape(transposed, (1, 8, 2, 1)), axis=(0, 3))
    concatenated = np.concatenate((squeezed, squeezed), axis=1)
    reversed_value = np.flip(concatenated, axis=(0, 1))
    return tuple(
        v.copy()
        for v in (copied, reshaped, transposed, squeezed, concatenated, reversed_value)
    )


@onnx_function
class _ComplexLayoutCapture:
    def __init__(self, capture: np.ndarray) -> None:
        self.capture = capture

    def __call__(self, x: Any) -> Any:
        values = _complete_family(self.capture)
        return x, *values, values[-1]


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("named", [False, True])
def test_complete_family_capture_names_real_sibling_and_repeated_outputs(
    dtype: str, x64: bool, opset: int, named: bool, tmp_path: Path
) -> None:
    capture = _payload(dtype, (8, 4))[::2]
    assert capture.shape == (4, 4) and not capture.flags.c_contiguous
    frozen, strides = capture.tobytes(), capture.strides

    def original(x: Any) -> Any:
        values = _complete_family(capture)
        return x, *values, values[-1]

    owner = _ComplexLayoutCapture(capture) if named else None
    fn = owner if owner is not None else original
    if owner is not None:
        assert owner.capture is capture and owner.capture.strides == strides
    real_pair = np.asarray([1.25, -0.0], dtype=np.float32)
    expected = _complete_oracle(capture)
    with jax.enable_x64(x64):
        proto = _export_original(
            fn, (real_pair,), (real_pair, *expected, expected[-1]), tmp_path, opset
        )
        equations = _record_layout_trace(fn, (real_pair,), tmp_path)
    present = {e.primitive for e in equations}
    # Array construction from a captured host constant may need no copy
    # equation. The unchanged public source remains; argument witnesses above
    # prove genuine copy admission. Record this native distinction explicitly.
    assert {p for name, p in _primitives().items() if name != "copy"} <= present
    (tmp_path / "capture-native-copy.json").write_text(
        json.dumps(
            {
                "copy_equation_present": _primitives()["copy"] in present,
                "public_copy_retained_in_original": True,
            }
        )
    )
    assert len(proto.functions) == (1 if named else 0)
    assert capture.tobytes() == frozen and capture.strides == strides
    if owner is not None:
        assert owner.capture is capture and owner.capture.strides == strides


def _weak_case(operation: str, value: Any) -> tuple[Any, tuple[Any, ...], np.ndarray]:
    if operation == "copy":
        return jnp.copy, (value,), np.asarray(value).copy()
    if operation == "reshape":
        return (
            lambda x: jax.lax.reshape(x, (8,)),
            (value,),
            np.asarray(value).reshape((8,)).copy(),
        )
    if operation == "transpose":
        return (
            lambda x: jax.lax.transpose(x, (1, 0)),
            (value,),
            np.asarray(value).T.copy(),
        )
    if operation == "squeeze":
        operand = jnp.reshape(value, (1, 2, 4, 1))
        return (
            lambda x: jax.lax.squeeze(x, (0, 3)),
            (operand,),
            np.squeeze(np.asarray(operand), axis=(0, 3)).copy(),
        )
    if operation == "concatenate":
        return (
            lambda x, y: jax.lax.concatenate((x, y), 1),
            (value, value),
            np.concatenate((np.asarray(value), np.asarray(value)), axis=1),
        )
    if operation == "rev":
        return (
            lambda x: jax.lax.rev(x, (0, 1)),
            (value,),
            np.flip(np.asarray(value), axis=(0, 1)).copy(),
        )
    raise AssertionError(operation)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("weak", [False, True])
def test_actual_native_family_weakness(
    operation: str, x64: bool, weak: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        scalar = jnp.asarray(1.25 + 2.5j)
        if not weak:
            scalar = jnp.asarray(np.asarray(scalar))
        value = jnp.broadcast_to(scalar, (2, 4))
        original, values, expected = _weak_case(operation, value)
        assert all(jax.typeof(v).weak_type is weak for v in values)
        _export_original(original, values, (expected,), tmp_path, 23)
    recording = json.loads((tmp_path / "native-recording.json").read_text())
    assert all(
        s["leaves"][0]["weak_type"] is weak for s in recording["signatures"].values()
    )


@pytest.mark.parametrize("weak_first", [False, True])
def test_concatenate_weakness_uses_every_actual_operand(
    weak_first: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(False):
        weak = jnp.broadcast_to(jnp.asarray(1.25 + 2.5j), (2, 4))
        strong = jnp.asarray(np.asarray(weak))
        values = (weak, strong) if weak_first else (strong, weak)
        assert [jax.typeof(v).weak_type for v in values] == [weak_first, not weak_first]

        def original(x: Any, y: Any) -> Any:
            return jax.lax.concatenate((x, y), 1)

        _export_original(
            original,
            values,
            (np.concatenate(tuple(np.asarray(v) for v in values), axis=1),),
            tmp_path,
            23,
        )
    recording = json.loads((tmp_path / "native-recording.json").read_text())
    assert all(
        s["leaves"][0]["weak_type"] is False for s in recording["signatures"].values()
    )


def _equation(operation: str, dtype: str = "complex64") -> Any:
    original, values, _ = _case(operation, "regular", dtype)
    with jax.enable_x64(dtype == "complex128"):
        trace = jax.make_jaxpr(original)(
            *[jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
        )
    return next(
        e for e in _layout_equations(trace) if e.primitive is _primitives()[operation]
    )


def _changed(equation: Any) -> SimpleNamespace:
    return SimpleNamespace(
        primitive=equation.primitive,
        effects=equation.effects,
        params=dict(equation.params),
        invars=list(equation.invars),
        outvars=list(equation.outvars),
    )


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize(
    "defect",
    [
        "identity",
        "effects",
        "container",
        "unknown",
        "placement",
        "width",
        "weakness",
        "shape",
        "arity",
    ],
)
def test_family_logical_contract_rejects_forgery(operation: str, defect: str) -> None:
    from jax2onnx.converter.source_profile import _validate_complex_layout

    changed = _changed(_equation(operation))
    if defect == "identity":
        changed.primitive = SimpleNamespace(name=operation)
    elif defect == "effects":
        changed.effects = {"unexpected"}
    elif defect == "container":
        changed.params = []
    elif defect == "unknown":
        changed.params["unknown"] = None
    elif defect == "placement":
        changed.params["sharding"] = object()
    elif defect == "width":
        changed.outvars[0] = Var(
            changed.outvars[0].aval.update(dtype=np.dtype(np.complex128))
        )
    elif defect == "weakness":
        changed.outvars[0] = Var(changed.outvars[0].aval.update(weak_type=True))
    elif defect == "shape":
        shape = tuple(changed.outvars[0].aval.shape)
        changed.outvars[0] = Var(
            changed.outvars[0].aval.update(shape=(shape[0] + 1, *shape[1:]))
        )
    else:
        changed.outvars = []
    site = _Site("complex layout negative", 23, operation)
    with pytest.raises(SourcePrecisionError):
        _validate_complex_layout(
            changed,
            [_tensor(v, site) for v in changed.invars],
            [_tensor(v, site) for v in changed.outvars],
            site,
        )


@pytest.mark.parametrize(
    "operation,defect",
    [
        ("reshape", "count"),
        ("reshape", "duplicate"),
        ("transpose", "duplicate"),
        ("transpose", "incomplete"),
        ("squeeze", "non-singleton"),
        ("squeeze", "duplicate"),
        ("concatenate", "channel-axis"),
        ("concatenate", "non-axis"),
        ("rev", "channel-axis"),
        ("rev", "duplicate"),
    ],
)
def test_family_preserves_owning_shape_and_axis_laws(
    operation: str, defect: str
) -> None:
    from jax2onnx.converter.source_profile import _validate_complex_layout

    changed = _changed(_equation(operation))
    if operation == "reshape":
        if defect == "count":
            changed.params["new_sizes"] = (3, 6)
            changed.outvars[0] = Var(changed.outvars[0].aval.update(shape=(3, 6)))
        else:
            changed.params["dimensions"] = (0, 0)
    elif operation == "transpose":
        changed.params["permutation"] = (0, 0) if defect == "duplicate" else (0,)
    elif operation == "squeeze":
        changed.params["dimensions"] = (1,) if defect == "non-singleton" else (0, 0)
    elif operation == "concatenate":
        if defect == "channel-axis":
            changed.params["dimension"] = len(changed.invars[0].aval.shape)
        else:
            changed.invars[1] = Var(changed.invars[1].aval.update(shape=(2, 5)))
    else:
        changed.params["dimensions"] = (2,) if defect == "channel-axis" else (0, 0)
    site = _Site("complex layout shape law", 23, operation)
    with pytest.raises(SourcePrecisionError):
        _validate_complex_layout(
            changed,
            [_tensor(v, site) for v in changed.invars],
            [_tensor(v, site) for v in changed.outvars],
            site,
        )


def _physical_bindings(equation: Any) -> dict[int, ir.Value]:
    return {
        id(var): ir.val(
            f"input_{i}",
            dtype=(
                ir.DataType.FLOAT
                if np.dtype(var.aval.dtype) == np.dtype(np.complex64)
                else ir.DataType.DOUBLE
            ),
            shape=(*var.aval.shape, 2),
        )
        for i, var in enumerate(equation.invars)
    }


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    pytest.fail("invalid complex layout reached allocation or emission")


def _drift(value: ir.Value, defect: str) -> None:
    assert value.shape is not None
    shape = tuple(value.shape)
    if defect == "width":
        value.type = ir.TensorType(
            ir.DataType.DOUBLE
            if value.dtype == ir.DataType.FLOAT
            else ir.DataType.FLOAT
        )
    elif defect == "missing-pair":
        value.shape = ir.Shape(shape[:-1])
    elif defect == "extra-pair":
        value.shape = ir.Shape((*shape, 2))
    elif defect == "extent":
        value.shape = ir.Shape((12345, *shape[1:]))
    else:
        value.shape = None


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "defect", ["width", "missing-pair", "extra-pair", "extent", "unknown"]
)
def test_every_member_checks_physical_input_before_allocation(
    operation: str, dtype: str, defect: str
) -> None:
    from jax2onnx.plugins.jax.lax._source_complex_layout import (
        lower_source_complex_layout,
    )

    equation = _equation(operation, dtype)
    bindings = _physical_bindings(equation)
    value = bindings[id(equation.invars[0])]
    _drift(value, defect)
    before = value.type, value.shape
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=lambda v: bindings[id(v)],
        fresh_name=_forbidden,
        allocate_value_for_var=_forbidden,
        bind_value_for_var=_forbidden,
    )
    with pytest.raises(TypeError):
        lower_source_complex_layout(ctx, equation)
    assert (value.type, value.shape) == before


@pytest.mark.parametrize(
    "defect", ["width", "missing-pair", "extra-pair", "extent", "unknown"]
)
def test_second_concatenate_operand_is_checked_before_allocation(defect: str) -> None:
    from jax2onnx.plugins.jax.lax._source_complex_layout import (
        lower_source_complex_layout,
    )

    equation = _equation("concatenate")
    bindings = _physical_bindings(equation)
    second = bindings[id(equation.invars[1])]
    _drift(second, defect)
    before = [(v.type, v.shape) for v in bindings.values()]
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=lambda v: bindings[id(v)],
        fresh_name=_forbidden,
        allocate_value_for_var=_forbidden,
        bind_value_for_var=_forbidden,
    )
    with pytest.raises(TypeError):
        lower_source_complex_layout(ctx, equation)
    assert [(v.type, v.shape) for v in bindings.values()] == before


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_output_descriptor_is_checked_after_allocation_before_emission(
    operation: str,
) -> None:
    from jax2onnx.plugins.jax.lax._source_complex_layout import (
        lower_source_complex_layout,
    )

    equation = _equation(operation)
    bindings = _physical_bindings(equation)
    output = ir.val("bad_output", dtype=ir.DataType.FLOAT, shape=(1,))
    allocated = []

    def allocate(*args: Any, **kwargs: Any) -> ir.Value:
        allocated.append(True)
        return output

    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23, op=_forbidden),
        require_value_for_var=lambda v: bindings[id(v)],
        fresh_name=lambda hint: hint,
        allocate_value_for_var=allocate,
        bind_value_for_var=_forbidden,
        bind_const_for_var=_forbidden,
    )
    with pytest.raises(TypeError):
        lower_source_complex_layout(ctx, equation)
    assert allocated == [True]
    assert output.dtype == ir.DataType.FLOAT and output.shape == ir.Shape((1,))


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_static_dimensions_must_fit_int64_before_binding_or_emission(
    operation: str,
) -> None:
    from jax2onnx.plugins.jax.lax._source_complex_layout import (
        lower_source_complex_layout,
    )

    changed = _changed(_equation(operation))
    huge = 1 << 63
    shape = (1, huge) if operation == "squeeze" else (huge,)
    changed.invars = [Var(ShapedArray(shape, np.complex64)) for _ in changed.invars]
    target_shape = (
        (huge,)
        if operation == "squeeze"
        else (huge * 2,) if operation == "concatenate" else shape
    )
    changed.outvars = [Var(ShapedArray(target_shape, np.complex64))]
    changed.params = {
        "copy": {},
        "reshape": {"new_sizes": shape, "dimensions": None},
        "transpose": {"permutation": (0,)},
        "squeeze": {"dimensions": (0,)},
        "concatenate": {"dimension": 0},
        "rev": {"dimensions": (0,)},
    }[operation]
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=_forbidden,
        fresh_name=_forbidden,
        allocate_value_for_var=_forbidden,
        bind_value_for_var=_forbidden,
    )
    with pytest.raises(SourcePrecisionError):
        lower_source_complex_layout(ctx, changed)


@pytest.mark.parametrize(
    "kind", ["first-real", "second-real", "output-real", "only-output-complex"]
)
def test_any_complex_operand_or_result_selects_strict_family(kind: str) -> None:
    from jax2onnx.plugins.jax.lax._source_complex_layout import (
        lower_source_complex_layout,
    )

    changed = _changed(_equation("concatenate"))
    if kind in {"first-real", "second-real", "only-output-complex"}:
        positions = (
            (0, 1)
            if kind == "only-output-complex"
            else (0,) if kind == "first-real" else (1,)
        )
        for i in positions:
            changed.invars[i] = Var(
                changed.invars[i].aval.update(dtype=np.dtype(np.float32))
            )
    else:
        changed.outvars[0] = Var(
            changed.outvars[0].aval.update(dtype=np.dtype(np.float32))
        )
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=_forbidden,
        fresh_name=_forbidden,
        allocate_value_for_var=_forbidden,
        bind_value_for_var=_forbidden,
    )
    with pytest.raises(SourcePrecisionError):
        lower_source_complex_layout(ctx, changed)


@pytest.mark.parametrize("kind", ["real-pair", "typed-key", "legacy-complex"])
def test_nonfamily_values_return_false_without_builder_access(kind: str) -> None:
    from jax2onnx.plugins.jax.lax._source_complex_layout import (
        lower_source_complex_layout,
    )

    if kind == "typed-key":
        dtype = jax.random.key(17).dtype
        var = SimpleNamespace(aval=SimpleNamespace(dtype=dtype))
    else:
        var = Var(
            ShapedArray((2, 2), np.float32 if kind == "real-pair" else np.complex64)
        )
    equation = SimpleNamespace(
        primitive=jax.lax.copy_p,
        invars=[var],
        outvars=[var],
        params={},
        effects=frozenset(),
    )
    ctx = SimpleNamespace(
        precision_policy=(
            PrecisionPolicy.LEGACY
            if kind == "legacy-complex"
            else PrecisionPolicy.SOURCE
        )
    )
    assert lower_source_complex_layout(ctx, equation) is False


@pytest.mark.parametrize("opset", [21, 24])
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_unselected_layout_opsets_remain_explicit(operation: str, opset: int) -> None:
    original, values, _ = _case(operation, "regular", "complex64")
    trace = jax.make_jaxpr(original)(
        *[jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(trace, opset)


def test_default_complex_stop_gradient_is_not_admitted_by_copy_extension() -> None:
    trace = jax.make_jaxpr(jax.lax.stop_gradient)(
        jax.ShapeDtypeStruct((4,), np.complex64)
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(trace, 23)
