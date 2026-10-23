# tests/extra_tests/converter/test_source_static_empty_add.py

"""Static empty native Add retains every port while avoiding target empty Add.

These fixed finite bitwise controls leave original model recipes, callbacks and
missing numerical criteria unchanged.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
from jax.extend.core import jaxpr_as_fun
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Var
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax.add import lower_add
from jax2onnx.ir_utils import numpy_dtype_to_ir
from tests.extra_tests.converter.test_source_mixed_width_contraction import (
    _runtime_payloads,
)
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _body,
    _fresh,
    _observe,
    _port,
    _restore_settings as _restore_settings,
    _same,
    _save,
)

_CASES = [
    ("both-empty", (0,), (0,), "float32", False),
    ("empty-left-scalar", (0,), (), "float32", False),
    ("empty-right-scalar", (), (0,), "float32", False),
    ("left-empty-singletons", (2, 0, 3), (1, 1, 3), "float32", False),
    ("right-empty-singletons", (1, 1, 3), (2, 0, 3), "float32", False),
    ("empty-leading-x64on", (0, 3), (1, 3), "float32", True),
    ("empty-inner-f64", (2, 0, 3), (2, 1, 1), "float64", True),
    ("complementary-left", (2, 0, 1), (1, 1, 3), "float32", False),
    ("complementary-right", (1, 1, 3), (2, 0, 1), "float32", False),
]


def _value(shape: Any, dtype: Any) -> np.ndarray:
    return np.full(shape, 0.5, dtype)


@pytest.mark.parametrize(
    "name,left,right,dtype,x64", _CASES, ids=[v[0] for v in _CASES]
)
def test_native_static_empty_add_keeps_outputs_inputs_and_aliases(
    name: str, left: Any, right: Any, dtype: str, x64: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        a, b = _value(left, dtype), _value(right, dtype)
        wanted = np.add(a, b)
        following = np.add(wanted, np.asarray(0.5, dtype))

        def fn(x: Any, y: Any) -> Any:
            result = jax.lax.add(x, y)
            next_result = jax.lax.add(result, np.asarray(0.5, dtype))
            return result, next_result, result, x, x, y, y

        expected = wanted, following, wanted, a, a, b, b
        with _runtime_payloads(tmp_path):
            model, trace, _ = _observe(
                fn,
                [jax.ShapeDtypeStruct(v.shape, dtype) for v in (a, b)],
                [(jax.device_put(a), jax.device_put(b))],
                [expected],
                tmp_path,
                output_shapes=[v.shape for v in expected],
            )
        body = _body(trace)
        assert sum(e.primitive is jax.lax.add_p for e in body.eqns) == 2
        assert body.outvars[0] is body.outvars[2]
        assert body.outvars[3] is body.outvars[4]
        assert body.outvars[5] is body.outvars[6]
        saved = onnx.load_model_from_string(model.SerializeToString())
        onnx.checker.check_model(saved, full_check=True)
        assert not any(n.op_type == "Add" for n in saved.graph.node)
        if name.startswith("complementary"):
            (constant,) = [
                n for n in saved.graph.node if n.op_type == "ConstantOfShape"
            ]
            assert any(
                attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT
                for attr in constant.attribute
            )
        else:
            assert not any(n.op_type == "ConstantOfShape" for n in saved.graph.node)
        assert any(n.op_type == "Identity" for n in saved.graph.node)
        for port, expected_value in zip(saved.graph.output, expected, strict=True):
            _port(port, dtype, expected_value.shape)
        for port, original in zip(saved.graph.input, (a, b), strict=True):
            _port(port, dtype, original.shape)


@onnx_function(name="NativeStaticEmptyAddLeaf")
def _empty_leaf(x: Any, y: Any) -> Any:
    return jax.lax.add(x, y)


@onnx_function(name="NativeStaticEmptyAddOuter")
def _empty_outer(x: Any, y: Any) -> Any:
    return _empty_leaf(x, y)


@pytest.mark.parametrize("compiled,dtype", [(False, "float32"), (True, "float64")])
def test_actual_named_function_edges_and_jit_empty_add_ports(
    compiled: bool, dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(dtype == "float64"):
        a, b = _value((2, 0, 3), dtype), _value((1, 1, 3), dtype)

        def fn(x: Any, y: Any) -> Any:
            result = jax.jit(_empty_outer)(x, y) if compiled else _empty_outer(x, y)
            return result, result, x, y

        arguments = jax.device_put(a), jax.device_put(b)
        expected = np.add(a, b), np.add(a, b), a, b
        warmed = jax.jit(_fresh(fn))
        for actual, target in zip(warmed(*arguments), expected, strict=True):
            _same(actual, target)
        with _runtime_payloads(tmp_path):
            model, _, built = _observe(
                fn,
                [jax.ShapeDtypeStruct(v.shape, dtype) for v in (a, b)],
                [arguments],
                [expected],
                tmp_path,
                output_shapes=[v.shape for v in expected],
                named=True,
            )
        definitions = {(f.domain, f.name): f for f in model.functions}
        assert len(built) == len(definitions) == 2
        root_calls = {
            (n.domain, n.op_type)
            for n in model.graph.node
            if (n.domain, n.op_type) in definitions
        }
        (outer,) = root_calls
        child_calls = {
            (n.domain, n.op_type)
            for n in definitions[outer].node
            if (n.domain, n.op_type) in definitions
        }
        assert child_calls == set(definitions) - {outer}
        (leaf,) = child_calls
        assert [n.op_type for n in definitions[leaf].node] == ["Identity"]
        for definition, closed in built:
            body = _body(closed)
            validate_source_jaxpr(body, 23)
            serialized = definitions[(definition.domain, definition.name)]
            infos = {v.name: v for v in serialized.value_info}
            for names, variables in (
                (serialized.input, body.invars),
                (serialized.output, body.outvars),
            ):
                for name, variable in zip(names, variables, strict=True):
                    _port(infos[name], variable.aval.dtype, variable.aval.shape)
            if any(e.primitive is jax.lax.add_p for e in body.eqns):
                evaluator = jaxpr_as_fun(closed)
                for role, call in (
                    ("eager", evaluator),
                    ("fresh-jit", jax.jit(_fresh(evaluator))),
                ):
                    result = tuple(call(*arguments))
                    _save(tmp_path / ("leaf-body-" + role + ".npz"), result)
                    _same(result[0], expected[0])
        for role, call in (
            ("eager", fn),
            ("same-warmed", warmed),
            ("fresh-jit", jax.jit(_fresh(fn))),
        ):
            result = tuple(call(*arguments))
            _save(tmp_path / ("post-export-" + role + ".npz"), result)
            for actual, target in zip(result, expected, strict=True):
                _same(actual, target)


def _equation(*, complementary: bool = False, dtype: Any = np.float32) -> Any:
    native = jax.make_jaxpr(jax.lax.add)(
        jax.ShapeDtypeStruct((2, 0, 1) if complementary else (2, 0, 3), dtype),
        jax.ShapeDtypeStruct((1, 1, 3), dtype),
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


def _context(eqn: Any, *, forbidden: bool = False) -> Any:
    emissions, allocations, bindings = [], [], []
    recorded_attributes = []
    values = [
        ir.val(
            f"operand{i}",
            dtype=numpy_dtype_to_ir(np.dtype(v.aval.dtype)),
            shape=v.aval.shape,
        )
        for i, v in enumerate(eqn.invars)
    ]
    output = ir.val(
        "fresh_result",
        dtype=numpy_dtype_to_ir(np.dtype(eqn.outvars[0].aval.dtype)),
        shape=eqn.outvars[0].aval.shape,
    )

    def op(name: str, inputs: Any, attributes: Any = None, *, output: Any) -> Any:
        assert not forbidden, "hostile empty Add reached emission"
        emissions.append((name, tuple(inputs), output))
        recorded_attributes.append(attributes)
        return output

    def allocate(*args: Any, **kwargs: Any) -> Any:
        allocations.append(True)
        return output

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE empty Add reached mutable legacy lookup")

    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23, op=op, Add=unexpected),
        require_value_for_var=lambda v: values[eqn.invars.index(v)],
        allocate_value_for_var=allocate,
        bind_value_for_var=lambda v, value: bindings.append((v, value)),
        get_value_for_var=unexpected,
        fresh_name=lambda name: f"fresh-{name}-{len(emissions)}",
        recorded_attributes=recorded_attributes,
    )
    return context, values, output, emissions, allocations, bindings


@pytest.mark.parametrize(
    "port,fault",
    [
        (0, "dtype"),
        (0, "shape"),
        (0, "missing-type"),
        (1, "dtype"),
        (1, "shape"),
        (1, "missing-shape"),
        (2, "dtype"),
        (2, "shape"),
        (2, "missing-type"),
    ],
)
def test_empty_identity_never_repairs_either_operand_or_output_metadata(
    port: int, fault: str
) -> None:
    eqn = _equation()
    context, values, output, emitted, allocated, bound = _context(eqn)
    lower_add(context, eqn)
    assert emitted == [("Identity", (values[0],), output)]
    assert len(allocated) == 1 and bound == [(eqn.outvars[0], output)]
    assert all(output is not value and output.name != value.name for value in values)
    context, values, output, emitted, allocated, bound = _context(eqn, forbidden=True)
    target = (*values, output)[port]
    if fault == "dtype":
        target.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "missing-type":
        target.type = None
    elif fault == "missing-shape":
        target.shape = None
    else:
        target.shape = ir.Shape((2, 0, 4))
    before = [(v.name, v.type, v.shape) for v in (*values, output)]
    with pytest.raises(ValueError, match="SOURCE empty Add physical"):
        lower_add(context, eqn)
    assert not emitted and not bound
    if port < 2:
        assert not allocated
    assert before == [(v.name, v.type, v.shape) for v in (*values, output)]


@pytest.mark.parametrize(
    "fault",
    ["arity", "parameter", "primitive", "dtype", "shape", "weak", "context", "effects"],
)
def test_empty_native_guards_run_before_ir_access(fault: str) -> None:
    eqn = _equation()
    baseline = _context(eqn)
    lower_add(baseline[0], eqn)
    assert baseline[3]
    original = tuple((v, v.aval) for v in (*eqn.invars, *eqn.outvars))
    if fault == "arity":
        eqn.invars = eqn.invars[:1]
    elif fault == "parameter":
        eqn.params = {"out_dtype": np.dtype(np.float32)}
    elif fault == "primitive":
        eqn.primitive = jax.lax.mul_p
    elif fault in {"dtype", "shape", "weak"}:
        update = (
            {"dtype": np.float64}
            if fault == "dtype"
            else {"shape": (2, 0, 4)} if fault == "shape" else {"weak_type": True}
        )
        eqn.outvars = [Var(eqn.outvars[0].aval.update(**update))]
    elif fault == "effects":
        eqn.effects = frozenset({object()})
    else:
        eqn.ctx = SimpleNamespace(manager=object())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid native empty Add reached IR access")

    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=forbidden,
        allocate_value_for_var=forbidden,
        get_value_for_var=forbidden,
    )
    with pytest.raises(SourcePrecisionError):
        lower_add(context, eqn)
    assert all(v.aval is aval for v, aval in original)


@pytest.mark.parametrize(
    "port,fault", [(0, "shape"), (1, "dtype"), (2, "missing-type")]
)
def test_complementary_broadcast_guards_precede_typed_empty_constant(
    port: int, fault: str
) -> None:
    eqn = _equation(complementary=True)
    context, values, output, emitted, allocated, bound = _context(eqn)
    lower_add(context, eqn)
    assert [entry[0] for entry in emitted] == ["ConstantOfShape"]
    assert len(allocated) == 1 and bound == [(eqn.outvars[0], output)]
    context, values, output, emitted, allocated, bound = _context(eqn, forbidden=True)
    target = (*values, output)[port]
    if fault == "shape":
        target.shape = ir.Shape((2, 0, 2))
    else:
        target.type = (
            None if fault == "missing-type" else ir.TensorType(ir.DataType.DOUBLE)
        )
    before = [(v.name, v.type, v.shape) for v in (*values, output)]
    with pytest.raises(ValueError, match="SOURCE empty Add physical"):
        lower_add(context, eqn)
    assert not emitted and not bound
    if port < 2:
        assert not allocated
    assert before == [(v.name, v.type, v.shape) for v in (*values, output)]


@pytest.mark.parametrize(
    "dtype,complementary",
    [(np.float16, True), (jnp.bfloat16, False), (np.int32, True), (np.uint32, False)],
)
def test_other_existing_native_add_types_keep_exact_empty_storage(
    dtype: Any, complementary: bool
) -> None:
    eqn = _equation(dtype=dtype, complementary=complementary)
    context, values, output, emitted, allocated, bound = _context(eqn)
    original = [(v.name, v.type, v.shape) for v in values]
    lower_add(context, eqn)
    assert [entry[0] for entry in emitted] == (
        ["ConstantOfShape"] if complementary else ["Identity"]
    )
    assert output.dtype == numpy_dtype_to_ir(np.dtype(dtype))
    assert tuple(output.shape) == (2, 0, 3)
    if complementary:
        tensor = context.recorded_attributes[0]["value"].as_tensor()
        assert tensor.dtype == output.dtype
        assert tensor.numpy().dtype == np.dtype(dtype)
        assert not tensor.numpy().any()
    assert len(allocated) == 1 and bound == [(eqn.outvars[0], output)]
    assert original == [(v.name, v.type, v.shape) for v in values]


def test_empty_add_uses_frozen_native_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    eqn = _equation()
    lower_add(_context(eqn)[0], eqn)
    called = []

    def replacement(*args: Any, **kwargs: Any) -> Any:
        called.append(True)
        return eqn.outvars[0].aval, frozenset()

    with monkeypatch.context() as temporary:
        temporary.setitem(vars(jax.lax.add_p), "abstract_eval", replacement)
        lower_add(_context(eqn)[0], eqn)
        eqn.outvars = [Var(eqn.outvars[0].aval.update(weak_type=True))]
        with pytest.raises(SourcePrecisionError):
            lower_add(_context(eqn)[0], eqn)
        assert not called


def test_symbolic_zero_tail_add_retains_existing_profile_exclusion() -> None:
    (batch,) = export.symbolic_shape("B")
    captured = np.empty((3, 0), np.float32)

    def fn(x: Any) -> Any:
        empty = jax.lax.dot_general(x, captured, (((1,), (0,)), ((), ())))
        return jax.lax.add(empty, empty)

    native = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((batch, 3), np.float32))
    assert any(e.primitive is jax.lax.add_p for e in _body(native).eqns)
    with pytest.raises(SourcePrecisionError, match="zero-tail symbolic layout"):
        ui._to_onnx(
            fn,
            [jax.ShapeDtypeStruct((batch, 3), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )


def test_nonempty_add_retains_arithmetic_and_finite_bitwise_oracle(
    tmp_path: Path,
) -> None:
    a, b = np.asarray([0.5, -0.25], np.float32), np.asarray([0.25, 0.5], np.float32)

    def fn(x: Any, y: Any) -> Any:
        return jax.lax.add(x, y), x, y

    expected = np.add(a, b), a, b
    with _runtime_payloads(tmp_path):
        model, _, _ = _observe(
            fn,
            [jax.ShapeDtypeStruct(v.shape, np.float32) for v in (a, b)],
            [(jax.device_put(a), jax.device_put(b))],
            [expected],
            tmp_path,
            output_shapes=[v.shape for v in expected],
        )
    assert sum(n.op_type == "Add" for n in model.graph.node) == 1
