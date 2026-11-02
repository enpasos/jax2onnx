# tests/extra_tests/converter/test_function_precision_context.py

"""Named functions retain call-site precision context through tracing and reuse."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

import jax2onnx
from jax2onnx import onnx_function, to_onnx
from jax2onnx._compat.jax import TypePromotionError, explicit_x64_dtypes
from jax2onnx.plugins.plugin_system import FunctionPlugin


def _rounding_body(x, bias):
    wide = jnp.asarray(x, dtype=jnp.float64)
    return ((wide + bias) - wide).astype(jnp.float32)


@onnx_function
def _source_rounding(x, *, _jax2onnx_trace_context):
    # The private metadata must not collide with a legitimate static keyword.
    return _rounding_body(x, _jax2onnx_trace_context)


@onnx_function(unique=True)
def _source_unique_rounding(x, *, bias):
    return _rounding_body(x, bias)


@onnx_function
def _source_nested_rounding(x):
    return _source_rounding(x, _jax2onnx_trace_context=1.0)


@onnx_function
def _source_negated_existing(x):
    return jax.lax.neg(x)


@onnx_function
def _source_existing_value_and_default_arithmetic(x):
    default = jnp.asarray(float(2**24))
    rounded = (default + 1.0) - default
    # An accidental x64 override would make rounded == 1 and return +x.
    return jax.lax.select(rounded == 0, jax.lax.neg(x), x)


@pytest.mark.parametrize("with_default_arithmetic", [False, True])
def test_existing_fp64_formal_survives_local_x64_disabled(with_default_arithmetic):
    # Reuse the #277 counterexample: disabling x64 inside the model must not
    # narrow an already supplied FP64 value while retracing a named function.
    def target(x):
        with jax.enable_x64(False):
            if with_default_arithmetic:
                return _source_existing_value_and_default_arithmetic(x)
            return _source_negated_existing(x)

    values = np.asarray(1 + 2**-30, dtype=np.float64)
    with jax.enable_x64():
        model = _assert_exact_references(lambda x: (target(x),), (values,), (-values,))
    for function in model.functions:
        inputs = {
            value.name: value.type.tensor_type.elem_type
            for value in function.value_info
            if value.name in function.input
        }
        assert list(inputs.values()) == [onnx.TensorProto.DOUBLE]
        if with_default_arithmetic:
            types = {
                value.name: value.type.tensor_type.elem_type
                for value in function.value_info
            }
            arithmetic = [
                node for node in function.node if node.op_type in {"Add", "Sub"}
            ]
            assert [node.op_type for node in arithmetic] == ["Add", "Sub"]
            assert all(
                types[node.output[0]] == onnx.TensorProto.FLOAT for node in arithmetic
            )


@onnx_function(unique=True)
def _source_matmul(x):
    return jax.lax.dot_general(x, x, (((1,), (0,)), ((), ())))


@onnx_function
def _source_promotion(x):
    try:
        total = x + jnp.asarray(1.0, dtype=jnp.float32)
    except TypePromotionError:
        # Strict promotion accepts the weak Python scalar and keeps FP16.
        total = x + 1.0
    return (total - x).astype(jnp.float32)


def _assert_exact_references(target, inputs, expected):
    assert (
        Path(jax2onnx.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[3])
    ), jax2onnx.__file__
    eager = target(*(jnp.asarray(value) for value in inputs))
    compiled = jax.jit(lambda *args: target(*args))
    references = (eager, compiled(*inputs), compiled(*inputs))
    for reference in references:
        assert jax.tree.structure(reference) == jax.tree.structure(expected)
        leaves = jax.tree.leaves(reference)
        assert len(leaves) == len(expected)
        for actual, wanted in zip(leaves, expected):
            assert actual.shape == wanted.shape
            assert actual.dtype == wanted.dtype
            np.testing.assert_array_equal(actual, wanted)
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(wanted))
    model = to_onnx(target, inputs, input_names=[f"x{i}" for i in range(len(inputs))])
    onnx.checker.check_model(model, full_check=True)
    for level in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actuals = session.run(None, {f"x{i}": x for i, x in enumerate(inputs)})
        assert len(actuals) == len(expected)
        for actual, wanted in zip(actuals, expected):
            assert actual.shape == wanted.shape
            assert actual.dtype == wanted.dtype
            np.testing.assert_array_equal(
                actual, wanted, err_msg=onnx.printer.to_text(model)
            )
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(wanted))
    return model


@pytest.mark.parametrize(
    "first_x64,kind",
    [(False, "ordinary"), (True, "ordinary"), (False, "unique"), (True, "nested")],
)
def test_same_signature_uses_call_site_x64(first_x64, kind):
    def target(x):
        results = []
        for x64 in (first_x64, not first_x64, first_x64):
            with jax.enable_x64(x64):
                if kind == "ordinary":
                    value = _source_rounding(x, _jax2onnx_trace_context=1.0)
                elif kind == "unique":
                    value = _source_unique_rounding(x, bias=1.0)
                else:
                    value = _source_nested_rounding(x)
                results.append(value)
        return tuple(results)

    x = np.asarray([2**24], dtype=np.float32)
    expected = tuple(
        np.asarray([float(flag)], dtype=np.float32)
        for flag in (first_x64, not first_x64, first_x64)
    )
    with jax.enable_x64(False), jax.default_matmul_precision("default"):
        model = _assert_exact_references(target, (x,), expected)
        assert not jax.config.jax_enable_x64
        assert jax.config.jax_default_matmul_precision == "default"
    assert len(model.functions) == (4 if kind == "nested" else 2)
    assert all(
        output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for output in model.graph.output
    )
    double_casts = [
        function
        for function in model.functions
        if any(
            node.op_type == "Cast"
            and any(
                attr.name == "to" and attr.i == onnx.TensorProto.DOUBLE
                for attr in node.attribute
            )
            for node in function.node
        )
    ]
    assert len(double_casts) == 1


def _equations(closed):
    results = []
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                results.append(equation)
                for parameter in equation.params.values():
                    visit(parameter)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            inner = getattr(value, "jaxpr", None)
            if inner is not None and inner is not value:
                visit(inner)

    visit(closed)
    return results


def _dot_precisions(closed):
    return [
        equation.params["precision"]
        for equation in _equations(closed)
        if equation.primitive.name == "dot_general"
    ]


def test_delayed_dot_trace_retains_precision_and_reuses_equal_context(monkeypatch):
    traced = []
    original = FunctionPlugin._trace_function_body

    def record(self, *args, **kwargs):
        closed = original(self, *args, **kwargs)
        if self.target.__name__ == "_source_matmul":
            traced.append(
                (jax.config.jax_default_matmul_precision, _dot_precisions(closed))
            )
        return closed

    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", record)

    def target(x):
        results = []
        for precision in ("default", "highest", "default"):
            with jax.default_matmul_precision(precision):
                results.append(_source_matmul(x))
        return tuple(results)

    x = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    expected = (np.asarray([[7, 10], [15, 22]], dtype=np.float32),) * 3
    with jax.enable_x64(False), jax.default_matmul_precision("high"):
        model = _assert_exact_references(target, (x,), expected)
        assert jax.config.jax_default_matmul_precision == "high"
    assert len(model.functions) == 2
    assert traced == [
        ("default", [(jax.lax.Precision.DEFAULT,) * 2]),
        ("highest", [(jax.lax.Precision.HIGHEST,) * 2]),
    ]


def test_explicit_x64_policy_distinguishes_same_signature():
    def target(x):
        results = []
        for mode in ("warn", "allow", "warn"):
            with explicit_x64_dtypes(mode):
                results.append(_source_rounding(x, _jax2onnx_trace_context=1.0))
        return tuple(results)

    expected = tuple(np.asarray([value], dtype=np.float32) for value in (0, 1, 0))
    with jax.enable_x64(False), explicit_x64_dtypes("warn"):
        model = _assert_exact_references(
            target, (np.asarray([2**24], dtype=np.float32),), expected
        )
        assert jax.config.jax_explicit_x64_dtypes.name.lower() == "warn"
    assert len(model.functions) == 2


def test_dtype_promotion_distinguishes_same_signature(monkeypatch):
    original = FunctionPlugin._trace_function_body

    def inspect(self, *args, **kwargs):
        closed = original(self, *args, **kwargs)
        if self.target.__name__ == "_source_promotion":
            mode = jax.config.jax_numpy_dtype_promotion
            expected_dtype = np.dtype(np.float16 if mode == "strict" else np.float32)
            arithmetic = [
                equation
                for equation in _equations(closed)
                if equation.primitive.name in ("add", "sub")
            ]
            assert [equation.primitive.name for equation in arithmetic] == [
                "add",
                "sub",
            ]
            assert all(
                output.aval.dtype == expected_dtype
                for equation in arithmetic
                for output in equation.outvars
            ), str(closed)
        return closed

    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", inspect)

    def target(x):
        results = []
        for mode in ("standard", "strict", "standard"):
            with jax.numpy_dtype_promotion(mode):
                results.append(_source_promotion(x))
        return tuple(results)

    expected = tuple(np.asarray([value], dtype=np.float32) for value in (1, 0, 1))
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        model = _assert_exact_references(
            target,
            (np.asarray([2048], dtype=np.float16),),
            expected,
        )
        assert jax.config.jax_numpy_dtype_promotion == "standard"
    assert len(model.functions) == 2

    rounded_functions = []
    for function in model.functions:
        boundaries = [
            node
            for node in function.node
            if node.op_type == "Cast"
            and any(
                attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT16
                for attribute in node.attribute
            )
        ]
        if not boundaries:
            continue
        rounded_functions.append(function)
        producers = {name: node for node in function.node for name in node.output}
        work = [producers[boundary.input[0]] for boundary in boundaries]
        assert [node.op_type for node in work] == ["Add", "Sub"]
        types = {
            value.name: value.type.tensor_type.elem_type
            for value in function.value_info
        }
        assert all(types[node.output[0]] == onnx.TensorProto.FLOAT for node in work)
    assert len(rounded_functions) == 1


def test_delayed_trace_error_restores_precision_settings(monkeypatch):
    original = FunctionPlugin._trace_function_body

    def fail(self, *args, **kwargs):
        if self.target.__name__ == "_source_matmul":
            assert jax.config.jax_enable_x64
            assert jax.config.jax_default_matmul_precision == "highest"
            assert jax.config.jax_explicit_x64_dtypes.name.lower() == "allow"
            assert jax.config.jax_numpy_dtype_promotion == "strict"
            raise RuntimeError("source body failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", fail)

    def target(x):
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            explicit_x64_dtypes("allow"),
            jax.numpy_dtype_promotion("strict"),
        ):
            return _source_matmul(x)

    with (
        jax.enable_x64(False),
        jax.default_matmul_precision("default"),
        explicit_x64_dtypes("warn"),
        jax.numpy_dtype_promotion("standard"),
    ):
        with pytest.raises(RuntimeError, match="source body failure"):
            to_onnx(target, [jax.ShapeDtypeStruct((2, 2), jnp.float32)])
        assert not jax.config.jax_enable_x64
        assert jax.config.jax_default_matmul_precision == "default"
        assert jax.config.jax_explicit_x64_dtypes.name.lower() == "warn"
        assert jax.config.jax_numpy_dtype_promotion == "standard"


def test_half_add_sub_preserves_shared_source_binding():
    def target(x):
        rounded = x + 1.0
        return x, rounded, rounded - x

    x = np.asarray([2048, -0.0], dtype=np.float16)
    expected = (
        x,
        np.asarray([2048, 1], dtype=np.float16),
        np.asarray([0, 1], dtype=np.float16),
    )
    with jax.enable_x64(False), jax.numpy_dtype_promotion("standard"):
        model = _assert_exact_references(target, (x,), expected)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert all(
        output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
        for output in model.graph.output
    )
