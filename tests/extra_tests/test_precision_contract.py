# tests/extra_tests/test_precision_contract.py

"""Precision contract regressions through the ordinary API and existing models."""
from pathlib import Path
import hashlib
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from flax import nnx

import jax2onnx
from jax2onnx import to_onnx, allclose
from jax2onnx._compat.jax import explicit_x64_dtypes
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.examples.linen.mlp_sequential import LinenMLPSequential
from jax2onnx.plugins.examples.nnx.mlp import MLP
from jax2onnx.plugins.examples.nnx.gpt_oss_flax import RMSNorm


def original_inputs(context, component, testcase, dtype):
    is_double = dtype is np.float64
    if is_double:
        testcase += "_f64"
    material = f"{context}|{component}|{testcase}|{int(is_double)}".encode()
    seed = int(hashlib.sha256(material).hexdigest()[:16], 16)
    return (np.random.default_rng(seed).standard_normal((7, 30)) * 0.25).astype(dtype)


def check(fn, args, *, rtol=0, atol=0, specs=None):
    assert (
        Path(jax2onnx.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[2])
    )
    native_args = [
        x if isinstance(x, jax.Array) else jnp.asarray(x, dtype=x.dtype) for x in args
    ]
    eager = jax.tree.map(lambda x: np.asarray(x), fn(*native_args))
    expected, tree = jax.tree.flatten(eager)
    compiled = jax.jit(lambda *xs: fn(*xs))
    candidates = [
        ("fresh JIT", compiled(*native_args)),
        ("warmed JIT", compiled(*native_args)),
    ]
    model = to_onnx(fn, args if specs is None else specs)
    onnx.checker.check_model(model, full_check=True)
    for optimization in [
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None,
            {meta.name: np.asarray(x) for meta, x in zip(session.get_inputs(), args)},
        )
        assert len(actual) == len(expected)
        candidates.append((f"ORT {optimization}", jax.tree.unflatten(tree, actual)))
    for label, candidate in candidates:
        actual, actual_tree = jax.tree.flatten(candidate)
        assert actual_tree == tree
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype
            assert x.shape == y.shape
            if rtol == atol == 0:
                np.testing.assert_array_equal(x, y, err_msg=label)
                if np.issubdtype(y.dtype, np.floating):
                    zeros = y == 0
                    np.testing.assert_array_equal(
                        np.signbit(x[zeros]), np.signbit(y[zeros])
                    )
            else:
                np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=label)
    return model, eager


def test_original_fp32_rounding_and_intentional_fp64_counterpart():
    def source(x):
        rounded = x.astype(jnp.float32)
        return (rounded + np.float32(1)) - rounded, (x + np.float64(1)) - x

    with jax.enable_x64():
        model, reference = check(source, [np.asarray([2**24], np.float64)])
    np.testing.assert_array_equal(reference[0], np.asarray([0], np.float32))
    np.testing.assert_array_equal(reference[1], np.asarray([1], np.float64))
    assert [x.type.tensor_type.elem_type for x in model.graph.output] == [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_existing_full_linen_mlp_explicit_configuration(dtype, tmp_path):
    # Same source, RNG, dimensions, complete input and original numerical limits
    # as simple_linen_mlp_sequential_static[_f64]. Configure the actual module
    # field too: linen_to_nnx's dtype argument controls only its dummy input.
    with jax.enable_x64(dtype is np.float64):
        # Retain the original initialized weights, then migrate their storage
        # and the module configuration explicitly instead of retraining.
        with jax.enable_x64(False):
            original = linen_to_nnx(
                LinenMLPSequential,
                input_shape=(1, 30),
                dtype=np.float32,
                rngs=nnx.Rngs(17),
                din=30,
                dmid=20,
                dout=10,
            )
        if dtype is np.float64:
            configured = partial(LinenMLPSequential, dtype=dtype)
            fn = linen_to_nnx(
                configured,
                input_shape=(1, 30),
                dtype=dtype,
                rngs=nnx.Rngs(17),
                din=30,
                dmid=20,
                dout=10,
            )
            state = jax.tree.map(
                lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x,
                nnx.state(original._model),
            )
            nnx.update(fn._model, state)
        else:
            fn = original
        values = original_inputs(
            "examples.linen",
            "LinenMLPSequential",
            "simple_linen_mlp_sequential_static",
            dtype,
        )
        tolerance = 1e-7 if dtype is np.float64 else 1e-5
        model, reference = check(fn, [values], rtol=tolerance, atol=tolerance)
        assert reference.shape == (7, 10)
        assert reference.dtype == np.dtype(dtype)
        file = tmp_path / "full-linen-mlp.onnx"
        onnx.save_model(model, file)
        valid, message = allclose(
            fn, str(file), [values], rtol=tolerance, atol=tolerance
        )
        assert valid, message


def test_existing_full_nnx_mlp_fp64_migration():
    with jax.enable_x64():
        fn = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(17))
        values = original_inputs("examples.nnx", "MLP", "simple_mlp_static", np.float64)
        _, reference = check(fn, [values], rtol=1e-7, atol=1e-7)
        assert reference.shape == (7, 10)
        assert reference.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_incompatible_input_rejected_without_caller_configuration_change(dtype):
    before = jax.config.x64_enabled
    with jax.enable_x64(False), explicit_x64_dtypes("warn"):
        with pytest.raises(ValueError, match="would be narrowed"):
            to_onnx(lambda x: x, [np.asarray([1], dtype)])
        assert not jax.config.x64_enabled
    assert jax.config.x64_enabled == before


def test_explicitly_allowed_fp64_input_follows_official_tracing():
    with jax.enable_x64(False), explicit_x64_dtypes("allow"):
        spec = jax.ShapeDtypeStruct((1,), np.float64)
        native_signature = jax.eval_shape(lambda x: x, spec)
        if native_signature.dtype != np.float64:
            # Official JAX0.10's abstract input preparation still narrows here.
            # Verify a diagnostic; this is not numerical FP64 qualification.
            with pytest.raises(ValueError, match="would be narrowed"):
                to_onnx(lambda x: x, [spec])
        else:
            _, reference = check(lambda x: x, [np.asarray([2**24 + 1], np.float64)])
            assert reference.dtype == np.float64
            assert reference.item() == 2**24 + 1
        assert not jax.config.x64_enabled


@pytest.mark.parametrize("x64", [False, True])
def test_shape_only_is_fp32_and_export_failure_preserves_context(x64):
    with jax.enable_x64(x64):
        _, reference = check(
            lambda x: x + 1, [np.asarray([2, 3], np.float32)], specs=[(2,)]
        )
        assert reference.dtype == np.float32

        def raises(x):
            raise RuntimeError("model error")

        with pytest.raises(RuntimeError, match="model error"):
            to_onnx(raises, [np.asarray([1], np.float32)])
        assert jax.config.x64_enabled == x64


@pytest.mark.parametrize("flag", [False, True, None])
def test_obsolete_flag_diagnosed_on_export_and_validation(flag):
    with pytest.raises(TypeError, match="configure the JAX model"):
        to_onnx(lambda x: x, [(1,)], enable_double_precision=flag)
    with pytest.raises(TypeError, match="same JAX context"):
        allclose(
            lambda x: x,
            "unused.onnx",
            [np.ones(1, np.float32)],
            enable_double_precision=flag,
        )


@pytest.mark.parametrize("api", ["export", "validation"])
def test_unknown_options_are_rejected_before_model_execution_or_io(api, tmp_path):
    calls = []

    def source(x):
        calls.append("traced")
        return x

    with pytest.raises(TypeError, match="unexpected_option"):
        if api == "export":
            to_onnx(source, [np.ones(1, np.float32)], unexpected_option=True)
        else:
            allclose(
                source,
                str(tmp_path / "does-not-exist.onnx"),
                [np.ones(1, np.float32)],
                unexpected_option=True,
            )
    assert not calls


def test_callable_precision_keyword_remains_a_legitimate_input_param(tmp_path):
    def source(x, *, enable_double_precision):
        return x + enable_double_precision

    inputs = [np.asarray([1], np.float32)]
    params = {"enable_double_precision": np.float32(2)}
    model = to_onnx(source, inputs, input_params=params)
    path = tmp_path / "callable-keyword.onnx"
    onnx.save_model(model, path)
    valid, reason = allclose(
        source, str(path), inputs, input_params=params, rtol=0, atol=0
    )
    assert valid, reason


def test_unchanged_fixed_fp32_gptoss_rmsnorm_on_fp64_input():
    # Original existing supplied class retains intentional FP32 work and casts
    # back to its FP64 input type; no source rewrite or special export wrapper.
    with jax.enable_x64():
        fn = RMSNorm(64)
        values = np.linspace(-1.5, 2.0, 128, dtype=np.float64).reshape(2, 64)
        model, reference = check(fn, [values], rtol=1e-7, atol=1e-7)
        assert reference.dtype == np.float64
        np.testing.assert_array_equal(
            reference, reference.astype(np.float32).astype(np.float64)
        )
        casts = [
            a.i
            for n in [*model.graph.node, *[n for f in model.functions for n in f.node]]
            if n.op_type == "Cast"
            for a in n.attribute
            if a.name == "to"
        ]
        assert onnx.TensorProto.FLOAT in casts
        assert onnx.TensorProto.DOUBLE in casts


def test_step1_weak_and_strong_scalar_witnesses_remain_exact():
    def fn(weak, strong):
        return weak + jnp.asarray(1, dtype=jnp.float16), strong + jnp.asarray(
            1, dtype=jnp.float16
        )

    with jax.enable_x64(False):
        weak = jnp.asarray(2048.0)
        strong = np.asarray(2048.0, np.float32)
        _, reference = check(fn, [weak, strong])
    assert reference[0].dtype == np.float16 and reference[0].item() == 2048
    assert reference[1].dtype == np.float32 and reference[1].item() == 2049


def test_step1_exact_integer_accumulation_witness_remains_exact():
    def fn(lhs, rhs):
        return jax.lax.dot_general(
            lhs, rhs, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32
        )

    lhs = np.full((2, 1041), 127, np.int8)
    rhs = np.full((1041, 3), 127, np.int8)
    _, reference = check(fn, [lhs, rhs])
    np.testing.assert_array_equal(reference, np.full((2, 3), 16790289, np.int32))


@pytest.mark.parametrize("operation", ["where", "concatenate"])
def test_weak_fp64_operand_does_not_widen_strong_fp32(operation):
    def source(weak, strong):
        if operation == "where":
            return jnp.where(jnp.asarray([True, False]), weak, strong)
        return jnp.concatenate((weak, strong))

    with jax.enable_x64():
        weak = jnp.asarray(2048.0).reshape((1,))
        assert weak.weak_type and weak.dtype == np.float64
        _, reference = check(source, [weak, np.asarray([1], np.float32)])
    assert reference.dtype == np.float32


@pytest.mark.parametrize("operation", ["where", "concatenate"])
def test_local_primitive_scope_receives_existing_fp64_formal(operation):
    def source(value):
        with jax.enable_x64(False):
            if operation == "where":
                return jnp.where(jnp.asarray([True, False]), value, value)
            return jnp.concatenate((value, value))

    with jax.enable_x64():
        value = jnp.asarray([1 + 2**-30], dtype=jnp.float64)
        check(source, [value])


@pytest.mark.parametrize("operation", ["add", "maximum", "minimum", "divide"])
def test_numpy_binary_weak_literal_keeps_strong_fp32(operation):
    def source(value):
        return getattr(jnp, operation)(value, 2.0)

    with jax.enable_x64():
        _, reference = check(source, [np.asarray([1, 2**24], np.float32)])
    assert reference.dtype == np.float32


def test_numpy_unary_result_retains_weakness_for_half_consumer():
    def source(value):
        return jnp.cos(value) + jnp.asarray(1, dtype=jnp.float16)

    with jax.enable_x64():
        weak = jnp.asarray(0.0)
        _, reference = check(source, [weak])
    assert reference.dtype == np.float16


@pytest.mark.parametrize(
    "operation",
    ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"],
)
def test_numpy_comparison_promotes_int32_and_fp32_as_jax(operation):
    def source(integer, floating):
        compare = getattr(jnp, operation)
        return compare(integer, floating), compare(floating, integer)

    with jax.enable_x64():
        check(
            source,
            [np.asarray([2**24 + 1], np.int32), np.asarray([2**24], np.float32)],
        )


@pytest.mark.parametrize(
    "operation",
    [
        "outer",
        "select",
        "searchsorted",
        "digitize",
        "histogram",
        "histogram2d",
        "histogramdd",
        "interp",
    ],
)
def test_remaining_numpy_precision_boundaries(operation):
    def source(*values):
        if operation == "outer":
            return jnp.outer(*values)
        if operation == "select":
            return jnp.select([jnp.asarray([True, False])], [values[0]], default=0.0)
        if operation == "searchsorted":
            return jnp.searchsorted(*values, side="left")
        if operation == "digitize":
            bins, queries = values
            return jnp.digitize(queries, bins, right=True)
        if operation == "interp":
            return jnp.interp(*values)
        samples, edges = values
        if operation == "histogram":
            return jnp.histogram(samples, bins=edges)
        if operation == "histogram2d":
            return jnp.histogram2d(samples, samples, bins=[edges, edges])
        return jnp.histogramdd(
            jnp.stack((samples, samples), axis=1), bins=[edges, edges]
        )

    if operation == "outer":
        values = [np.asarray([2**24 + 1], np.int32), np.asarray([1], np.float32)]
    elif operation == "select":
        values = [np.asarray([2048, 2048], np.float16)]
    elif operation in {"searchsorted", "digitize"}:
        values = [np.asarray([2**24], np.float32), np.asarray([2**24 + 1], np.int32)]
    elif operation == "interp":
        values = [
            np.asarray([1], np.float32),
            np.asarray([0, 3], np.float32),
            np.asarray([0, 1], np.float64),
        ]
    else:
        values = [
            np.asarray([2**24 + 1], np.int32),
            np.asarray([2**24 - 2, 2**24], np.float32),
        ]
    with jax.enable_x64():
        check(source, values)


def test_allclose_rejects_reference_dtype_narrowing(tmp_path):
    x = np.asarray([2**24 + 1], np.float64)
    with jax.enable_x64():
        model = to_onnx(lambda y: y, [x])
        path = tmp_path / "identity-f64.onnx"
        onnx.save_model(model, path)
    with jax.enable_x64(False), explicit_x64_dtypes("warn"):
        with pytest.raises(ValueError, match="Reference input dtype"):
            allclose(lambda y: y, str(path), [x])


@pytest.mark.parametrize("compute_dtype", [np.float32, np.float64])
def test_linen_dense_configured_precision_overrides_input_dtype(compute_dtype):
    from flax import linen as nn

    with jax.enable_x64():
        dense = nn.Dense(1, dtype=compute_dtype, param_dtype=np.float32)
        weights = {
            "params": {
                "kernel": jnp.ones((1, 1), dtype=jnp.float32),
                "bias": jnp.ones((1,), dtype=jnp.float32),
            }
        }

        def fn(x):
            return dense.apply(weights, x)

        model, reference = check(fn, [np.asarray([[2**24]], np.float64)])
        assert reference.dtype == np.dtype(compute_dtype)
        assert reference.item() == (2**24 if compute_dtype is np.float32 else 2**24 + 1)
