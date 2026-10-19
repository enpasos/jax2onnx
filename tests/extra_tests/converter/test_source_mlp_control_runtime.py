# tests/extra_tests/converter/test_source_mlp_control_runtime.py

"""Declared MLP control/state boundaries, before numerical runtime approval."""

from typing import Any
from types import SimpleNamespace

from flax import nnx
import jax
from jax import export as jax_export
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx import onnx_function
from jax2onnx import user_interface as ui
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from jax2onnx._compat.jax import (
    ClosedJaxpr,
    Literal,
    ShapedArray,
    Var,
    has_selected_prng_context,
    is_unpartitioned_prng_placement,
)
from jax2onnx.plugins.jax.random._source_prng import (
    NativePRNGExclusion,
    emit_bits,
    emit_fold_in,
    emit_native_bernoulli,
    emit_native_uniform,
    emit_seed,
    native_bernoulli_uniform,
    prove_native_uniform,
    require_native_prng_opset,
    validate_selected_random_equation,
)
from jax2onnx.plugins.jax.core.jit import JitPlugin
from jax2onnx.plugins.jax.random.random_seed import RandomSeedPlugin
from jax2onnx.plugins.flax.nnx.dropout import DropoutPlugin
from tests.extra_tests.converter.test_source_mlp_state_preparation import (
    PreparedMLP,
    _restore_configuration as _restore_configuration,
    original_mlp,
)


@pytest.mark.parametrize("kind", ["nnx", "linen"])
def test_prepared_mlp_declares_every_state_word_and_real_flag(kind: str) -> None:
    prepared = PreparedMLP.from_original(original_mlp(kind), kind)
    B = jax_export.symbolic_shape("B")[0]
    specs = (
        jax.ShapeDtypeStruct((B, 30), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        *(
            jax.ShapeDtypeStruct(value.shape, value.dtype)
            for value in prepared.initial_words
        ),
    )
    graph = jax.make_jaxpr(lambda *args: prepared(*args))(*specs)
    assert len(graph.invars) == 2 + len(prepared.initial_words)
    assert len(graph.outvars) == 1 + len(prepared.initial_words)
    assert graph.invars[1].aval.dtype == jnp.bool_
    assert graph.invars[1].aval.shape == ()
    assert all(var.aval.dtype == jnp.uint32 for var in graph.invars[2:])
    assert all(var.aval.dtype == jnp.uint32 for var in graph.outvars[1:])


@pytest.mark.parametrize("kind", ["nnx", "linen"])
@pytest.mark.parametrize(
    "x64,dtype", [(False, np.float32), (True, np.float32), (True, np.float64)]
)
def test_prepared_original_mlp_source_training_trace_is_audited(
    kind: str, x64: bool, dtype: Any
) -> None:
    with jax.enable_x64(x64):
        prepared = PreparedMLP.from_original(original_mlp(kind), kind)
        B = jax_export.symbolic_shape("B")[0]
        specs = (
            jax.ShapeDtypeStruct((B, 30), dtype),
            jax.ShapeDtypeStruct((), np.bool_),
            *(
                jax.ShapeDtypeStruct(value.shape, value.dtype)
                for value in prepared.initial_words
            ),
        )
        native = jax.make_jaxpr(lambda *args: prepared(*args))(*specs)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(lambda *args: prepared(*args))(*specs)
        validate_source_jaxpr(activated, 23)
        assert str(activated) == str(native)
        assert len(activated.invars) == 2 + len(prepared.initial_words)
        assert len(activated.outvars) == 1 + len(prepared.initial_words)


@pytest.mark.parametrize("kind", ["nnx", "linen"])
@pytest.mark.parametrize(
    "x64,dtype", [(False, np.float32), (True, np.float32), (True, np.float64)]
)
def test_original_mlp_serialization_keeps_parent_batch_binding_in_prng_carrier(
    kind: str, x64: bool, dtype: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("serialization must not construct a target session")

    monkeypatch.setattr("onnxruntime.InferenceSession", forbidden)
    with jax.enable_x64(x64):
        prepared = PreparedMLP.from_original(original_mlp(kind), kind)
        B = jax_export.symbolic_shape("B")[0]
        specs = [
            jax.ShapeDtypeStruct((B, 30), dtype),
            jax.ShapeDtypeStruct((), np.bool_),
            *(
                jax.ShapeDtypeStruct(value.shape, value.dtype)
                for value in prepared.initial_words
            ),
        ]

        def original(x: jax.Array, deterministic: jax.Array, *words: jax.Array) -> Any:
            return prepared(x, deterministic, *words)

        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert isinstance(model, onnx.ModelProto)
        report = get_precision_report(model)
        assert report is not None and report.profile == "source-static-v1"
        checks = {check.category: check.status for check in report.checks}
        assert checks["representation"] is CheckStatus.VALIDATED
        assert checks["lowering"] is CheckStatus.VALIDATED
        assert checks["runtime-execution"] is CheckStatus.UNVERIFIED
        assert len(model.graph.input) == 2 + len(prepared.initial_words)
        assert len(model.graph.output) == 1 + len(prepared.initial_words)
        assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "B"
        assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "B"
        assert any(node.op_type == "If" for node in model.graph.node)


@pytest.mark.parametrize("kind", ["nnx", "linen"])
def test_source_raw_dynamic_model_retains_native_tracer_error(kind: str) -> None:
    target = original_mlp(kind)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.make_jaxpr(lambda x, flag: target(x, deterministic=flag))(
                jax.ShapeDtypeStruct((2, 30), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            )


def test_source_dropout_forwards_explicit_rng_and_retains_missing_rng_error() -> None:
    original = nnx.Dropout.__call__
    wrapped = DropoutPlugin._make_patch(original)
    native = nnx.Dropout(0.1, rngs=None)
    source = nnx.Dropout(0.1, rngs=None)
    x = jnp.ones((2, 20), jnp.float32)
    expected = original(native, x, deterministic=False, rngs=nnx.Rngs(17))
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        actual = wrapped(source, x, deterministic=False, rngs=nnx.Rngs(17))
        with pytest.raises(ValueError, match="no `rngs` argument"):
            wrapped(source, x, deterministic=False)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_unselected_native_key_implementation_fails_audited_validation() -> None:
    graph = jax.make_jaxpr(lambda words: jax.random.wrap_key_data(words, impl="rbg"))(
        jax.ShapeDtypeStruct((4,), np.uint32)
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == "profile-exclusion"


def _native_uniform_graph(dtype: Any) -> tuple[Any, Any]:
    B = jax_export.symbolic_shape("B")[0]

    def original(x: jax.Array, words: jax.Array, low: Any, high: Any) -> jax.Array:
        return jax.random.uniform(
            jax.random.wrap_key_data(words, impl="threefry2x32"),
            x.shape,
            dtype=dtype,
            minval=low,
            maxval=high,
        )

    graph = jax.make_jaxpr(original)(
        jax.ShapeDtypeStruct((B, 20), dtype),
        jax.ShapeDtypeStruct((2,), np.uint32),
        jax.ShapeDtypeStruct((), dtype, weak_type=True),
        jax.ShapeDtypeStruct((), dtype, weak_type=True),
    )
    uniform = next(
        equation for equation in graph.eqns if equation.primitive.name == "jit"
    )
    assert prove_native_uniform(uniform) is not None
    validate_source_jaxpr(graph, 23)
    return graph, uniform


def _native_bernoulli_graph() -> tuple[Any, Any, Any, list[Any]]:
    B = jax_export.symbolic_shape("B")[0]

    def original(x: jax.Array, words: jax.Array, p: jax.Array) -> jax.Array:
        return jax.random.bernoulli(
            jax.random.wrap_key_data(words, impl="threefry2x32"), p, x.shape
        )

    specs = [
        jax.ShapeDtypeStruct((B, 20), np.float32),
        jax.ShapeDtypeStruct((2,), np.uint32),
        jax.ShapeDtypeStruct((), np.float32),
    ]
    graph = jax.make_jaxpr(original)(*specs)
    carrier = next(eqn for eqn in graph.eqns if eqn.primitive.name == "jit")
    return graph, carrier, original, specs


@pytest.mark.parametrize("x64", [False, True])
def test_native_bernoulli_uniform_keeps_configuration_bound_dtypes(x64: bool) -> None:
    with jax.enable_x64(x64):
        graph, carrier, original, specs = _native_bernoulli_graph()
        uniform = native_bernoulli_uniform(carrier)
        assert uniform is not None
        body = uniform.params["jaxpr"].jaxpr
        expected_dtype = np.dtype(np.float64 if x64 else np.float32)
        for formal, actual in zip(body.invars[1:], uniform.invars[1:], strict=True):
            assert isinstance(actual, Literal)
            assert isinstance(actual.aval, ShapedArray)
            assert formal.aval.dtype == actual.aval.dtype == expected_dtype
            assert formal.aval.shape == actual.aval.shape == ()
            assert formal.aval.weak_type is actual.aval.weak_type is True
        for conversion in body.eqns[:2]:
            assert conversion.primitive.name == "convert_element_type"
            assert conversion.params == {
                "new_dtype": np.dtype(np.float32),
                "weak_type": False,
                "sharding": None,
            }
            assert conversion.outvars[0].aval.dtype == np.dtype(np.float32)
        validate_source_jaxpr(graph, 23)
        samples = []
        for batch in (2, 5):
            values = (
                np.zeros((batch, 20), np.float32),
                np.asarray([0xFFFFFFFF, 0x1BD11BDA], np.uint32),
                np.asarray(0.5, np.float32),
            )
            eager = np.asarray(original(*values))
            compiled = np.asarray(jax.jit(original)(*values))
            assert eager.dtype == compiled.dtype == np.dtype(np.bool_)
            assert eager.shape == compiled.shape == (batch, 20)
            assert eager.tobytes() == compiled.tobytes()
            assert eager.any() and (~eager).any()
            samples.append(SourceSample(values, [eager], SampleCriteria("bitwise")))
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(original)(*specs)
        assert str(activated) == str(graph)
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert isinstance(model, onnx.ModelProto)
        onnx.checker.check_model(model, full_check=True)
        assert model.graph.input[2].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.BOOL
        assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "B"
        for optimization in ("disabled", "all"):
            result = validate_source_samples(
                model, samples=samples, optimization=optimization
            )
            assert result.status is CheckStatus.VALIDATED, result
            assert len(result.samples) == 2
            result.require_sample_acceptance()


@pytest.mark.parametrize(
    "corruption",
    [
        "tiny-low",
        "rounded-high",
        "negative-zero",
        "actual-dtype",
        "actual-weakness",
        "payload-dtype",
    ],
)
def test_mixed_uniform_bounds_reject_inexact_payloads_and_metadata(
    corruption: str,
) -> None:
    with jax.enable_x64(True):
        graph, carrier, _, _ = _native_bernoulli_graph()
        uniform = native_bernoulli_uniform(carrier)
        assert uniform is not None
        original = str(graph)
        index = 2 if corruption == "rounded-high" else 1
        before = uniform.invars[index]
        assert before.aval.dtype == np.dtype(np.float64)
        if corruption == "tiny-low":
            payload = np.nextafter(np.float64(0), np.float64(1))
            assert np.float32(payload) == 0
        elif corruption == "rounded-high":
            payload = np.nextafter(np.float64(1), np.float64(2))
            assert np.float32(payload) == 1
        elif corruption == "negative-zero":
            payload = np.float64(-0.0)
        else:
            payload = np.float64(0)
        aval = before.aval
        if corruption == "actual-dtype":
            aval = aval.update(dtype=np.dtype(np.float32))
            payload = np.float32(0)
        elif corruption == "actual-weakness":
            aval = aval.update(weak_type=False)
        elif corruption == "payload-dtype":
            payload = np.float32(0)
        inputs = list(uniform.invars)
        inputs[index] = Literal(payload, aval)
        changed_uniform = uniform.replace(invars=inputs)
        closed = carrier.params["jaxpr"]
        changed_body = closed.jaxpr.replace(
            eqns=[
                changed_uniform if eqn is uniform else eqn for eqn in closed.jaxpr.eqns
            ]
        )
        changed_carrier = carrier.replace(
            params={**carrier.params, "jaxpr": ClosedJaxpr(changed_body, closed.consts)}
        )
        changed_graph = graph.jaxpr.replace(
            eqns=[changed_carrier if eqn is carrier else eqn for eqn in graph.eqns]
        )
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(changed_graph, 23)
        assert caught.value.category == (
            "profile-exclusion"
            if corruption in {"tiny-low", "rounded-high", "negative-zero"}
            else "lowering-defect"
        )
        assert str(graph) == original


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape", [(32,), (31,), (33,), ("B",), (2, 3, 4, 5), ("B", 2, 3, 4)]
)
def test_native_rank_one_uniform_is_limited_to_original_static_dropout_extent(
    dtype: Any,
    shape: tuple[Any, ...],
) -> None:
    with jax.enable_x64(True):
        if shape[0] == "B":
            shape = (jax_export.symbolic_shape("B")[0], *shape[1:])

        def original(x: Any, words: Any, low: Any, high: Any) -> Any:
            return jax.random.uniform(
                jax.random.wrap_key_data(words, impl="threefry2x32"),
                x.shape,
                dtype=dtype,
                minval=low,
                maxval=high,
            )

        graph = jax.make_jaxpr(original)(
            jax.ShapeDtypeStruct(shape, dtype),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((), dtype, weak_type=True),
            jax.ShapeDtypeStruct((), dtype, weak_type=True),
        )
        carrier = next(eqn for eqn in graph.eqns if eqn.primitive.name == "jit")
        if len(shape) == 1 and type(shape[0]) is int and shape[0] == 32:
            proof = prove_native_uniform(carrier)
            assert proof is not None and proof.shape == (32,)
            assert proof.dtype == np.dtype(dtype)
            validate_source_jaxpr(graph, 23)
        else:
            with pytest.raises(SourcePrecisionError) as caught:
                validate_source_jaxpr(graph, 23)
            assert caught.value.category == "profile-exclusion"


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_native_random_bits_keeps_its_own_symbolic_rank_limit(
    symbolic: bool, rank: int
) -> None:
    leading = jax_export.symbolic_shape("B")[0] if symbolic else 2
    shape = (leading, *(3 for _ in range(rank - 1)))

    def original(x: Any, words: Any) -> Any:
        key = jax.random.wrap_key_data(words, impl="threefry2x32")
        return jax.random.bits(key, x.shape, dtype=np.uint32)

    graph = jax.make_jaxpr(original)(
        jax.ShapeDtypeStruct(shape, np.float32),
        jax.ShapeDtypeStruct((2,), np.uint32),
    )
    bits = next(eqn for eqn in graph.eqns if eqn.primitive.name == "random_bits")
    if symbolic and rank not in (2, 3):
        with pytest.raises(NativePRNGExclusion, match="rank-two/rank-three"):
            validate_selected_random_equation(bits)
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(graph, 23)
        assert caught.value.primitive == "random_bits"
        assert caught.value.category == "profile-exclusion"
    else:
        validate_selected_random_equation(bits)
        for opset in (23, 26):
            validate_source_jaxpr(graph, opset)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_native_bernoulli_retains_symbolic_rank_two_and_three(rank: int) -> None:
    batch = jax_export.symbolic_shape("B")[0]

    def original(x: Any, words: Any, probability: Any) -> Any:
        return jax.random.bernoulli(
            jax.random.wrap_key_data(words, impl="threefry2x32"),
            probability,
            x.shape,
        )

    graph = jax.make_jaxpr(original)(
        jax.ShapeDtypeStruct((batch, *(3 for _ in range(rank - 1))), np.float32),
        jax.ShapeDtypeStruct((2,), np.uint32),
        jax.ShapeDtypeStruct((), np.float32),
    )
    if rank in (2, 3):
        for opset in (23, 26):
            validate_source_jaxpr(graph, opset)
    else:
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(graph, 23)
        assert caught.value.category == "profile-exclusion"


@pytest.mark.parametrize(
    "entry",
    [
        "seed",
        "fold-in",
        "bits",
        "uniform",
        "bernoulli",
        "seed-plugin",
        "jit-uniform",
        "jit-bernoulli",
    ],
)
def test_opset16_rejects_every_independent_native_prng_emitter_before_builder_access(
    entry: str,
) -> None:
    if entry in {"uniform", "jit-uniform"}:
        graph, equation = _native_uniform_graph(np.float32)
        emitter = emit_native_uniform
    elif entry in {"bernoulli", "jit-bernoulli"}:
        graph, equation, _, _ = _native_bernoulli_graph()
        validate_source_jaxpr(graph, 23)
        emitter = emit_native_bernoulli
    else:

        def original(seed: Any, words: Any, data: Any) -> Any:
            seeded = jax.random.key(seed, impl="threefry2x32")
            folded = jax.random.fold_in(
                jax.random.wrap_key_data(words, impl="threefry2x32"), data
            )
            return seeded, folded, jax.random.bits(folded, (2, 3), dtype=np.uint32)

        graph = jax.make_jaxpr(original)(
            jax.ShapeDtypeStruct((), np.int32),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((), np.uint32),
        )
        primitive, emitter = {
            "seed": ("random_seed", emit_seed),
            "seed-plugin": ("random_seed", emit_seed),
            "fold-in": ("random_fold_in", emit_fold_in),
            "bits": ("random_bits", emit_bits),
        }[entry]
        equation = next(eqn for eqn in graph.eqns if eqn.primitive.name == primitive)
        validate_selected_random_equation(equation)
    for opset in (23, 26):
        require_native_prng_opset(opset, equation)
    # An incomplete context deliberately has no builder, values or binding hooks.
    # The version diagnostic must precede every one of those emission operations.
    context = SimpleNamespace(opset=16, precision_policy=PrecisionPolicy.SOURCE)
    with pytest.raises(
        SourcePrecisionError, match="selected native PRNG emission"
    ) as caught:
        if entry.startswith("jit-"):
            JitPlugin.lower(None, context, equation)
        elif entry == "seed-plugin":
            RandomSeedPlugin.lower(None, context, equation)
        else:
            emitter(context, equation)
    assert caught.value.opset == 16
    assert caught.value.primitive == equation.primitive.name
    assert caught.value.category == "profile-exclusion"


@onnx_function(name="SourceCapturedFixedKeyInner")
class _CapturedFixedKeyInner:
    def __init__(self, key: Any) -> None:
        self.key = key

    def __call__(self, x: Any, probability: Any) -> Any:
        return jax.random.bernoulli(self.key, probability, x.shape)


@onnx_function(name="SourceCapturedFixedKeyOuter")
class _CapturedFixedKeyOuter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, x: Any, probability: Any) -> Any:
        return self.inner(x, probability)


def _captured_key_function(scope: str, impl: str = "threefry2x32") -> tuple[Any, Any]:
    captured = jax.random.key(17, impl=impl)
    outer = _CapturedFixedKeyOuter(_CapturedFixedKeyInner(captured))

    def original(x: Any, probability: Any, gate: Any) -> Any:
        other = np.asarray(1, np.float32) - probability
        if scope == "function":
            return outer(x, jax.lax.select(gate, probability, other))
        return jax.lax.cond(
            gate, lambda: outer(x, probability), lambda: outer(x, other)
        )

    return original, captured


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scope", ["function", "if"])
def test_native_key_captured_inside_nested_function_keeps_exact_words(
    x64: bool,
    scope: str,
) -> None:
    with jax.enable_x64(x64):
        original, captured = _captured_key_function(scope)
        expected_words = np.asarray([0, 17], np.uint32)
        assert (
            np.asarray(jax.random.key_data(captured)).tobytes()
            == expected_words.tobytes()
        )
        B = jax_export.symbolic_shape("B")[0]
        specs = [
            jax.ShapeDtypeStruct((B, 20), np.float32),
            jax.ShapeDtypeStruct((), np.float32),
            jax.ShapeDtypeStruct((), np.bool_),
        ]
        native = jax.make_jaxpr(original)(*specs)
        assert len(native.constvars) == len(native.consts) == 1
        assert native.consts[0] is captured
        assert isinstance(native.constvars[0].aval, ShapedArray)
        assert native.constvars[0].aval.dtype == captured.dtype
        assert native.constvars[0].aval.shape == ()
        assert native.constvars[0].aval.weak_type is False
        samples = []
        for batch in (2, 5):
            masks = []
            for flag in (False, True):
                values = (
                    np.zeros((batch, 20), np.float32),
                    np.asarray(0.25, np.float32),
                    np.asarray(flag),
                )
                eager = np.asarray(original(*values))
                compiled = np.asarray(jax.jit(original)(*values))
                assert eager.dtype == compiled.dtype == np.dtype(np.bool_)
                assert eager.shape == compiled.shape == (batch, 20)
                assert eager.tobytes() == compiled.tobytes()
                masks.append(eager)
                samples.append(SourceSample(values, [eager], SampleCriteria("bitwise")))
            assert masks[0].tobytes() != masks[1].tobytes()
        model = ui._to_onnx(
            original,
            specs,
            input_names=["features", "probability", "gate"],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )
        assert isinstance(model, onnx.ModelProto)
        onnx.checker.check_model(model, full_check=True)
        assert [value.name for value in model.graph.input] == [
            "features",
            "probability",
            "gate",
        ]
        assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.BOOL
        inner_functions = [
            function
            for function in model.functions
            if function.name.startswith("SourceCapturedFixedKeyInner")
        ]
        assert inner_functions
        for function in inner_functions:
            constants = [
                attribute.t
                for node in function.node
                if node.op_type == "Constant"
                for attribute in node.attribute
                if attribute.type == onnx.AttributeProto.TENSOR
                and attribute.t.data_type == onnx.TensorProto.UINT32
                and tuple(attribute.t.dims) == (2,)
                and onnx.numpy_helper.to_array(attribute.t).tobytes()
                == expected_words.tobytes()
            ]
            assert len(constants) == 1
        if scope == "if":
            assert any(node.op_type == "If" for node in model.graph.node)
        payload = model.SerializeToString()
        for optimization in ("disabled", "all"):
            result = validate_source_samples(
                model, samples=samples, optimization=optimization
            )
            assert result.status is CheckStatus.VALIDATED, result
            assert len(result.samples) == 4
            result.require_sample_acceptance()
        assert model.SerializeToString() == payload
        assert (
            np.asarray(jax.random.key_data(captured)).tobytes()
            == expected_words.tobytes()
        )


@pytest.mark.parametrize("scope", ["function", "if"])
def test_unsupported_key_captured_inside_nested_function_remains_excluded(
    scope: str,
) -> None:
    original, captured = _captured_key_function(scope, "rbg")
    before = np.asarray(jax.random.key_data(captured)).tobytes()
    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            original,
            [
                jax.ShapeDtypeStruct((2, 20), np.float32),
                jax.ShapeDtypeStruct((), np.float32),
                jax.ShapeDtypeStruct((), np.bool_),
            ],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )
    assert caught.value.category == "profile-exclusion"
    assert caught.value.phase == "profile-validation"
    assert caught.value.requested_dtypes == (str(captured.dtype),)
    assert caught.value.opset == 23
    assert "constant 0" in caught.value.location
    assert "SourceCapturedFixedKeyInner" in caught.value.location
    assert "SourceCapturedFixedKeyOuter" in caught.value.location
    assert np.asarray(jax.random.key_data(captured)).tobytes() == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate-low-high",
        "affine-operands",
        "shift",
        "exponent",
        "result-weakness",
        "foreign-scope",
        "extra-equation",
    ],
)
def test_actual_uniform_corruption_fails_mandatory_validation(
    dtype: Any, corruption: str
) -> None:
    with jax.enable_x64(True):
        graph, uniform = _native_uniform_graph(dtype)
        original = str(graph)
        closed = uniform.params["jaxpr"]
        body = closed.jaxpr
        equations = list(body.eqns)
        formals = list(body.invars)
        if corruption == "duplicate-low-high":
            low, high = formals[1:]
            formals[2] = low
            equations = [
                equation.replace(
                    invars=[low if var is high else var for var in equation.invars]
                )
                for equation in equations
            ]
            assert uniform.invars[1] is not uniform.invars[2]
        elif corruption == "affine-operands":
            equations[9] = equations[9].replace(
                invars=list(reversed(equations[9].invars))
            )
        elif corruption in {"shift", "exponent"}:
            index = 5 if corruption == "shift" else 6
            before = equations[index].invars[1]
            value = np.asarray(before.val) + np.asarray(1, dtype=before.aval.dtype)
            literal = Literal(value, before.aval)
            equations[index] = equations[index].replace(
                invars=[equations[index].invars[0], literal]
            )
        elif corruption == "extra-equation":
            equations.append(equations[-1])
        else:
            before = equations[-1].outvars[0]
            aval = (
                before.aval.update(weak_type=True)
                if corruption == "result-weakness"
                else before.aval.update(shape=(jax_export.symbolic_shape("B")[0], 20))
            )
            after = Var(aval)
            equations[-1] = equations[-1].replace(outvars=[after])
        outputs = [equations[-1].outvars[0]]
        changed_body = body.replace(invars=formals, eqns=equations, outvars=outputs)
        changed = uniform.replace(
            params={**uniform.params, "jaxpr": ClosedJaxpr(changed_body, closed.consts)}
        )
        root_equations = [
            changed if equation is uniform else equation for equation in graph.eqns
        ]
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(graph.jaxpr.replace(eqns=root_equations), 23)
        assert str(graph) == original


def test_native_random_bits_nonpartitionable_mode_remains_explicit_gap() -> None:
    previous = jax.config.jax_threefry_partitionable
    try:
        jax.config.update("jax_threefry_partitionable", True)
        graph = jax.make_jaxpr(
            lambda words: jax.random.bits(
                jax.random.wrap_key_data(words, impl="threefry2x32"),
                (2, 20),
                dtype=np.uint32,
            )
        )(jax.ShapeDtypeStruct((2,), np.uint32))
        validate_source_jaxpr(graph, 23)
        jax.config.update("jax_threefry_partitionable", False)
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(graph, 23)
        assert caught.value.category == "profile-exclusion"
    finally:
        jax.config.update("jax_threefry_partitionable", previous)


def test_random_bits_captured_nonpartitionable_context_is_not_ambient_mode() -> None:
    previous = jax.config.jax_threefry_partitionable
    try:
        jax.config.update("jax_threefry_partitionable", False)
        graph = jax.make_jaxpr(
            lambda words: jax.random.bits(
                jax.random.wrap_key_data(words, impl="threefry2x32"),
                (2, 20),
                dtype=np.uint32,
            )
        )(jax.ShapeDtypeStruct((2,), np.uint32))
        bits = next(eqn for eqn in graph.eqns if eqn.primitive.name == "random_bits")
        assert bits.ctx.threefry_partitionable is False
        jax.config.update("jax_threefry_partitionable", True)
        assert not has_selected_prng_context(bits)
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(graph, 23)
        assert caught.value.category == "profile-exclusion"
    finally:
        jax.config.update("jax_threefry_partitionable", previous)


@pytest.mark.parametrize("role", ["key", "bits", "uniform-result", "bernoulli-mask"])
def test_actual_prng_aval_placement_is_checked_inside_native_carriers(
    role: str,
) -> None:
    if role == "bernoulli-mask":
        B = jax_export.symbolic_shape("B")[0]
        graph = jax.make_jaxpr(
            lambda x, words, p: jax.random.bernoulli(
                jax.random.wrap_key_data(words, impl="threefry2x32"), p, x.shape
            )
        )(
            jax.ShapeDtypeStruct((B, 20), np.float32),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((), np.float32),
        )
        carrier = next(eqn for eqn in graph.eqns if eqn.primitive.name == "jit")
    else:
        graph, carrier = _native_uniform_graph(np.float32)
    validate_source_jaxpr(graph, 23)
    original = str(graph)
    closed = carrier.params["jaxpr"]
    body = closed.jaxpr
    target = (
        body.invars[0]
        if role == "key"
        else body.eqns[4].outvars[0] if role == "bits" else body.outvars[0]
    )
    rank = len(target.aval.shape)
    assert is_unpartitioned_prng_placement(target.aval.sharding, rank)
    allowed = NamedSharding(
        AbstractMesh((), ()), PartitionSpec(*(None for _ in range(rank)))
    )
    assert is_unpartitioned_prng_placement(allowed, rank)
    mesh = AbstractMesh((1,), ("device",), axis_types=(AxisType.Explicit,))
    bad = NamedSharding(
        mesh,
        PartitionSpec("device", *([None] * (rank - 1))) if rank else PartitionSpec(),
    )
    assert not is_unpartitioned_prng_placement(bad, rank)
    replacement = Var(target.aval.update(sharding=bad))
    assert replacement.aval.dtype == target.aval.dtype
    assert replacement.aval.shape == target.aval.shape
    assert replacement.aval.weak_type == target.aval.weak_type

    def replace(variables: Any) -> list[Any]:
        return [replacement if var is target else var for var in variables]

    changed_body = body.replace(
        invars=replace(body.invars),
        eqns=[
            equation.replace(
                invars=replace(equation.invars), outvars=replace(equation.outvars)
            )
            for equation in body.eqns
        ],
        outvars=replace(body.outvars),
    )
    changed = carrier.replace(
        params={**carrier.params, "jaxpr": ClosedJaxpr(changed_body, closed.consts)}
    )
    equations = [
        changed if equation is carrier else equation for equation in graph.eqns
    ]
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph.jaxpr.replace(eqns=equations), 23)
    assert caught.value.category == "profile-exclusion"
    assert str(graph) == original


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
@pytest.mark.parametrize("shape", [(2, 0), (0, 20), (2, 0, 3)])
def test_native_empty_prng_grid_serializes_literal_zero_extents(
    dtype: Any, shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("empty-grid serialization must not construct a target session")

    monkeypatch.setattr("onnxruntime.InferenceSession", forbidden)
    with jax.enable_x64(dtype is np.uint64):
        words = np.asarray([0xFFFFFFFF, 0x1BD11BDA], np.uint32)

        def original(words: jax.Array) -> jax.Array:
            key = jax.random.wrap_key_data(words, impl="threefry2x32")
            return jax.random.bits(key, shape, dtype=dtype)

        eager = np.asarray(original(words))
        compiled = np.asarray(jax.jit(original)(words))
        assert eager.shape == compiled.shape == shape
        assert eager.dtype == compiled.dtype == np.dtype(dtype)
        assert eager.tobytes() == compiled.tobytes() == b""
        specs = [jax.ShapeDtypeStruct((2,), np.uint32)]
        graph = jax.make_jaxpr(original)(*specs)
        bits = next(eqn for eqn in graph.eqns if eqn.primitive.name == "random_bits")
        assert bits.params["shape"] == shape
        assert bits.params["bit_width"] == np.dtype(dtype).itemsize * 8
        validate_source_jaxpr(graph, 23)
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert isinstance(model, onnx.ModelProto)
        report = get_precision_report(model)
        assert report is not None and report.profile == "source-static-v1"
        checks = {check.category: check.status for check in report.checks}
        assert checks["representation"] is CheckStatus.VALIDATED
        assert checks["lowering"] is CheckStatus.VALIDATED
        assert checks["runtime-execution"] is CheckStatus.UNVERIFIED
        reshapes = [node for node in model.graph.node if node.op_type == "Reshape"]
        assert len(reshapes) == 1
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in reshapes[0].attribute
        }
        assert attributes["allowzero"] == 1
        output = model.graph.output[0].type.tensor_type
        assert output.elem_type == onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        assert tuple(dimension.dim_value for dimension in output.shape.dim) == shape


def _captured_key_fold_counter(x: Any, key: Any, count: Any) -> Any:
    folded = jax.random.fold_in(key, count)
    return jax.random.key_data(folded), count + np.asarray(1, np.uint32)


@onnx_function(name="SourceCapturedKeyFoldCounter")
def _named_captured_key_fold_counter(x: Any, key: Any, count: Any) -> Any:
    return jax.jit(_captured_key_fold_counter)(x, key, count)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
def test_captured_native_key_root_and_function_jit_if_serialize(
    x64: bool, scoped: bool, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("captured-key serialization must not construct a target session")

    monkeypatch.setattr("onnxruntime.InferenceSession", forbidden)
    with jax.enable_x64(x64):
        captured_native_key = jax.random.key(17, impl="threefry2x32")

        def root(count: Any) -> Any:
            return _captured_key_fold_counter(None, captured_native_key, count)

        def combined(
            x: Any, words: Any, count: Any, gate: Any, *probability: Any
        ) -> Any:
            key = jax.random.wrap_key_data(words, impl="threefry2x32")
            return jax.lax.cond(
                gate,
                lambda args: _named_captured_key_fold_counter(*args),
                lambda args: _named_captured_key_fold_counter(
                    args[0], captured_native_key, *args[2:]
                ),
                (x, key, count, *probability),
            )

        original = combined if scoped else root
        specs = [jax.ShapeDtypeStruct((), np.uint32)]
        names = ["count"]
        if scoped:
            specs = [
                jax.ShapeDtypeStruct((2, 20), np.float32),
                jax.ShapeDtypeStruct((2,), np.uint32),
                *specs,
                jax.ShapeDtypeStruct((), np.bool_),
            ]
            names = ["x", "words", "count", "gate"]
        references = {}
        compiled = jax.jit(original)
        for counter in (0, 0xFFFFFFFF):
            rows = []
            for index, flag in enumerate((True, False, True) if scoped else (None,)):
                values = (np.asarray(counter, np.uint32),)
                if scoped:
                    values = (
                        np.arange(40, dtype=np.float32).reshape(2, 20) / 16,
                        np.asarray([0xFFFFFFFF, 0x1BD11BDA], np.uint32),
                        *values,
                        np.asarray(flag, np.bool_),
                    )
                eager = tuple(np.asarray(value) for value in original(*values))
                jitted = tuple(np.asarray(value) for value in compiled(*values))
                for output, (a, b) in enumerate(zip(eager, jitted, strict=True)):
                    assert a.dtype == b.dtype == np.dtype(np.uint32)
                    assert a.shape == b.shape == ((2,) if output == 0 else ())
                    assert a.tobytes() == b.tobytes()
                    references[f"{counter}-{index}-eager-{output}"] = a
                    references[f"{counter}-{index}-jit-{output}"] = b
                assert int(eager[1]) == (counter + 1) & 0xFFFFFFFF
                for operand, value in enumerate(values):
                    references[f"{counter}-{index}-feed-{operand}"] = value
                rows.append(eager[0])
            if scoped:
                assert rows[0].tobytes() == rows[2].tobytes()
                assert rows[0].tobytes() != rows[1].tobytes()
        references["captured-words"] = np.asarray(
            jax.random.key_data(captured_native_key)
        )
        np.savez(tmp_path / "native-references-before-export.npz", **references)
        native = jax.make_jaxpr(original)(*specs)
        assert len(native.constvars) == 1
        assert native.constvars[0].aval.shape == ()
        assert native.constvars[0].aval.weak_type is False
        assert native.constvars[0].aval.dtype == captured_native_key.dtype
        (tmp_path / "native-before-export.txt").write_text(str(native))
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            source = jax.make_jaxpr(original)(*specs)
        validate_source_jaxpr(source, 23)
        (tmp_path / "source-before-export.txt").write_text(str(source))
        if scoped:
            conditional = next(
                eqn for eqn in source.eqns if eqn.primitive.name == "cond"
            )
            assert all(
                branch.eqns[0].primitive.name.startswith("onnx_fn::")
                for branch in conditional.params["branches"]
            )
        model = ui._to_onnx(
            original,
            specs,
            input_names=names,
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )
        assert isinstance(model, onnx.ModelProto)
        onnx.save(model, tmp_path / "captured-key.onnx")
        report = get_precision_report(model)
        assert report is not None and report.profile == "source-static-v1"
        checks = {check.category: check.status for check in report.checks}
        assert checks["representation"] is checks["lowering"] is CheckStatus.VALIDATED
        assert (
            checks["runtime-execution"]
            is checks["numerical-acceptance"]
            is CheckStatus.UNVERIFIED
        )
        assert [value.name for value in model.graph.input] == names
        captures = [
            tensor.name
            for tensor in model.graph.initializer
            if tensor.data_type == onnx.TensorProto.UINT32
            and tuple(tensor.dims) == (2,)
            and onnx.numpy_helper.to_array(tensor).tobytes()
            == np.asarray([0, 17], np.uint32).tobytes()
        ]
        assert len(captures) == 1
        if scoped:
            assert model.functions
            assert (
                model.graph.input[1].type.tensor_type.elem_type
                == onnx.TensorProto.UINT32
            )
            assert model.graph.input[1].type.tensor_type.shape.dim[0].dim_value == 2
            conditional = next(
                node for node in model.graph.node if node.op_type == "If"
            )
            aliases = {
                node.output[0]: node.input[0]
                for node in model.graph.node
                if node.op_type == "Identity"
            }

            def resolve(name: str) -> str:
                while name in aliases:
                    name = aliases[name]
                return name

            arms = {
                attribute.name: {
                    resolve(name) for node in attribute.g.node for name in node.input
                }
                for attribute in conditional.attribute
                if attribute.type == onnx.AttributeProto.GRAPH
            }
            assert "words" in arms["then_branch"]
            assert captures[0] in arms["else_branch"]
            assert captures[0] not in arms["then_branch"]
            assert "words" not in arms["else_branch"]


@pytest.mark.parametrize("kind", ["other-implementation", "array-key"])
def test_captured_unselected_native_key_fails_before_root_constant_binding(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax2onnx.converter import conversion_api

    key = (
        jax.random.key(17, impl="rbg")
        if kind == "other-implementation"
        else jax.random.split(jax.random.key(17, impl="threefry2x32"), 2)
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unsupported capture must fail before root constant binding")

    monkeypatch.setattr(conversion_api, "_bind_closed_jaxpr_constants", forbidden)
    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(lambda: key, [], precision_policy=PrecisionPolicy.SOURCE, opset=23)
    assert caught.value.category == "profile-exclusion"


@pytest.mark.parametrize(
    "dtype,shape", [(np.uint32, (1,)), (np.uint64, (2,)), (np.float32, (2,))]
)
def test_selected_key_boundary_rejects_raw_malformed_payloads(
    dtype: Any, shape: tuple[int, ...]
) -> None:
    from jax2onnx.converter.ir_context import IRContext
    from jax2onnx.converter.source_representation import representation_for_aval

    key = jax.random.key(17, impl="threefry2x32")
    graph = jax.make_jaxpr(lambda: key)()
    variable = graph.constvars[0]
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    payload = np.zeros(shape, dtype)
    with pytest.raises(ValueError):
        ctx.bind_const_for_var(variable, payload)
    assert not ctx.builder.initializers
    representation = representation_for_aval(variable.aval)
    with pytest.raises(ValueError):
        representation.decode_array(payload)


@pytest.mark.parametrize(
    "policy,dtype",
    [
        (PrecisionPolicy.SOURCE, np.float32),
        (PrecisionPolicy.SOURCE, np.complex64),
        (PrecisionPolicy.LEGACY, np.float32),
    ],
)
def test_root_constant_binding_keeps_nonkey_declared_dtype_canonicalization(
    policy: PrecisionPolicy, dtype: Any
) -> None:
    from jax2onnx.converter.conversion_api import _bind_closed_jaxpr_constants
    from jax2onnx.converter.ir_context import IRContext

    with jax.enable_x64(False):
        payload = (
            [complex(1.25, -0.0), complex(-2.5, 3.75)]
            if dtype is np.complex64
            else [1.25, -0.0]
        )
        declared = np.asarray(payload, dtype)
        graph = jax.make_jaxpr(lambda: declared)()
        ctx = IRContext(
            opset=23, enable_double_precision=False, precision_policy=policy
        )
        _bind_closed_jaxpr_constants(
            ctx,
            graph,
            [payload],
            default_float=np.dtype(np.float32),
            enable_double_precision=False,
        )
        value = ctx.require_value_for_var(graph.constvars[0])
        actual = value.const_value.numpy()
        expected = (
            np.stack((declared.real, declared.imag), axis=-1)
            if dtype is np.complex64
            else declared
        )
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
