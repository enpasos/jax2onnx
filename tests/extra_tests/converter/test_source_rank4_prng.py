# tests/extra_tests/converter/test_source_rank4_prng.py

"""Exact rank-four native counters/masks for the original conv-ViT dependency."""

from typing import Any

import jax
from jax import export as jax_export
import numpy as np
import onnx
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, Var
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import validate_source_jaxpr
from jax2onnx.plugins.jax.random._source_prng import (
    native_bernoulli_uniform,
    prove_native_uniform,
    validate_selected_random_equation,
)
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)


@pytest.mark.parametrize("x64", (False, True))
@pytest.mark.parametrize("kind", ("bits32", "mask"))
def test_native_rank4_counter_mapping_and_masks_are_bit_exact_on_cpu(
    x64: bool, kind: str
) -> None:
    # The unchanged original14 native trace has (B,7,7,256) UINT32 bits. This
    # witness checks the shared counter/emitter, not full-model acceptance.
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]

        def original(x: Any, words: Any, count: Any) -> Any:
            key = jax.random.fold_in(
                jax.random.wrap_key_data(words, impl="threefry2x32"), count
            )
            if kind == "mask":
                result = jax.random.bernoulli(key, np.float32(0.5), x.shape)
            else:
                result = jax.random.bits(key, x.shape, dtype=np.uint32)
            return result, jax.random.key_data(key), count + np.uint32(1)

        specs = [
            jax.ShapeDtypeStruct((batch, 7, 7, 256), np.float32),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((), np.uint32),
        ]
        graph = jax.make_jaxpr(original)(*specs)
        validate_source_jaxpr(graph, 23)
        if kind == "mask":
            carrier = next(e for e in graph.eqns if e.primitive.name == "jit")
            uniform = native_bernoulli_uniform(carrier)
            assert uniform is not None
            proof = prove_native_uniform(uniform)
            assert proof is not None and proof.shape == (batch, 7, 7, 256)
            validate_selected_random_equation(proof.bits)
        else:
            bits = next(e for e in graph.eqns if e.primitive.name == "random_bits")
            validate_selected_random_equation(bits)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(original)(*specs)
        assert str(activated) == str(graph)
        samples = []
        first_jit = jax.jit(original)
        # Both signed-word extremes and count wraparound exercise actual uint32
        # semantics. Native eager, new JIT and the same warmed JIT agree first.
        for size, count in ((2, 0), (5, 1), (2, 0xFFFFFFFF)):
            values = (
                np.zeros((size, 7, 7, 256), np.float32),
                np.asarray([0xFFFFFFFF, 0x1BD11BDA], np.uint32),
                np.asarray(count, np.uint32),
            )
            expected = tuple(np.asarray(v) for v in original(*values))
            for result in (first_jit(*values), first_jit(*values)):
                for actual, reference in zip(result, expected, strict=True):
                    np.testing.assert_array_equal(actual, reference, strict=True)
            assert expected[0].shape == (size, 7, 7, 256)
            assert expected[0].dtype == np.dtype(
                np.bool_ if kind == "mask" else np.uint32
            )
            assert expected[1].dtype == np.uint32 and expected[1].shape == (2,)
            assert expected[2].dtype == np.uint32 and expected[2].shape == ()
            assert int(expected[2]) == (count + 1) % (1 << 32)
            if kind == "mask":
                assert expected[0].any() and (~expected[0]).any()
            samples.append(SourceSample(values, expected, SampleCriteria("bitwise")))
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert isinstance(model, onnx.ModelProto)
        onnx.checker.check_model(model, full_check=True)
        shape = model.graph.output[0].type.tensor_type.shape.dim
        assert shape[0].dim_param == "B"
        assert tuple(dim.dim_value for dim in shape[1:]) == (7, 7, 256)
        for optimization in ("disabled", "all"):
            outcome = validate_source_samples(
                model, samples=samples, optimization=optimization
            )
            assert outcome.status is CheckStatus.VALIDATED, outcome
            outcome.require_sample_acceptance()


@pytest.mark.parametrize("corruption", ("foreign-scope", "bits-dtype", "width"))
def test_rank4_native_uniform_keeps_exact_scope_dtype_and_bits_schema(
    corruption: str,
) -> None:
    with _runtime_configuration(False):
        batch = jax_export.symbolic_shape("B")[0]

        def original(x: Any, words: Any) -> Any:
            return jax.random.uniform(
                jax.random.wrap_key_data(words, impl="threefry2x32"),
                x.shape,
                dtype=np.float32,
            )

        graph = jax.make_jaxpr(original)(
            jax.ShapeDtypeStruct((batch, 7, 7, 256), np.float32),
            jax.ShapeDtypeStruct((2,), np.uint32),
        )
        uniform = next(e for e in graph.eqns if e.primitive.name == "jit")
        proof = prove_native_uniform(uniform)
        assert proof is not None
        original_trace = str(graph)
        bits = proof.bits
        if corruption == "foreign-scope":
            other_batch = jax_export.symbolic_shape("B")[0]
            changed_bits = bits.replace(
                params={**bits.params, "shape": (other_batch, 7, 7, 256)}
            )
        elif corruption == "bits-dtype":
            changed_bits = bits.replace(
                outvars=[Var(bits.outvars[0].aval.update(dtype=np.dtype(np.uint64)))]
            )
        else:
            changed_bits = bits.replace(params={**bits.params, "bit_width": 64})
        closed = uniform.params["jaxpr"]
        body = closed.jaxpr.replace(
            eqns=[changed_bits if e is bits else e for e in closed.jaxpr.eqns]
        )
        changed = uniform.replace(
            params={**uniform.params, "jaxpr": ClosedJaxpr(body, closed.consts)}
        )
        with pytest.raises(ValueError):
            prove_native_uniform(changed)
        assert str(graph) == original_trace


def test_rank4_counter_limit_rejects_static_int64_overflow_before_emission() -> None:
    with _runtime_configuration(False):

        def original(words: Any) -> Any:
            return jax.random.bits(
                jax.random.wrap_key_data(words, impl="threefry2x32"),
                (2, 7, 7, 256),
                dtype=np.uint32,
            )

        graph = jax.make_jaxpr(original)(jax.ShapeDtypeStruct((2,), np.uint32))
        bits = next(e for e in graph.eqns if e.primitive.name == "random_bits")
        huge_shape = (1 << 62, 7, 7, 256)
        changed = bits.replace(params={**bits.params, "shape": huge_shape})
        with pytest.raises(ValueError, match="signed-int64 physical tensor limit"):
            validate_selected_random_equation(changed)
