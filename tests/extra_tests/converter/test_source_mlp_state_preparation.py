# tests/extra_tests/converter/test_source_mlp_state_preparation.py

"""Explicit word-state preparation of the unchanged registered MLP calls."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.source_precision_inventory import raw_registrations
from jax2onnx._compat.jax import selected_prng_key_from_words, selected_prng_key_words
from jax2onnx._source_state_preparation import (
    PreparedStatefulCall,
    prepare_linen_call,
    prepare_nnx_call,
)
from jax2onnx.converter.source_representation import representation_for_aval
from tests.extra_tests.converter.source_precision_support import (
    _global_source_configuration,
)


@pytest.fixture(autouse=True)
def _restore_configuration() -> Iterator[None]:
    previous = _global_source_configuration()
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def original_mlp(kind: str) -> Any:
    origin = (
        "example::examples.nnx::MLP"
        if kind == "nnx"
        else "example::examples.linen::LinenMLP"
    )
    registrations, _ = raw_registrations()
    metadata = next(data for name, _, data in registrations if name == origin)
    return metadata["testcases"][2]["callable"].instantiate()


def nonzero_mlp_input(batch: int, dtype: Any = np.float32) -> np.ndarray:
    values = ((np.arange(batch * 30, dtype=np.int32) % 23) - 11) / 16
    return values.astype(dtype).reshape(batch, 30)


def array_snapshot(module: Any) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    leaves = jax.tree_util.tree_leaves(nnx.state(module))
    return tuple(
        (
            str(value.dtype),
            tuple(value.shape),
            np.asarray(
                jax.random.key_data(value)
                if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
                else value
            ).tobytes(),
        )
        for value in leaves
    )


def test_selected_key_representation_retains_native_type_and_two_words() -> None:
    key = jax.random.key(17, impl="threefry2x32")
    representation = representation_for_aval(jax.typeof(key))
    assert (
        representation.logical_dtype is key.dtype
        or representation.logical_dtype == key.dtype
    )
    assert representation.logical_shape == ()
    assert representation.encoding == "threefry-key-words"
    assert representation.physical_dtype == np.dtype(np.uint32)
    assert representation.physical_shape == (2,)
    words = representation.encode_array(key)
    np.testing.assert_array_equal(
        words, np.asarray(jax.random.key_data(key)), strict=True
    )
    restored = representation.decode_array(words)
    assert jax.random.key_impl(restored) == jax.random.key_impl(key)
    np.testing.assert_array_equal(jax.random.key_data(restored), words, strict=True)
    ordinary = representation_for_aval(jax.ShapeDtypeStruct((2,), np.uint32))
    assert ordinary.encoding == "identity" and ordinary.logical_shape == (2,)
    with pytest.raises(ValueError):
        representation.encode_array(words)


@pytest.mark.parametrize("kind", ["array-key", "other-implementation"])
def test_key_representation_rejects_unselected_native_shape_or_implementation(
    kind: str,
) -> None:
    key = (
        jax.random.split(jax.random.key(17), 2)
        if kind == "array-key"
        else jax.random.key(17, impl="rbg")
    )
    with pytest.raises(ValueError):
        representation_for_aval(jax.typeof(key))


@dataclass
class PreparedMLP:
    """Retain established MLP selectors on the production whole-call adapter."""

    original: Any
    kind: str
    prepared: PreparedStatefulCall
    initial_words: tuple[Any, ...]

    @classmethod
    def from_original(cls, original: Any, kind: str) -> "PreparedMLP":
        if kind not in {"nnx", "linen"}:
            raise ValueError("Unknown original MLP framework")
        prepared = (
            prepare_nnx_call(original)
            if kind == "nnx"
            else prepare_linen_call(original)
        )
        words = prepared.initial_words
        if kind == "linen":
            # Preserve the old helper's explicit fixed caller-key port. It is
            # bound unchanged for a whole call and returned without advancing.
            words += (selected_prng_key_words(original._rngs),)
        return cls(original, kind, prepared, words)

    def words_from_model(self, target: Any) -> tuple[Any, ...]:
        words = self.prepared.words_from_target(target)
        if self.kind == "linen":
            words += (selected_prng_key_words(target._rngs),)
        return words

    def fresh_original(self) -> Any:
        return self.prepared.fresh_original()

    def _branch(self, flag: bool, x: Any, *words: Any) -> tuple[Any, ...]:
        if self.kind == "nnx":
            return self.prepared.concrete(flag, x, *words)
        key = selected_prng_key_from_words(words[-1])
        bound = type(self.original)(self.original._model, key)
        prepared = prepare_linen_call(bound)
        return (*prepared.concrete(flag, x, *words[:-1]), words[-1])

    def __call__(self, x: Any, deterministic: Any, *words: Any) -> tuple[Any, ...]:
        if self.kind == "nnx":
            return self.prepared(x, deterministic, *words)
        key = selected_prng_key_from_words(words[-1])
        bound = type(self.original)(self.original._model, key)
        prepared = prepare_linen_call(bound)
        return (*prepared(x, deterministic, *words[:-1]), words[-1])


@pytest.mark.parametrize("kind", ["nnx", "linen"])
@pytest.mark.parametrize("batch", [2, 5])
def test_prepared_mlp_preserves_native_state_sequence(kind: str, batch: int) -> None:
    prepared = PreparedMLP.from_original(original_mlp(kind), kind)
    module = prepared.original if kind == "nnx" else prepared.original._model
    before = array_snapshot(module)
    native = prepared.fresh_original()
    words = prepared.initial_words
    x = jnp.asarray(nonzero_mlp_input(batch))
    for flag in (True, False, False, True):
        reference = native(x, deterministic=flag)
        y, *next_words = prepared(x, jnp.asarray(flag), *words)
        assert y.shape == reference.shape == (batch, 10)
        assert y.dtype == reference.dtype == jnp.float32
        expected_words = prepared.words_from_model(native)
        for actual, expected in zip(next_words, expected_words, strict=True):
            np.testing.assert_array_equal(actual, expected, strict=True)
        words = tuple(next_words)
    assert array_snapshot(module) == before


@pytest.mark.parametrize("kind", ["nnx", "linen"])
def test_original_mlp_dynamic_flag_keeps_native_tracer_error(kind: str) -> None:
    prepared = PreparedMLP.from_original(original_mlp(kind), kind)
    module = prepared.original if kind == "nnx" else prepared.original._model
    before = array_snapshot(module)
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.make_jaxpr(lambda x, flag: prepared.original(x, deterministic=flag))(
            jax.ShapeDtypeStruct((2, 30), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    assert array_snapshot(module) == before
