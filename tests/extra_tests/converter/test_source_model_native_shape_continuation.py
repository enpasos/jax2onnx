# tests/extra_tests/converter/test_source_model_native_shape_continuation.py

"""Shared shape provenance and native SpectralNorm; original full owners stay separate."""

from typing import Any

import jax
from jax import export
import numpy as np
import pytest

from jax2onnx._compat.jax import atomic_symbolic_dimension
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import validate_source_jaxpr
from jax2onnx.converter.source_representation import (
    ValueRepresentation,
    native_shape_contract,
    source_shape_contract,
    validate_native_shapes,
    validate_source_shapes,
)
from jax2onnx.plugins.flax.linen.spectral_norm import SpectralNormPlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _fresh,
    _observe,
    _restore_settings as _restore_settings,
    _specs,
)


@pytest.mark.parametrize("rank,axis", [(r, a) for r in (5, 6) for a in range(r)])
def test_native_decoder_retains_actual_atomic_scope_and_each_axis(
    rank: int, axis: int
) -> None:
    (batch,) = export.symbolic_shape("B")
    shape = tuple(batch if i == axis else 2 for i in range(rank))
    contract = native_shape_contract(shape)
    assert contract.shape == tuple("B" if i == axis else 2 for i in range(rank))
    assert contract.symbol is not None and contract.symbol[0] == "B"
    assert (
        contract.symbol[1] is atomic_symbolic_dimension(batch)[1]
        and contract.symbol_axis == axis
    )


@pytest.mark.parametrize("rank", [5, 6])
def test_source_decoder_keeps_leading_batch_and_positive_static_tails(
    rank: int,
) -> None:
    (batch,) = export.symbolic_shape("B")
    for first in (batch, "B"):
        normalized, symbol = source_shape_contract(
            (first, *(2 for _ in range(rank - 1)))
        )
        assert normalized == ("B", *(2 for _ in range(rank - 1)))
        assert symbol is not None and symbol[0] == "B"
        assert symbol[1] is (
            None if isinstance(first, str) else atomic_symbolic_dimension(batch)[1]
        )


@pytest.mark.parametrize("rank", [5, 6])
def test_shared_higher_rank_scope_is_identical_not_just_same_label(rank: int) -> None:
    (batch,) = export.symbolic_shape("B")
    (foreign,) = export.symbolic_shape("B")
    left = (batch, *(2 for _ in range(rank - 1)))
    right = (batch, *(3 for _ in range(rank - 1)))
    assert (
        atomic_symbolic_dimension(batch)[1] is not atomic_symbolic_dimension(foreign)[1]
    )
    assert validate_native_shapes((left, right)) is atomic_symbolic_dimension(batch)[1]
    assert validate_source_shapes((left, right)) is atomic_symbolic_dimension(batch)[1]
    for validator in (validate_native_shapes, validate_source_shapes):
        with pytest.raises(ValueError, match="scope"):
            validator((left, (foreign, *right[1:])))


@pytest.mark.parametrize(
    "fault",
    ["text", "repeated", "expression", "unknown", "bool", "negative", "foreign-label"],
)
def test_higher_rank_native_decoder_rejects_false_dimensions(fault: str) -> None:
    batch, other = export.symbolic_shape("B, C")
    first: Any = batch
    tail: tuple[Any, ...] = (2, 3, 1, 4)
    if fault == "text":
        first = "B"
    elif fault == "repeated":
        tail = (2, batch, 1, 4)
    elif fault == "expression":
        first = 2 * batch
    elif fault == "unknown":
        tail = (2, None, 1, 4)
    elif fault == "bool":
        tail = (2, True, 1, 4)
    elif fault == "negative":
        tail = (2, -1, 1, 4)
    else:
        tail = (2, other, 1, 4)
    with pytest.raises(ValueError):
        native_shape_contract((first, *tail))


@pytest.mark.parametrize(
    "fault",
    ["nonleading", "repeated", "expression", "zero", "bool", "negative", "unknown"],
)
def test_higher_rank_source_declaration_still_rejects_invalid_layout(
    fault: str,
) -> None:
    (batch,) = export.symbolic_shape("B")
    shapes = {
        "nonleading": (2, batch, 3, 1, 4),
        "repeated": (batch, batch, 3, 1, 4),
        "expression": (2 * batch, 2, 3, 1, 4),
        "zero": (batch, 0, 3, 1, 4),
        "bool": (batch, True, 3, 1, 4),
        "negative": (batch, -1, 3, 1, 4),
        "unknown": (batch, None, 3, 1, 4),
    }
    with pytest.raises(ValueError):
        source_shape_contract(shapes[fault])


@pytest.mark.parametrize("rank", [5, 6])
def test_higher_rank_decoder_does_not_widen_NCHW(rank: int) -> None:
    (batch,) = export.symbolic_shape("B")
    with pytest.raises(ValueError, match="logical rank four"):
        ValueRepresentation(
            np.dtype("float32"),
            (batch, *(2 for _ in range(rank - 1))),
            "identity",
            "NCHW",
        )


def _chain(rank: int, x: Any) -> Any:
    b = x.shape[0]
    shape = (b, 2, 3, 1, 4) if rank == 5 else (b, 2, 3, 1, 2, 2)
    expanded = (*shape[:3], 5, *shape[4:])
    grouped = jax.lax.reshape(x, shape)
    repeated = jax.lax.broadcast_in_dim(grouped, expanded, tuple(range(rank)))
    moved = jax.lax.transpose(repeated, (*range(1, rank), 0))
    restored = jax.lax.transpose(moved, (rank - 1, *range(rank - 1)))
    return jax.lax.reshape(restored, (b, 6, 20)), x


@pytest.mark.parametrize("rank", [5, 6])
def test_actual_GPT_extent_ranked_reshape_keeps_native_typed_contract(
    rank: int,
) -> None:
    (batch,) = export.symbolic_shape("B")
    shape = (batch, 1024, 12, 1, 64) if rank == 5 else (batch, 1024, 12, 1, 8, 8)

    def target(x: Any) -> Any:
        return jax.lax.reshape(x, shape)

    spec = jax.ShapeDtypeStruct((batch, 1024, 768), np.float32)
    native = jax.make_jaxpr(_fresh(target))(spec)
    validate_source_jaxpr(native, 23)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(target))(spec)
    assert typed_trace_manifest(activated) == typed_trace_manifest(native)
    contract = native_shape_contract(shape)
    assert (
        contract.symbol is not None
        and contract.symbol[1] is atomic_symbolic_dimension(batch)[1]
    )
    assert contract.symbol_axis == 0


@pytest.mark.parametrize("rank", [5, 6])
@pytest.mark.parametrize("caller_x64", [False, True])
def test_higher_rank_reshape_broadcast_transpose_transport_one_symbolic_artifact(
    rank: int,
    caller_x64: bool,
    tmp_path: Any,
) -> None:
    with jax.enable_x64(caller_x64):

        def target(x: Any) -> Any:
            return _chain(rank, x)

        feeds, expected = [], []
        for batch in (2, 5):
            value = (
                np.arange(batch * 24, dtype=np.float32).reshape(batch, 6, 4)
                - np.float32(13)
            ) / np.float32(8)
            shape = (batch, 2, 3, 1, 4) if rank == 5 else (batch, 2, 3, 1, 2, 2)
            expanded = (*shape[:3], 5, *shape[4:])
            reference = (
                np.broadcast_to(value.reshape(shape), expanded)
                .reshape(batch, 6, 20)
                .copy()
            )
            feeds.append((value,))
            expected.append((reference, value.copy()))
        # Existing exact bitwise transport criterion and both native/CPU profiles;
        # finite full original model tolerances remain independently frozen67cd.
        _observe(
            target,
            _specs([("B", 6, 4)], "float32"),
            feeds,
            expected,
            tmp_path,
            output_shapes=[("B", 6, 20), ("B", 6, 4)],
        )


@pytest.mark.parametrize("update_stats", [False, True])
def test_spectral_SOURCE_passes_exact_original_receiver_arguments_and_flags(
    update_stats: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver, argument, option, token = object(), object(), object(), object()
    calls = []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return token

    with monkeypatch.context() as temporary:
        temporary.setattr(
            SpectralNormPlugin, "_ORIGINAL_CALL", SpectralNormPlugin._ORIGINAL_CALL
        )
        patched = SpectralNormPlugin._make_patch(original)
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            assert (
                patched(
                    receiver, argument, update_stats=update_stats, method_option=option
                )
                is token
            )
    assert len(calls) == 1 and calls[0][0] == (receiver, argument)
    assert calls[0][1]["update_stats"] is update_stats
    assert calls[0][1]["method_option"] is option
