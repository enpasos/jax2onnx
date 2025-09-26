from __future__ import annotations

import pytest

from jax import lax

from jax2onnx.plugins2.jax.lax.scatter_utils import ensure_supported_mode


@pytest.mark.parametrize(
    "mode",
    [
        None,
        lax.GatherScatterMode.FILL_OR_DROP,
        lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        "fill_or_drop",
        "promiSe_in_bounds",
    ],
)
def test_supported_modes_pass(mode):
    # should not raise
    ensure_supported_mode(mode)


@pytest.mark.parametrize("mode", ["clip", "wrap", lax.GatherScatterMode.CLIP])
def test_unsupported_modes_raise(mode):
    with pytest.raises(NotImplementedError):
        ensure_supported_mode(mode)
