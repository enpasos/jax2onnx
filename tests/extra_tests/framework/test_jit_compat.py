# tests/extra_tests/framework/test_jit_compat.py

from __future__ import annotations

import jax
import numpy as np

from jax2onnx._compat.jax import ClosedJaxpr
from jax2onnx.plugins.jax.core.jit import JitPlugin


def test_jit_freshening_preserves_constants_and_uses_independent_native_variables() -> (
    None
):
    constant = np.asarray([1.0, 2.0], dtype=np.float32)
    original = jax.make_jaxpr(lambda x: x + constant)(
        jax.ShapeDtypeStruct((2,), np.float32)
    )

    first = JitPlugin._freshen_closed_jaxpr(original)
    second = JitPlugin._freshen_closed_jaxpr(original)

    assert isinstance(first, ClosedJaxpr)
    assert isinstance(second, ClosedJaxpr)
    for cloned in (first, second):
        assert len(cloned.consts) == len(original.consts) == 1
        np.testing.assert_array_equal(cloned.consts[0], original.consts[0])
        assert cloned.jaxpr.eqns[0].primitive is original.jaxpr.eqns[0].primitive
        assert cloned.jaxpr.eqns[0].ctx is original.jaxpr.eqns[0].ctx
        assert cloned.jaxpr.eqns[0].invars == [
            cloned.jaxpr.invars[0],
            cloned.jaxpr.constvars[0],
        ]
        assert cloned.jaxpr.eqns[0].outvars == cloned.jaxpr.outvars
    for source, left, right in zip(
        (
            *original.jaxpr.constvars,
            *original.jaxpr.invars,
            *original.jaxpr.outvars,
        ),
        (*first.jaxpr.constvars, *first.jaxpr.invars, *first.jaxpr.outvars),
        (*second.jaxpr.constvars, *second.jaxpr.invars, *second.jaxpr.outvars),
        strict=True,
    ):
        assert source is not left
        assert source is not right
        assert left is not right
        assert source.aval is left.aval is right.aval
