# jax2onnx/plugins/jax/_autodiff_utils.py

from __future__ import annotations

from typing import Any, Callable

from jax.extend.core import Primitive
from jax.interpreters import ad


def register_fallback_jvp_rule(prim: Primitive, impl: Callable[..., Any]) -> None:
    """Register a generic fallback JVP: evaluate primal/tangent via ``impl``."""

    def _jvp_rule(
        primals: tuple[Any, ...], tangents: tuple[Any, ...], **params: Any
    ) -> tuple[Any, Any]:
        tangent_args = tuple(ad.instantiate_zeros(t) for t in tangents)
        primal_out = impl(*primals, **params)
        tangent_out = impl(*tangent_args, **params)
        return primal_out, tangent_out

    ad.primitive_jvps[prim] = _jvp_rule
