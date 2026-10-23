# jax2onnx/plugins/jax/core/platform_index.py

from __future__ import annotations

import numpy as np

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins.jax.core._platform_index_contract import (
    mosaic_default_platform_index,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive="platform_index",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.platform_dependent.html",
    onnx=[],
    since="0.18.0",
    context="primitives.core",
    component="platform_index",
    testcases=[],
)
class PlatformIndexPlugin(PrimitiveLeafPlugin):
    """Resolve only the native Mosaic/default selector in ordinary ONNX lowering."""

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        index = mosaic_default_platform_index(eqn)
        ctx.bind_const_for_var(eqn.outvars[0], np.asarray(index, dtype=np.int32))
