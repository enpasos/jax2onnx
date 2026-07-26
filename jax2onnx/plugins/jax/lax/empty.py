# jax2onnx/plugins/jax/lax/empty.py

from __future__ import annotations

from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


# JAX 0.11 lowers ``jnp.empty`` through a dedicated ``empty`` primitive instead
# of a zero ``broadcast_in_dim``. Its reference lowering materialises zeros, so
# the ONNX export folds it into a constant initializer.
_EMPTY_PRIM_NAME: Final[str] = getattr(
    getattr(jax.lax, "empty_p", None), "name", "empty"
)


@register_primitive(
    jaxpr_primitive=_EMPTY_PRIM_NAME,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html",
    onnx=[
        {
            "component": "Constant",
            "doc": "https://onnx.ai/onnx/operators/onnx__Constant.html",
        }
    ],
    since="0.15.0",
    context="primitives.lax",
    component="empty",
    testcases=[
        {
            "testcase": "empty_float32",
            "callable": lambda: jnp.empty((2, 3), dtype=jnp.float32),
            "input_shapes": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG([], no_unused_inputs=True),
        },
        {
            "testcase": "empty_int32",
            "callable": lambda: jnp.empty((4,), dtype=jnp.int32),
            "input_shapes": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG([], no_unused_inputs=True),
        },
    ],
)
class EmptyPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        aval = getattr(out_var, "aval", None)
        shape = params.get("shape", getattr(aval, "shape", ()))
        dtype = np.dtype(params.get("dtype", getattr(aval, "dtype", np.float32)))
        if any(not isinstance(dim, (int, np.integer)) for dim in shape):
            raise NotImplementedError(
                "lax.empty with symbolic dimensions is not supported; "
                f"got shape {shape!r}"
            )

        # ``lax.empty``'s reference lowering broadcasts a zero scalar, so the
        # uninitialised buffer is observable as zeros.
        ctx.bind_const_for_var(
            out_var, np.zeros(tuple(int(d) for d in shape), dtype=dtype)
        )
