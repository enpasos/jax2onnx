# jax2onnx/plugins/jax/numpy/add.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax2onnx.plugins.jax.lax.add import lower_add

from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_ADD_PRIM: Final = make_jnp_primitive("jax.numpy.add")


@register_primitive(
    jaxpr_primitive=_ADD_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html",
    onnx=[{"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"}],
    since="v0.8.0",
    context="primitives.jnp",
    component="add",
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x, y: jnp.add(x, y),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "jnp_add_vector",
            "callable": lambda x, y: jnp.add(x, y),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "jnp_add_broadcast",
            "callable": lambda x: jnp.add(x, 1.0),
            "input_shapes": [(2, 3)],
        },
    ],
)
class JnpAddPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.numpy.add``."""

    _PRIM: ClassVar = _ADD_PRIM
    _FUNC_NAME: ClassVar[str] = "add"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, y):
        out_shape = tuple(jnp.broadcast_shapes(x.shape, y.shape))
        out_dtype = np.promote_types(x.dtype, y.dtype)
        return jax.core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_add(ctx, eqn)

    @classmethod
    def binding_specs(cls):
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpAddPlugin._PRIM.def_impl
def _add_impl(*args, **kwargs):
    orig = get_orig_impl(JnpAddPlugin._PRIM, JnpAddPlugin._FUNC_NAME)
    return orig(*args, **kwargs)
