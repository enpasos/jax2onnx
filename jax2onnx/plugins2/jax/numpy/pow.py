from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.jax.lax.pow import lower_pow
from jax2onnx.plugins2.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


def _broadcast_shape(x_shape, y_shape):
    if len(x_shape) != len(y_shape):
        raise ValueError("Power operands must have matching rank for simple broadcast.")
    dims = []
    for xs, ys in zip(x_shape, y_shape):
        if xs == -1 or ys == -1:
            dims.append(xs if ys == -1 else ys)
        elif xs != ys and xs != 1 and ys != 1:
            raise ValueError(f"Shapes {x_shape} and {y_shape} are not broadcastable.")
        else:
            dims.append(xs if ys == 1 else ys)
    return tuple(dims)


class _BaseJnpPow(PrimitiveLeafPlugin):
    _FUNC_NAME: ClassVar[str]

    @staticmethod
    def abstract_eval(x, y):
        shape = _broadcast_shape(x.shape, y.shape)
        return jax.core.ShapedArray(shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_pow(ctx, eqn)

    @classmethod
    def binding_specs(cls):
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


_POWER_PRIM = make_jnp_primitive("jax.numpy.power")


@register_primitive(
    jaxpr_primitive=_POWER_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.0",
    context="primitives2.jnp",
    component="power",
    testcases=[
        {
            "testcase": "jnp_power_vector",
            "callable": lambda x, y: jnp.power(x, y),
            "input_shapes": [(3,), (3,)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpPowerPlugin(_BaseJnpPow):
    _PRIM: ClassVar = _POWER_PRIM
    _FUNC_NAME: ClassVar[str] = "power"


@JnpPowerPlugin._PRIM.def_impl
def _power_impl(*args, **kwargs):
    orig = get_orig_impl(JnpPowerPlugin._PRIM, JnpPowerPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


_POW_PRIM = make_jnp_primitive("jax.numpy.pow")


@register_primitive(
    jaxpr_primitive=_POW_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.0",
    context="primitives2.jnp",
    component="pow",
    testcases=[
        {
            "testcase": "jnp_pow_vector",
            "callable": lambda x, y: jnp.pow(x, y),
            "input_shapes": [(3,), (3,)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpPowPlugin(_BaseJnpPow):
    _PRIM: ClassVar = _POW_PRIM
    _FUNC_NAME: ClassVar[str] = "pow"


@JnpPowPlugin._PRIM.def_impl
def _pow_impl(*args, **kwargs):
    orig = get_orig_impl(JnpPowPlugin._PRIM, JnpPowPlugin._FUNC_NAME)
    return orig(*args, **kwargs)
