# jax2onnx/plugins/jax/numpy/pow.py

from __future__ import annotations

from typing import ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.pow import lower_pow
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _broadcast_shape(x_shape, y_shape):
    try:
        return tuple(np.broadcast_shapes(tuple(x_shape), tuple(y_shape)))  # type: ignore[arg-type]
    except ValueError:
        max_rank = max(len(x_shape), len(y_shape))
        x = (1,) * (max_rank - len(x_shape)) + tuple(x_shape)
        y = (1,) * (max_rank - len(y_shape)) + tuple(y_shape)
        dims = []
        for xs, ys in zip(x, y):
            if xs == ys:
                dims.append(xs)
            elif xs == 1:
                dims.append(ys)
            elif ys == 1:
                dims.append(xs)
            elif xs == -1 or ys == -1:
                dims.append(xs if ys == 1 or ys == -1 else ys)
            else:
                raise ValueError(
                    f"Shapes {x_shape} and {y_shape} are not broadcastable."
                )
        # Remove any leading broadcast dims we artificially introduced when both were 1.
        while dims and dims[0] == 1 and max(len(x_shape), len(y_shape)) > 0:
            dims = dims[1:]
        return tuple(dims) if dims else (1,)


class _BaseJnpPow(PrimitiveLeafPlugin):
    _FUNC_NAME: ClassVar[str]

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue, y: jax.core.AbstractValue):
        shape = _broadcast_shape(x.shape, y.shape)
        return jax.core.ShapedArray(shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_pow(ctx, eqn)

    @classmethod
    def binding_specs(cls):
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


_POWER_PRIM: Final = make_jnp_primitive("jax.numpy.power")


@register_primitive(
    jaxpr_primitive=_POWER_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.0",
    context="primitives.jnp",
    component="power",
    testcases=[
        {
            "testcase": "jnp_power_vector",
            "callable": lambda x, y: jnp.power(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_jnp_power",
            "callable": lambda: jnp.power(
                np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.0
            ),
            "input_values": [],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2.0}},
                        "path": "Pow:3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "power_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.power(y, 2.0))(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpPowerPlugin(_BaseJnpPow):
    _PRIM: ClassVar = _POWER_PRIM
    _FUNC_NAME: ClassVar[str] = "power"


@JnpPowerPlugin._PRIM.def_impl
def _power_impl(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpPowerPlugin._PRIM, JnpPowerPlugin._FUNC_NAME)
    return orig(x, y)


_POW_PRIM: Final = make_jnp_primitive("jax.numpy.pow")


@register_primitive(
    jaxpr_primitive=_POW_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.0",
    context="primitives.jnp",
    component="pow",
    testcases=[
        {
            "testcase": "jnp_pow_vector",
            "callable": lambda x, y: jnp.pow(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_jnp_pow",
            "callable": lambda: jnp.pow(
                np.array([1.0, 2.0, 3.0], dtype=np.float32), 3.0
            ),
            "input_values": [],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 3.0}},
                        "path": "Pow:3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.pow(y, 2.0))(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpPowPlugin(_BaseJnpPow):
    _PRIM: ClassVar = _POW_PRIM
    _FUNC_NAME: ClassVar[str] = "pow"


@JnpPowPlugin._PRIM.def_impl
def _pow_impl(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpPowPlugin._PRIM, JnpPowPlugin._FUNC_NAME)
    return orig(x, y)


def _make_pow_batch_rule(prim):
    def _batch_rule(args, dims, **params):
        return broadcast_batcher_compat(prim, args, dims, **params)

    return _batch_rule


batching.primitive_batchers[JnpPowerPlugin._PRIM] = _make_pow_batch_rule(
    JnpPowerPlugin._PRIM
)
batching.primitive_batchers[JnpPowPlugin._PRIM] = _make_pow_batch_rule(
    JnpPowPlugin._PRIM
)
