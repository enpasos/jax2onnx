# jax2onnx/plugins/jax/numpy/add.py

from __future__ import annotations

from typing import ClassVar, Final

from jax import core

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching
from jax2onnx.plugins.jax.lax.add import lower_add

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


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
            "post_check_onnx_graph": EG(
                ["Add:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_add_vector",
            "callable": lambda x, y: jnp.add(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Add:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_add_broadcast",
            "callable": lambda x: jnp.add(x, 1.0),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Add:2x3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "add_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.add(y, 1.0))(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpAddPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.numpy.add``."""

    _PRIM: ClassVar = _ADD_PRIM
    _FUNC_NAME: ClassVar[str] = "add"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue, y: core.AbstractValue) -> core.ShapedArray:
        out_shape = tuple(jnp.broadcast_shapes(x.shape, y.shape))
        out_dtype = np.promote_types(x.dtype, y.dtype)
        return jax.core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_add(ctx, eqn)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpAddPlugin._PRIM.def_impl
def _add_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpAddPlugin._PRIM, JnpAddPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


def _add_batch_rule(args, dims, **params):
    return broadcast_batcher_compat(JnpAddPlugin._PRIM, args, dims, **params)


batching.primitive_batchers[JnpAddPlugin._PRIM] = _add_batch_rule
