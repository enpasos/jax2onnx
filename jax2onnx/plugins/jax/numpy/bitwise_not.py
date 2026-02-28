# jax2onnx/plugins/jax/numpy/bitwise_not.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax.numpy._unary_utils import (
    abstract_eval_via_orig_unary,
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_BITWISE_NOT_PRIM: Final = make_jnp_primitive("jax.numpy.bitwise_not")


@register_primitive(
    jaxpr_primitive=_BITWISE_NOT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_not.html",
    onnx=[
        {
            "component": "Not",
            "doc": "https://onnx.ai/onnx/operators/onnx__Not.html",
        },
        {
            "component": "BitwiseNot",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseNot.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="bitwise_not",
    testcases=[
        {
            "testcase": "jnp_bitwise_not_bool",
            "callable": lambda x: jnp.bitwise_not(x),
            "input_values": [np.array([True, False, True], dtype=np.bool_)],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Not:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_bitwise_not_int",
            "callable": lambda x: jnp.bitwise_not(x),
            "input_values": [np.array([1, 2, 3], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitwiseNot:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "bitwise_not_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.bitwise_not)(x),
            "input_shapes": [(3, 4)],
            "input_dtypes": [np.int32],
        },
    ],
)
class JnpBitwiseNotPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _BITWISE_NOT_PRIM
    _FUNC_NAME: ClassVar[str] = "bitwise_not"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpBitwiseNotPlugin._PRIM,
            JnpBitwiseNotPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("jnp_bitwise_not_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_bitwise_not_out")
        )

        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.int32)
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_bitwise_not_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_bitwise_not_out")

        if np.issubdtype(x_dtype, np.bool_):
            result = ctx.builder.Not(x_val, _outputs=[desired_name])
        else:
            result = ctx.builder.BitwiseNot(x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpBitwiseNotPlugin._PRIM.def_impl
def _bitwise_not_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpBitwiseNotPlugin._PRIM, JnpBitwiseNotPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


register_unary_elementwise_batch_rule(JnpBitwiseNotPlugin._PRIM)
