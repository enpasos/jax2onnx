# jax2onnx/plugins/jax/numpy/invert.py

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


_INVERT_PRIM: Final = make_jnp_primitive("jax.numpy.invert")


@register_primitive(
    jaxpr_primitive=_INVERT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.invert.html",
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
    since="0.12.7",
    context="primitives.jnp",
    component="invert",
    testcases=[
        {
            "testcase": "jnp_invert_bool",
            "callable": lambda x: jnp.invert(x),
            "input_values": [np.array([True, False, True], dtype=np.bool_)],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Not:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_invert_int",
            "callable": lambda x: jnp.invert(x),
            "input_values": [np.array([1, 2, 3], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitwiseNot:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "invert_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.invert)(x),
            "input_shapes": [(3, 4)],
            "input_dtypes": [np.int32],
        },
    ],
)
class JnpInvertPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _INVERT_PRIM
    _FUNC_NAME: ClassVar[str] = "invert"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpInvertPlugin._PRIM,
            JnpInvertPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_invert_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_invert_out")
        )

        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.int32)
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_invert_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_invert_out")

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


@JnpInvertPlugin._PRIM.def_impl
def _invert_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpInvertPlugin._PRIM, JnpInvertPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


register_unary_elementwise_batch_rule(JnpInvertPlugin._PRIM)
