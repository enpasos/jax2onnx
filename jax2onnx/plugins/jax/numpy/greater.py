# jax2onnx/plugins/jax/numpy/greater.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    ShapedArray,
    batching,
)
import jax.numpy as jnp
import numpy as np

from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    abstract_eval_via_jax,
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
    promoted_dtype_via_jax,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_GREATER_PRIM: Final = make_jnp_primitive("jax.numpy.greater")


@register_native_primitive(
    jaxpr_primitive=_GREATER_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.greater.html",
    onnx=[
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="greater",
    testcases=[
        {
            "testcase": "jnp_greater_basic",
            "callable": lambda x, y: jnp.greater(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Greater:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_greater_broadcast",
            "callable": lambda x, y: jnp.greater(x, y),
            "input_shapes": [(2, 3), (1, 3)],
            "post_check_onnx_graph": EG(
                ["Greater:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_greater_mixed_dtype",
            "callable": lambda x, y: jnp.greater(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1.0, 2.5, 3.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Greater:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "greater_vmap_batching",
            "callable": lambda x, y: jax.vmap(jnp.greater)(x, y),
            "input_shapes": [(3, 4), (3, 4)],
        },
    ],
)
class JnpGreaterPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _GREATER_PRIM
    _FUNC_NAME: ClassVar[str] = "greater"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue, y: AbstractValue) -> ShapedArray:
        original = get_orig_impl(JnpGreaterPlugin._PRIM, JnpGreaterPlugin._FUNC_NAME)
        return abstract_eval_via_jax(original, x, y)

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(
            lhs_var,
            name_hint=ctx.fresh_name("jnp_greater_lhs"),
        )
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("jnp_greater_rhs"),
        )
        out_spec = ctx.allocate_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("jnp_greater_out"),
        )

        target_ir = numpy_dtype_to_ir(
            promoted_dtype_via_jax(lhs_var.aval, rhs_var.aval)
        )
        lhs_cmp = cast_value_to_dtype(
            ctx, lhs_val, target_ir, name_hint="jnp_greater_lhs_cast"
        )
        rhs_cmp = cast_value_to_dtype(
            ctx, rhs_val, target_ir, name_hint="jnp_greater_rhs_cast"
        )
        result = ctx.builder.op("Greater", [lhs_cmp, rhs_cmp], output=out_spec)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpGreaterPlugin._PRIM.def_impl
def _greater_impl(x: object, y: object) -> object:
    orig = get_orig_impl(JnpGreaterPlugin._PRIM, JnpGreaterPlugin._FUNC_NAME)
    return orig(x, y)


def _greater_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpGreaterPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpGreaterPlugin._PRIM] = _greater_batch_rule
