# jax2onnx/plugins/jax/numpy/equal.py

from __future__ import annotations

from typing import Any, ClassVar, Final

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
from jax2onnx.plugins.jax.numpy._common import (
    abstract_eval_via_jax,
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
    promoted_dtype_via_jax,
)
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_EQUAL_PRIM: Final = make_jnp_primitive("jax.numpy.equal")


@register_native_primitive(
    jaxpr_primitive=_EQUAL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.equal.html",
    onnx=[
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="equal",
    testcases=[
        {
            "testcase": "jnp_equal_basic",
            "callable": lambda x, y: jnp.equal(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Equal:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_equal_broadcast",
            "callable": lambda x, y: jnp.equal(x, y),
            "input_shapes": [(2, 3), (1, 3)],
            "post_check_onnx_graph": EG(
                ["Equal:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_equal_mixed_dtype",
            "callable": lambda x, y: jnp.equal(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1.0, 2.5, 3.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Equal:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpEqualPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EQUAL_PRIM
    _FUNC_NAME: ClassVar[str] = "equal"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue, y: AbstractValue) -> ShapedArray:
        original = get_orig_impl(JnpEqualPlugin._PRIM, JnpEqualPlugin._FUNC_NAME)
        return abstract_eval_via_jax(original, x, y)

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(
            lhs_var,
            name_hint=ctx.fresh_name("jnp_equal_lhs"),
        )
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("jnp_equal_rhs"),
        )
        out_spec = ctx.allocate_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("jnp_equal_out"),
        )

        target_ir = numpy_dtype_to_ir(
            promoted_dtype_via_jax(lhs_var.aval, rhs_var.aval)
        )
        lhs_cmp = cast_value_to_dtype(
            ctx, lhs_val, target_ir, name_hint="jnp_equal_lhs_cast"
        )
        rhs_cmp = cast_value_to_dtype(
            ctx, rhs_val, target_ir, name_hint="jnp_equal_rhs_cast"
        )
        result = ctx.builder.op("Equal", [lhs_cmp, rhs_cmp], output=out_spec)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpEqualPlugin._PRIM.def_impl
def _equal_impl(x: object, y: object) -> object:
    orig = get_orig_impl(JnpEqualPlugin._PRIM, JnpEqualPlugin._FUNC_NAME)
    return orig(x, y)


def _equal_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> Any:
    return broadcast_batcher_compat(JnpEqualPlugin._PRIM, args, dims, **params)


batching.primitive_batchers[JnpEqualPlugin._PRIM] = _equal_batch_rule
