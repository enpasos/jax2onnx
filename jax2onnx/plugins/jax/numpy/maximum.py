# jax2onnx/plugins/jax/numpy/maximum.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    ShapedArray,
    batching,
)
import jax.numpy as jnp

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    abstract_eval_via_jax,
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_MAXIMUM_PRIM: Final = make_jnp_primitive("jax.numpy.maximum")


@register_native_primitive(
    jaxpr_primitive=_MAXIMUM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.maximum.html",
    onnx=[{"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"}],
    since="0.12.2",
    context="primitives.jnp",
    component="maximum",
    testcases=[
        {
            "testcase": "jnp_maximum_basic",
            "callable": lambda x, y: jnp.maximum(x, y),
            "input_shapes": [(2, 3), (2, 3)],
            "post_check_onnx_graph": EG(["Max:2x3"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_maximum_broadcast_scalar",
            "callable": lambda x: jnp.maximum(x, 0.5),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(["Max:2x3"], no_unused_inputs=True),
        },
    ],
)
class JnpMaximumPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.numpy.maximum``."""

    _PRIM: ClassVar = _MAXIMUM_PRIM
    _FUNC_NAME: ClassVar[str] = "maximum"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue, y: AbstractValue) -> ShapedArray:
        original = get_orig_impl(JnpMaximumPlugin._PRIM, JnpMaximumPlugin._FUNC_NAME)
        return abstract_eval_via_jax(original, x, y)

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(
            lhs_var, name_hint=ctx.fresh_name("maximum_lhs")
        )
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("maximum_rhs"),
        )
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("maximum_out")
        )

        dtype = numpy_dtype_to_ir(out_var.aval.dtype)
        lhs_val = cast_value_to_dtype(ctx, lhs_val, dtype, name_hint="maximum_lhs_cast")
        rhs_val = cast_value_to_dtype(ctx, rhs_val, dtype, name_hint="maximum_rhs_cast")
        result = ctx.builder.op("Max", [lhs_val, rhs_val], output=out_spec)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpMaximumPlugin._PRIM.def_impl
def _maximum_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpMaximumPlugin._PRIM, JnpMaximumPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


register_jvp_via_jax_jvp(JnpMaximumPlugin._PRIM, _maximum_impl)


def _maximum_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpMaximumPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpMaximumPlugin._PRIM] = _maximum_batch_rule
