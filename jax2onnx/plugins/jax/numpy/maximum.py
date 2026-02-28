# jax2onnx/plugins/jax/numpy/maximum.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_MAXIMUM_PRIM: Final = make_jnp_primitive("jax.numpy.maximum")


@register_primitive(
    jaxpr_primitive=_MAXIMUM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.maximum.html",
    onnx=[{"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"}],
    since="0.12.6",
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
    def abstract_eval(x: core.AbstractValue, y: core.AbstractValue) -> core.ShapedArray:
        out_shape = tuple(jnp.broadcast_shapes(x.shape, y.shape))
        out_dtype = np.promote_types(x.dtype, y.dtype)
        return core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(
            lhs_var, name_hint=ctx.fresh_name("maximum_lhs")
        )
        prefer_dtype: np.dtype[Any] = np.dtype(
            getattr(lhs_var.aval, "dtype", np.float32)
        )
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("maximum_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("maximum_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("maximum_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("maximum_out")

        result = ctx.builder.Max(lhs_val, rhs_val, _outputs=[desired_name])
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
