# jax2onnx/plugins/jax/numpy/less_equal.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.interpreters import batching

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LESS_EQUAL_PRIM: Final = make_jnp_primitive("jax.numpy.less_equal")


@register_primitive(
    jaxpr_primitive=_LESS_EQUAL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.less_equal.html",
    onnx=[
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        }
    ],
    since="0.12.7",
    context="primitives.jnp",
    component="less_equal",
    testcases=[
        {
            "testcase": "jnp_less_equal_basic",
            "callable": lambda x, y: jnp.less_equal(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["LessOrEqual:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_less_equal_broadcast",
            "callable": lambda x, y: jnp.less_equal(x, y),
            "input_shapes": [(2, 3), (1, 3)],
            "post_check_onnx_graph": EG(
                ["LessOrEqual:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_less_equal_mixed_dtype",
            "callable": lambda x, y: jnp.less_equal(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1.0, 2.5, 3.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> LessOrEqual:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "less_equal_vmap_batching",
            "callable": lambda x, y: jax.vmap(jnp.less_equal)(x, y),
            "input_shapes": [(3, 4), (3, 4)],
        },
    ],
)
class JnpLessEqualPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LESS_EQUAL_PRIM
    _FUNC_NAME: ClassVar[str] = "less_equal"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue, y: core.AbstractValue) -> core.ShapedArray:
        out_shape = tuple(jnp.broadcast_shapes(x.shape, y.shape))
        return core.ShapedArray(out_shape, np.dtype(np.bool_))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(
            lhs_var,
            name_hint=ctx.fresh_name("jnp_less_equal_lhs"),
        )
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("jnp_less_equal_rhs"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("jnp_less_equal_out"),
        )

        lhs_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(lhs_var, "aval", None), "dtype", np.float32)
        )
        rhs_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(rhs_var, "aval", None), "dtype", np.float32)
        )
        target_dtype: np.dtype[Any] = np.promote_types(lhs_dtype, rhs_dtype)
        target_ir = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)

        lhs_cmp = lhs_val
        if lhs_dtype != target_dtype:
            lhs_cmp = ctx.builder.Cast(
                lhs_val,
                _outputs=[ctx.fresh_name("jnp_less_equal_lhs_cast")],
                to=int(target_ir.value),
            )
            lhs_cmp.type = ir.TensorType(target_ir)
            lhs_cmp.shape = lhs_val.shape
            _ensure_value_metadata(ctx, lhs_cmp)

        rhs_cmp = rhs_val
        if rhs_dtype != target_dtype:
            rhs_cmp = ctx.builder.Cast(
                rhs_val,
                _outputs=[ctx.fresh_name("jnp_less_equal_rhs_cast")],
                to=int(target_ir.value),
            )
            rhs_cmp.type = ir.TensorType(target_ir)
            rhs_cmp.shape = rhs_val.shape
            _ensure_value_metadata(ctx, rhs_cmp)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_less_equal_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_less_equal_out")

        result = ctx.builder.LessOrEqual(lhs_cmp, rhs_cmp, _outputs=[desired_name])
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


@JnpLessEqualPlugin._PRIM.def_impl
def _less_equal_impl(x: object, y: object) -> object:
    orig = get_orig_impl(JnpLessEqualPlugin._PRIM, JnpLessEqualPlugin._FUNC_NAME)
    return orig(x, y)


def _less_equal_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpLessEqualPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpLessEqualPlugin._PRIM] = _less_equal_batch_rule
