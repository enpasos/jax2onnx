# jax2onnx/plugins/jax/nn/log1mexp.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LOG1MEXP_PRIM: Final[Primitive] = Primitive("jax.nn.log1mexp")
_LOG1MEXP_PRIM.multiple_results = False
_JAX_LOG1MEXP_ORIG: Final = jax.nn.log1mexp


@register_primitive(
    jaxpr_primitive=_LOG1MEXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.log1mexp.html",
    onnx=[
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="log1mexp",
    testcases=[
        {
            "testcase": "jaxnn_log1mexp_basic",
            "callable": lambda x: jax.nn.log1mexp(x),
            "input_values": [np.array([0.2, 0.8, 2.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Less -> Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_log1mexp_dynamic",
            "callable": lambda x: jax.nn.log1mexp(x),
            "input_shapes": [("B",)],
            "post_check_onnx_graph": EG(
                ["Less -> Where:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class Log1mexpPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.log1mexp`` as a numerically-stable ONNX branch."""

    _PRIM: ClassVar[Primitive] = _LOG1MEXP_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = jax.eval_shape(_JAX_LOG1MEXP_ORIG, spec)
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("log1mexp_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("log1mexp_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("log1mexp_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("log1mexp_out")

        out_type = getattr(out_spec, "type", None) or getattr(x_val, "type", None)
        out_shape = getattr(out_spec, "shape", None) or getattr(x_val, "shape", None)
        x_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))

        one_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("log1mexp_one"),
            array=np.asarray(1.0, dtype=x_dtype),
        )
        ln2_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("log1mexp_ln2"),
            array=np.asarray(np.log(2.0), dtype=x_dtype),
        )

        mask = ctx.builder.Less(
            x_val,
            ln2_const,
            _outputs=[ctx.fresh_name("log1mexp_mask")],
        )
        neg_x = ctx.builder.Neg(
            x_val,
            _outputs=[ctx.fresh_name("log1mexp_neg_x")],
        )
        exp_neg_x = ctx.builder.Exp(
            neg_x,
            _outputs=[ctx.fresh_name("log1mexp_exp_neg_x")],
        )
        expm1_neg_x = ctx.builder.Sub(
            exp_neg_x,
            one_const,
            _outputs=[ctx.fresh_name("log1mexp_expm1_neg_x")],
        )
        neg_expm1 = ctx.builder.Neg(
            expm1_neg_x,
            _outputs=[ctx.fresh_name("log1mexp_neg_expm1")],
        )
        left = ctx.builder.Log(
            neg_expm1,
            _outputs=[ctx.fresh_name("log1mexp_left")],
        )

        neg_exp = ctx.builder.Neg(
            exp_neg_x,
            _outputs=[ctx.fresh_name("log1mexp_neg_exp")],
        )
        one_plus_neg_exp = ctx.builder.Add(
            one_const,
            neg_exp,
            _outputs=[ctx.fresh_name("log1mexp_one_plus_neg_exp")],
        )
        right = ctx.builder.Log(
            one_plus_neg_exp,
            _outputs=[ctx.fresh_name("log1mexp_right")],
        )

        result = ctx.builder.Where(
            mask,
            left,
            right,
            _outputs=[desired_name],
        )
        for val in (
            neg_x,
            exp_neg_x,
            expm1_neg_x,
            neg_expm1,
            left,
            neg_exp,
            one_plus_neg_exp,
            right,
            result,
        ):
            if out_type is not None:
                val.type = out_type
            if out_shape is not None:
                val.shape = out_shape

        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.log1mexp not found")

            def _patched(x: ArrayLike) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x)
                return cls._PRIM.bind(x_arr)

            return _patched

        return [
            AssignSpec("jax.nn", "log1mexp_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="log1mexp",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@Log1mexpPlugin._PRIM.def_impl
def _log1mexp_impl(x: ArrayLike) -> ArrayLike:
    return _JAX_LOG1MEXP_ORIG(x)


register_unary_elementwise_batch_rule(Log1mexpPlugin._PRIM)
register_jvp_via_jax_jvp(Log1mexpPlugin._PRIM, _log1mexp_impl)
