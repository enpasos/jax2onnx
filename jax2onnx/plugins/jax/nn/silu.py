# jax2onnx/plugins/jax/nn/silu.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SILU_PRIM: Final[Primitive] = Primitive("jax.nn.silu")
_SILU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SILU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.silu.html",
    onnx=[
        {
            "component": "Swish",
            "doc": "https://onnx.ai/onnx/operators/onnx__Swish.html",
        },
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        },
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="silu",
    testcases=[
        {
            "testcase": "jaxnn_silu",
            "callable": lambda x: jax.nn.silu(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Sigmoid:2x5 -> Mul:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_silu_dynamic",
            "callable": lambda x: jax.nn.silu(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": EG(
                ["Sigmoid:Bx4 -> Mul:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_swish_alias",
            "callable": lambda x: jax.nn.swish(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Sigmoid:3x4 -> Mul:3x4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SiluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.silu``/``jax.nn.swish`` to ONNX ``Swish`` or ``Sigmoid`` + ``Mul``."""

    _PRIM: ClassVar[Primitive] = _SILU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("silu_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("silu_out"))

        x_type = getattr(x_val, "type", None)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("silu_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("silu_out")

        opset = int(getattr(ctx.builder, "opset", 0) or 0)
        if opset >= 24:
            result = ctx.builder.Swish(
                x_val,
                _outputs=[desired_name],
            )
        else:
            sigmoid_val = ctx.builder.Sigmoid(
                x_val,
                _outputs=[ctx.fresh_name("silu_sigmoid")],
            )
            if x_type is not None:
                sigmoid_val.type = x_type
            sigmoid_val.shape = getattr(x_val, "shape", None)
            result = ctx.builder.Mul(
                x_val,
                sigmoid_val,
                _outputs=[desired_name],
            )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = x_type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            result.shape = getattr(x_val, "shape", None)
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
                raise RuntimeError("Original jax.nn.silu/swish not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "silu_p", cls._PRIM, delete_if_missing=True),
            AssignSpec("jax.nn", "swish_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="silu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="swish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="silu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="swish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="silu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="swish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SiluPlugin._PRIM.def_impl
def _silu_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.silu(x)


register_unary_elementwise_batch_rule(SiluPlugin._PRIM)
