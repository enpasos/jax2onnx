# jax2onnx/plugins/jax/nn/hard_sigmoid.py

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


_HARD_SIGMOID_PRIM: Final[Primitive] = Primitive("jax.nn.hard_sigmoid")
_HARD_SIGMOID_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_HARD_SIGMOID_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.hard_sigmoid.html",
    onnx=[
        {
            "component": "HardSigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__HardSigmoid.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="hard_sigmoid",
    testcases=[
        {
            "testcase": "jaxnn_hard_sigmoid",
            "callable": lambda x: jax.nn.hard_sigmoid(x),
            "input_shapes": [(4,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["HardSigmoid:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_hard_sigmoid_dynamic",
            "callable": lambda x: jax.nn.hard_sigmoid(x),
            "input_shapes": [("B", 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["HardSigmoid:Bx5"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class HardSigmoidPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.hard_sigmoid`` to ONNX ``HardSigmoid``."""

    _PRIM: ClassVar[Primitive] = _HARD_SIGMOID_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(
            x_var,
            name_hint=ctx.fresh_name("hard_sigmoid_in"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("hard_sigmoid_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "hard_sigmoid_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("hard_sigmoid_out")

        result = ctx.builder.HardSigmoid(
            x_val,
            _outputs=[desired_name],
            alpha=1.0 / 6.0,
            beta=0.5,
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(x_val, "type", None)
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
                raise RuntimeError("Original jax.nn.hard_sigmoid not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "hard_sigmoid_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="hard_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="hard_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="hard_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@HardSigmoidPlugin._PRIM.def_impl
def _hard_sigmoid_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.hard_sigmoid(x)


register_unary_elementwise_batch_rule(HardSigmoidPlugin._PRIM)
