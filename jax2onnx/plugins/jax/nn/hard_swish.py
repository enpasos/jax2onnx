# jax2onnx/plugins/jax/nn/hard_swish.py

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


_HARD_SWISH_PRIM: Final[Primitive] = Primitive("jax.nn.hard_swish")
_HARD_SWISH_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_HARD_SWISH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.hard_swish.html",
    onnx=[
        {
            "component": "HardSwish",
            "doc": "https://onnx.ai/onnx/operators/onnx__HardSwish.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="hard_swish",
    testcases=[
        {
            "testcase": "jaxnn_hard_swish",
            "callable": lambda x: jax.nn.hard_swish(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["HardSwish:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_hard_swish_dynamic",
            "callable": lambda x: jax.nn.hard_swish(x),
            "input_shapes": [("B", 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["HardSwish:Bx3x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class HardSwishPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.hard_swish`` to ONNX ``HardSwish``."""

    _PRIM: ClassVar[Primitive] = _HARD_SWISH_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(
            x_var,
            name_hint=ctx.fresh_name("hard_swish_in"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("hard_swish_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "hard_swish_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("hard_swish_out")

        result = ctx.builder.HardSwish(
            x_val,
            _outputs=[desired_name],
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
                raise RuntimeError("Original jax.nn.hard_swish not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "hard_swish_p", cls._PRIM, delete_if_missing=True),
            AssignSpec("jax.nn", "hard_silu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="hard_swish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="hard_silu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="hard_swish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="hard_swish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@HardSwishPlugin._PRIM.def_impl
def _hard_swish_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.hard_swish(x)


register_unary_elementwise_batch_rule(HardSwishPlugin._PRIM)
