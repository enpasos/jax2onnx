# jax2onnx/plugins/jax/nn/softplus.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
    register_unary_elementwise_batch_rule,
)


_SOFTPLUS_PRIM: Final[Primitive] = Primitive("jax.nn.softplus")
_SOFTPLUS_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SOFTPLUS_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="softplus",
    testcases=[
        {
            "testcase": "jaxnn_softplus",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Softplus:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_softplus_1",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Softplus:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_softplus_basic",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(4,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Softplus:4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SoftplusPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.nn.softplus``."""

    _PRIM: ClassVar[Primitive] = _SOFTPLUS_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Softplus",
            input_hint="softplus_in",
            output_hint="softplus_out",
        )

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
                raise RuntimeError("Original jax.nn.softplus not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "softplus_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="softplus",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="softplus",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="softplus",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SoftplusPlugin._PRIM.def_impl
def _softplus_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.softplus(x)


register_unary_elementwise_batch_rule(SoftplusPlugin._PRIM)
