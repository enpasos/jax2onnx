# jax2onnx/plugins/jax/nn/tanh.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
import jax.numpy as jnp
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_TANH_PRIM: Final[Primitive] = Primitive("jax.nn.tanh")
_TANH_PRIM.multiple_results = False
_JAX_TANH_ORIG: Final = jax.nn.tanh


@register_primitive(
    jaxpr_primitive=_TANH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.tanh.html",
    onnx=[
        {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"}
    ],
    since="0.12.1",
    context="primitives.nn",
    component="tanh",
    testcases=[
        {
            "testcase": "jaxnn_tanh",
            "callable": lambda x: jax.nn.tanh(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Tanh:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_tanh_dynamic",
            "callable": lambda x: jax.nn.tanh(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": EG(
                ["Tanh:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class TanhPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.tanh`` to ONNX ``Tanh``."""

    _PRIM: ClassVar[Primitive] = _TANH_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Tanh",
            input_hint="tanh_in",
            output_hint="tanh_out",
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
                raise RuntimeError("Original jax.nn.tanh not found")

            def _patched(x: ArrayLike) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x)
                return cls._PRIM.bind(x_arr)

            return _patched

        return [
            AssignSpec("jax.nn", "tanh_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="tanh",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="tanh",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="tanh",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@TanhPlugin._PRIM.def_impl
def _tanh_impl(x: ArrayLike) -> ArrayLike:
    return _JAX_TANH_ORIG(x)


register_unary_elementwise_batch_rule(TanhPlugin._PRIM)
register_jvp_via_jax_jvp(TanhPlugin._PRIM, _tanh_impl)
