# jax2onnx/plugins/jax/nn/gelu.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise


_GELU_PRIM: Final[Primitive] = Primitive("jax.nn.gelu")
_GELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_GELU_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html",
    onnx=[
        {"component": "Gelu", "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html"}
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="gelu",
    testcases=[
        {
            "testcase": "jaxnn_gelu",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_1",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_approx",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [(3, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:3x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_exact",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_tanh",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [("B", 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:Bx3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class GeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.gelu`` to ONNX ``Gelu``."""

    _PRIM: ClassVar[Primitive] = _GELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, approximate: bool = True
    ) -> jax.core.ShapedArray:
        del approximate
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        approximate = bool(eqn.params.get("approximate", True))

        approx_attr = "tanh" if approximate else "none"

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Gelu",
            input_hint="gelu_in",
            output_hint="gelu_out",
            attrs={"approximate": approx_attr},
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
                raise RuntimeError("Original jax.nn.gelu not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "gelu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@GeluPlugin._PRIM.def_impl
def _gelu_impl(x: ArrayLike, approximate: bool = True) -> ArrayLike:
    return jax.nn.gelu(x, approximate=approximate)


def _gelu_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    approximate: bool = True,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (bd,) = batch_dims
    out = GeluPlugin._PRIM.bind(x, approximate=approximate)
    return out, bd


batching.primitive_batchers[GeluPlugin._PRIM] = _gelu_batch_rule
