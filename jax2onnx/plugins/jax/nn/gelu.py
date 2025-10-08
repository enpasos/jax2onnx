# jax2onnx/plugins/jax/nn/gelu.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
        },
        {
            "testcase": "jaxnn_gelu_1",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_gelu_approx",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [(3, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_gelu_exact",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(4, 4)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_gelu_tanh",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [("B", 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class GeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.gelu`` to ONNX ``Gelu``."""

    _PRIM: ClassVar[Primitive] = _GELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, approximate: bool = True):
        del approximate
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
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
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "gelu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="gelu",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@GeluPlugin._PRIM.def_impl
def _gelu_impl(*args, **kwargs):
    return jax.nn.gelu(*args, **kwargs)


def _gelu_batch_rule(batched_args, batch_dims, *, approximate=True):
    (x,) = batched_args
    (bd,) = batch_dims
    out = GeluPlugin._PRIM.bind(x, approximate=approximate)
    return out, bd


batching.primitive_batchers[GeluPlugin._PRIM] = _gelu_batch_rule
