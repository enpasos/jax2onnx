# jax2onnx/plugins/jax/nn/identity.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
    register_unary_elementwise_batch_rule,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_IDENTITY_PRIM: Final[Primitive] = Primitive("jax.nn.identity")
_IDENTITY_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_IDENTITY_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.identity.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="identity",
    testcases=[
        {
            "testcase": "jaxnn_identity",
            "callable": lambda x: jax.nn.identity(x),
            "input_shapes": [(1,)],
            "post_check_onnx_graph": EG(
                ["Identity:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_identity_1",
            "callable": lambda x: jax.nn.identity(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Identity:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_identity_basic",
            "callable": lambda x: jax.nn.identity(x),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                ["Identity:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_identity_dynamic",
            "callable": lambda x: jax.nn.identity(x),
            "input_shapes": [("B", 7)],
            "post_check_onnx_graph": EG(
                ["Identity:Bx7"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class IdentityPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.identity`` to ONNX ``Identity``."""

    _PRIM: ClassVar[Primitive] = _IDENTITY_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Identity",
            input_hint="identity_in",
            output_hint="identity_out",
        )

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "identity_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="identity",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@IdentityPlugin._PRIM.def_impl
def _identity_impl(*args, **kwargs):
    return jax.nn.identity(*args, **kwargs)


register_unary_elementwise_batch_rule(IdentityPlugin._PRIM)
