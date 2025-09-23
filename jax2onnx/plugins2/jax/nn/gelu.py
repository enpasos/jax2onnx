from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info as _add_value_info
from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_GELU_PRIM = Primitive("jax.nn.gelu")
_GELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_GELU_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html",
    onnx=[
        {"component": "Gelu", "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html"}
    ],
    since="v0.7.1",
    context="primitives2.nn",
    component="gelu",
    testcases=[
        {
            "testcase": "jaxnn_gelu_exact",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(4, 4)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_gelu_tanh",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [("B", 3)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
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
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars
        approximate = bool(eqn.params.get("approximate", True))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("gelu_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("gelu_out"))

        approx_attr = "tanh" if approximate else "none"

        attr = IRAttr("approximate", IRAttrType.STRING, approx_attr)
        ctx.add_node(
            ir.Node(
                op_type="Gelu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Gelu"),
                attributes=[attr],
            )
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(y_val, x_shape)
        _add_value_info(ctx, y_val)

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
