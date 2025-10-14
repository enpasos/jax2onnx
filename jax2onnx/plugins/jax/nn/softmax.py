# jax2onnx/plugins/jax/nn/softmax.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_SOFTMAX_PRIM: Final[Primitive] = Primitive("jax.nn.softmax")
_SOFTMAX_PRIM.multiple_results = False
_JAX_SOFTMAX_ORIG = jax.nn.softmax


@register_primitive(
    jaxpr_primitive=_SOFTMAX_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
    onnx=[
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="softmax",
    testcases=[
        {
            "testcase": "softmax",
            "callable": lambda x: jax.nn.softmax(x),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "softmax_2d",
            "callable": lambda x: jax.nn.softmax(x, axis=1),
            "input_shapes": [(4, 5)],
        },
        {
            "testcase": "softmax_3d",
            "callable": lambda x: jax.nn.softmax(x, axis=2),
            "input_shapes": [(2, 3, 4)],
        },
    ],
)
class SoftmaxPlugin(PrimitiveLeafPlugin):
    """IR lowering for ``jax.nn.softmax`` using ONNX ``Softmax``."""

    _PRIM: ClassVar[Primitive] = _SOFTMAX_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, axis: int = -1):
        del axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (x_var,) = eqn.invars

        axis = int(eqn.params.get("axis", -1))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        norm_axis = (axis % rank) if (axis < 0 and rank) else axis

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Softmax",
            input_hint="softmax_in",
            output_hint="softmax_out",
            attrs={"axis": int(norm_axis)},
        )

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "softmax_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="softmax",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@SoftmaxPlugin._PRIM.def_impl
def _softmax_impl(*args, **kwargs):
    return _JAX_SOFTMAX_ORIG(*args, **kwargs)


def _softmax_batch_rule(batched_args, batch_dims, *, axis=-1):
    (x,) = batched_args
    (x_bdim,) = batch_dims

    if x_bdim is None:
        return SoftmaxPlugin._PRIM.bind(x, axis=axis), None

    rank = x.ndim
    canon_axis = axis if axis >= 0 else axis + rank
    if canon_axis < 0 or canon_axis >= rank:
        raise ValueError("Invalid axis for softmax batching rule")

    kwargs = {"axis": axis}
    if x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)

    if canon_axis == x_bdim:
        result = _JAX_SOFTMAX_ORIG(x, axis=0)
        return result, 0

    if canon_axis < x_bdim:
        axis_body = canon_axis
    else:
        axis_body = canon_axis - 1

    result = jax.vmap(lambda t: _JAX_SOFTMAX_ORIG(t, axis=axis_body))(x)
    return result, 0


batching.primitive_batchers[SoftmaxPlugin._PRIM] = _softmax_batch_rule
