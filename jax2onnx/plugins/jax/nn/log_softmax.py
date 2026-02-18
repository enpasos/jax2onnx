# jax2onnx/plugins/jax/nn/log_softmax.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise


_LOG_SOFTMAX_PRIM: Final[Primitive] = Primitive("jax.nn.log_softmax")
_LOG_SOFTMAX_PRIM.multiple_results = False
_JAX_LOG_SOFTMAX_ORIG: Final = jax.nn.log_softmax


def _axis_attr_equals(model, expected: int) -> bool:
    node = next(
        (n for n in getattr(model.graph, "node", []) if n.op_type == "LogSoftmax"),
        None,
    )
    if node is None:
        return False
    for attr in getattr(node, "attribute", []):
        if getattr(attr, "name", "") != "axis":
            continue
        if hasattr(attr, "i"):
            return int(attr.i) == int(expected)
        if hasattr(attr, "INT"):
            return int(attr.INT) == int(expected)
        ints = getattr(attr, "ints", None)
        if ints and len(ints):
            return int(ints[0]) == int(expected)
    # Attribute missing implies ONNX default axis=-1.
    return int(expected) == -1


@register_primitive(
    jaxpr_primitive=_LOG_SOFTMAX_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_softmax.html",
    onnx=[
        {
            "component": "LogSoftmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__LogSoftmax.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="log_softmax",
    testcases=[
        {
            "testcase": "jaxnn_log_softmax_default_axis",
            "callable": lambda x: jax.nn.log_softmax(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 1),
        },
        {
            "testcase": "jaxnn_log_softmax_axis0",
            "callable": lambda x: jax.nn.log_softmax(x, axis=0),
            "input_shapes": [(3, 2)],
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 0),
        },
        {
            "testcase": "jaxnn_log_softmax_axis_last_3d",
            "callable": lambda x: jax.nn.log_softmax(x, axis=-1),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 2),
        },
    ],
)
class LogSoftmaxPlugin(PrimitiveLeafPlugin):
    """IR lowering for ``jax.nn.log_softmax`` via ONNX ``LogSoftmax``."""

    _PRIM: ClassVar[Primitive] = _LOG_SOFTMAX_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        *,
        axis: int = -1,
    ) -> jax.core.ShapedArray:
        del axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        axis_param = eqn.params.get("axis", -1)
        if isinstance(axis_param, (tuple, list)):
            if len(axis_param) != 1:
                raise NotImplementedError(
                    "jax.nn.log_softmax lowering only supports a single axis"
                )
            axis = int(axis_param[0])
        else:
            axis = int(axis_param)

        x_var = eqn.invars[0]
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        axis_attr = axis
        if rank:
            axis_attr = axis % rank if axis < 0 else axis
            if axis_attr < 0 or axis_attr >= rank:
                raise ValueError("Invalid axis for log_softmax lowering")

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="LogSoftmax",
            input_hint="log_softmax_in",
            output_hint="log_softmax_out",
            attrs={"axis": int(axis_attr)},
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
                raise RuntimeError("Original jax.nn.log_softmax not found")

            def _patched(x, axis=-1, where=None):
                if where is not None:
                    return orig(x, axis=axis, where=where)
                if isinstance(axis, (tuple, list)):
                    if len(axis) != 1:
                        return orig(x, axis=axis, where=None)
                    axis = int(axis[0])
                else:
                    axis = int(axis)
                return cls._PRIM.bind(x, axis=axis)

            return _patched

        return [
            AssignSpec("jax.nn", "log_softmax_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="log_softmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="log_softmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="log_softmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@LogSoftmaxPlugin._PRIM.def_impl
def _log_softmax_impl(
    x: ArrayLike,
    *,
    axis: int = -1,
) -> ArrayLike:
    return _JAX_LOG_SOFTMAX_ORIG(x, axis=axis, where=None)


def _log_softmax_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    axis: int = -1,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (x_bdim,) = batch_dims
    if x_bdim is None:
        return LogSoftmaxPlugin._PRIM.bind(x, axis=axis), None

    rank = x.ndim
    canon_axis = axis if axis >= 0 else axis + rank
    if canon_axis < 0 or canon_axis >= rank:
        raise ValueError("Invalid axis for log_softmax batching rule")

    if x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)

    if canon_axis == x_bdim:
        axis_body = 0
    elif canon_axis < x_bdim:
        axis_body = canon_axis
    else:
        axis_body = canon_axis - 1

    out = jax.vmap(
        lambda t: _JAX_LOG_SOFTMAX_ORIG(t, axis=axis_body, where=None),
        in_axes=0,
    )(x)
    return out, 0


batching.primitive_batchers[LogSoftmaxPlugin._PRIM] = _log_softmax_batch_rule
