# jax2onnx/plugins/jax/nn/softmax.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise


_SOFTMAX_PRIM: Final[Primitive] = Primitive("jax.nn.softmax")
_SOFTMAX_PRIM.multiple_results = False
_JAX_SOFTMAX_ORIG: Final = jax.nn.softmax


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
            "post_check_onnx_graph": EG(
                ["Softmax:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "softmax_2d",
            "callable": lambda x: jax.nn.softmax(x, axis=1),
            "input_shapes": [(4, 5)],
            "post_check_onnx_graph": EG(
                ["Softmax:4x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "softmax_3d",
            "callable": lambda x: jax.nn.softmax(x, axis=2),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Softmax:2x3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "softmax_mask_where",
            "callable": lambda x, m: jax.nn.softmax(x, where=m),
            "input_shapes": [(2, 4), (2, 4)],
            "input_dtypes": [jnp.float32, np.bool_],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Softmax:2x4",
                        "counts": {"Softmax": 1, "Where": 2},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SoftmaxPlugin(PrimitiveLeafPlugin):
    """IR lowering for ``jax.nn.softmax`` using ONNX ``Softmax``."""

    _PRIM: ClassVar[Primitive] = _SOFTMAX_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        where: jax.core.AbstractValue | None = None,
        *,
        axis: int = -1,
        has_where: bool = False,
    ) -> jax.core.ShapedArray:
        del axis
        if has_where and where is None:
            raise ValueError("softmax with has_where=True expects a mask operand")
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        if not eqn.invars:
            raise ValueError("softmax lowering expects at least the input tensor")
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        axis_param = eqn.params.get("axis", -1)
        if isinstance(axis_param, (tuple, list)):
            if len(axis_param) != 1:
                raise NotImplementedError(
                    "jax.nn.softmax lowering only supports a single axis"
                )
            axis = int(axis_param[0])
        else:
            axis = int(axis_param)
        has_where = bool(eqn.params.get("has_where", False))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        norm_axis = (axis % rank) if (axis < 0 and rank) else axis

        if not has_where:
            lower_unary_elementwise(
                ctx,
                eqn,
                op_name="Softmax",
                input_hint="softmax_in",
                output_hint="softmax_out",
                attrs={"axis": int(norm_axis)},
            )
            return

        if len(eqn.invars) < 2:
            raise ValueError(
                "jax.nn.softmax with a where mask must receive the mask operand"
            )
        where_var = eqn.invars[1]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("softmax_in"))
        mask_val = ctx.get_value_for_var(
            where_var, name_hint=ctx.fresh_name("softmax_where")
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("softmax"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("softmax")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("softmax")

        x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        np_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))

        mask_dtype = getattr(getattr(mask_val, "type", None), "dtype", None)
        mask_shape = tuple(getattr(getattr(where_var, "aval", None), "shape", ()))
        if mask_dtype != ir.DataType.BOOL:
            mask_bool = ctx.builder.Cast(
                mask_val,
                _outputs=[ctx.fresh_name("softmax_where_bool")],
                to=int(ir.DataType.BOOL.value),
            )
            mask_bool.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(mask_bool, mask_shape)
            _ensure_value_metadata(ctx, mask_bool)
        else:
            mask_bool = mask_val
            if mask_shape:
                _stamp_type_and_shape(mask_bool, mask_shape)
            _ensure_value_metadata(ctx, mask_bool)

        neg_inf = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("softmax_neg_inf"),
            value=np.asarray(-np.inf, dtype=np_dtype),
        )
        _stamp_type_and_shape(neg_inf, ())
        _ensure_value_metadata(ctx, neg_inf)

        masked = ctx.builder.Where(
            mask_bool,
            x_val,
            neg_inf,
            _outputs=[ctx.fresh_name("softmax_masked")],
        )
        if x_dtype is not None:
            masked.type = ir.TensorType(x_dtype)
        _stamp_type_and_shape(masked, x_shape)
        _ensure_value_metadata(ctx, masked)

        softmax_core = ctx.builder.Softmax(
            masked,
            _outputs=[ctx.fresh_name("softmax_core")],
            axis=int(norm_axis),
        )
        if x_dtype is not None:
            softmax_core.type = ir.TensorType(x_dtype)
        _stamp_type_and_shape(softmax_core, x_shape)
        _ensure_value_metadata(ctx, softmax_core)

        zero_init = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("softmax_zero"),
            value=np.asarray(0.0, dtype=np_dtype),
        )
        _stamp_type_and_shape(zero_init, ())
        _ensure_value_metadata(ctx, zero_init)

        result = ctx.builder.Where(
            mask_bool,
            softmax_core,
            zero_init,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif x_dtype is not None:
            result.type = ir.TensorType(x_dtype)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, x_shape)
        _ensure_value_metadata(ctx, result)

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
                raise RuntimeError("Original jax.nn.softmax not found")

            def _patched(x, axis: int = -1, where=None):
                if where is None:
                    return cls._PRIM.bind(x, axis=axis, has_where=False)
                where_arr = jnp.asarray(where)
                return cls._PRIM.bind(x, where_arr, axis=axis, has_where=True)

            return _patched

        return [
            AssignSpec("jax.nn", "softmax_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="softmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="softmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="softmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SoftmaxPlugin._PRIM.def_impl
def _softmax_impl(
    x: ArrayLike,
    where: ArrayLike | None = None,
    *,
    axis: int = -1,
    has_where: bool = False,
) -> ArrayLike:
    mask = where if has_where else None
    return _JAX_SOFTMAX_ORIG(x, axis=axis, where=mask)


def _softmax_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    axis: int = -1,
    has_where: bool = False,
) -> tuple[jax.Array, int | None]:
    x = batched_args[0]
    x_bdim = batch_dims[0]
    where = batched_args[1] if has_where and len(batched_args) > 1 else None
    where_bdim = batch_dims[1] if has_where and len(batch_dims) > 1 else None

    if x_bdim is None and (not has_where or where_bdim is None):
        bind_args = (x,) if not has_where else (x, where)
        return (
            SoftmaxPlugin._PRIM.bind(*bind_args, axis=axis, has_where=has_where),
            None,
        )

    rank = x.ndim
    canon_axis = axis if axis >= 0 else axis + rank
    if canon_axis < 0 or canon_axis >= rank:
        raise ValueError("Invalid axis for softmax batching rule")

    if x_bdim is not None and x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)
    if where is not None and where_bdim is not None and where_bdim != 0:
        where = jnp.moveaxis(where, where_bdim, 0)

    if x_bdim is None:
        axis_body = canon_axis
    elif canon_axis == x_bdim:
        axis_body = 0
    elif canon_axis < x_bdim:
        axis_body = canon_axis
    else:
        axis_body = canon_axis - 1

    in_axes = (0 if x_bdim is not None else None,)
    if has_where:
        in_axes = in_axes + (0 if where_bdim is not None else None,)

    vmapped = jax.vmap(
        lambda *vals: _JAX_SOFTMAX_ORIG(
            vals[0],
            axis=axis_body,
            where=(vals[1] if has_where else None),
        ),
        in_axes=in_axes,
    )
    result = vmapped(x) if not has_where else vmapped(x, where)
    return result, 0


batching.primitive_batchers[SoftmaxPlugin._PRIM] = _softmax_batch_rule
