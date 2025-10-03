from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType
from flax import nnx
from jax import core

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_TAKE_PRIM = make_jnp_primitive("jax.numpy.take")


class _ArangeTakeModule(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: jnp.dtype | type = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_embeddings, features), dtype=dtype)
        )

    def __call__(self, x: jax.Array):
        seq_len = x.shape[1]
        indices = jnp.arange(seq_len)
        return jnp.take(self.embedding.value, indices, axis=0)


def _canonical_axis(axis: int, rank: int) -> int:
    axis_norm = axis if axis >= 0 else axis + rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"jnp.take axis {axis} out of bounds for rank {rank}")
    return axis_norm


def _as_int64(
    ctx: "IRContext", value: ir.Value, shape: tuple[int | object, ...], name_hint: str
) -> ir.Value:
    current_type = getattr(value, "type", None)
    current_dtype = getattr(current_type, "dtype", None)
    if current_dtype == ir.DataType.INT64:
        _stamp_type_and_shape(value, shape)
        _ensure_value_info(ctx, value)
        return value

    cast_val = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(ir.DataType.INT64),
        shape=value.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[cast_val],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(ir.DataType.INT64.value))],
        )
    )
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_info(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=_TAKE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="v0.9.0",
    context="primitives.jnp",
    component="take",
    testcases=[
        {
            "testcase": "take_data_dependent_indices",
            "callable": construct_and_call(
                _ArangeTakeModule,
                num_embeddings=10,
                features=16,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 10)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "take_basic_axis1",
            "callable": lambda x, idx: jnp.take(x, idx, axis=1),
            "input_values": [
                np.arange(12, dtype=np.float32).reshape(3, 4),
                np.array([0, 2], dtype=np.int32),
            ],
            "expected_output_shapes": [(3, 2)],
            "expected_output_dtypes": [np.float32],
        },
    ],
)
class JnpTakePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TAKE_PRIM
    _FUNC_NAME: ClassVar[str] = "take"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(arr, indices, *, axis=None, mode=None):
        if mode is not None:
            raise NotImplementedError("jnp.take mode parameter is not supported")
        if axis is None:
            raise NotImplementedError("jnp.take with axis=None is not supported")
        rank = len(arr.shape)
        axis_norm = _canonical_axis(int(axis), rank)
        out_shape = arr.shape[:axis_norm] + indices.shape + arr.shape[axis_norm + 1 :]
        return core.ShapedArray(out_shape, arr.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        axis_param = params.get("axis")
        mode = params.get("mode")
        if mode is not None:
            raise NotImplementedError("jnp.take mode parameter is not supported")
        if axis_param is None:
            raise NotImplementedError("jnp.take with axis=None is not supported")

        arr_var, indices_var = eqn.invars
        out_var = eqn.outvars[0]

        arr_val = ctx.get_value_for_var(
            arr_var, name_hint=ctx.fresh_name("take_data"), prefer_np_dtype=None
        )
        indices_val = ctx.get_value_for_var(
            indices_var,
            name_hint=ctx.fresh_name("take_indices"),
            prefer_np_dtype=np.int64,
        )
        indices_dtype = np.dtype(getattr(indices_var.aval, "dtype", np.int64))
        if not np.issubdtype(indices_dtype, np.integer):
            raise TypeError("jnp.take indices must be integer typed")

        indices_shape = tuple(getattr(indices_var.aval, "shape", ()))
        indices_val = _as_int64(ctx, indices_val, indices_shape, "take_indices_int64")

        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("take_out"))

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        axis = _canonical_axis(int(axis_param), len(arr_shape))

        ctx.add_node(
            ir.Node(
                op_type="Gather",
                domain="",
                inputs=[arr_val, indices_val],
                outputs=[out_val],
                name=ctx.fresh_name("Gather"),
                attributes=[IRAttr("axis", IRAttrType.INT, int(axis))],
            )
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.take not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(arr, indices, *, axis=None, mode=None):
                if axis is None or mode is not None:
                    return orig(arr, indices, axis=axis, mode=mode)
                indices_arr = jnp.asarray(indices)
                return cls._PRIM.bind(
                    jnp.asarray(arr),
                    indices_arr,
                    axis=int(axis),
                    mode=None,
                )

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpTakePlugin._PRIM.def_impl
def _take_impl(arr, indices, *, axis=None, mode=None):
    orig = get_orig_impl(JnpTakePlugin._PRIM, JnpTakePlugin._FUNC_NAME)
    return orig(arr, indices, axis=axis, mode=mode)


JnpTakePlugin._PRIM.def_abstract_eval(JnpTakePlugin.abstract_eval)
