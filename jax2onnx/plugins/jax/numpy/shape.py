# jax2onnx/plugins/jax/numpy/shape.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core, lax
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape, _ensure_value_metadata
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SHAPE_PRIM: Final = make_jnp_primitive("jax.numpy.shape")


def _shape_eval(x):
    orig = getattr(_SHAPE_PRIM, "__orig_impl__shape", jnp.shape)
    result = jax.eval_shape(
        lambda arr: orig(arr), jax.ShapeDtypeStruct(x.shape, x.dtype)
    )
    return result


@register_primitive(
    jaxpr_primitive=_SHAPE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.shape.html",
    onnx=[
        {"component": "Shape", "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html"}
    ],
    since="v0.4.0",
    context="primitives.jnp",
    component="shape",
    testcases=[
        {
            "testcase": "shape_basic",
            "callable": lambda x: jnp.asarray(jnp.shape(x), dtype=jnp.int32),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Shape -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "shape_dynamic",
            "callable": lambda x: jnp.asarray(jnp.shape(x), dtype=jnp.int32),
            "input_shapes": [("B", 12, "T", "T")],
            "post_check_onnx_graph": EG(
                ["Shape -> Cast:4"],
                symbols={"B": None, "T": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "shape_vmap_batching",
            "callable": lambda x: jax.vmap(
                lambda y: jnp.asarray(jnp.shape(y), dtype=jnp.int32)
            )(x),
            "input_shapes": [(3, 2, 4)],
        },
    ],
)
class JnpShapePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SHAPE_PRIM
    _FUNC_NAME: ClassVar[str] = "shape"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        result = _shape_eval(x)
        if isinstance(result, tuple):
            rank = len(result)
            if rank == 0:
                return core.ShapedArray((0,), jnp.int32)
            dtype = getattr(result[0], "dtype", jnp.int32)
            return core.ShapedArray((rank,), dtype)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("shape_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("shape_out"))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for shape lowering")

        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        target_dtype = getattr(getattr(out_var, "aval", None), "dtype", jnp.int32)
        try:
            np_dtype = np.dtype(target_dtype)
        except TypeError:
            np_dtype = np.dtype("int32")

        dtype_map = {
            np.dtype("int64"): ir.DataType.INT64,
            np.dtype("int32"): ir.DataType.INT32,
            np.dtype("int16"): ir.DataType.INT16,
            np.dtype("int8"): ir.DataType.INT8,
            np.dtype("uint8"): ir.DataType.UINT8,
        }
        desired = dtype_map.get(np_dtype, ir.DataType.INT32)

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("Shape")
        if desired == ir.DataType.INT64:
            result = builder.Shape(
                arr_val,
                _outputs=[out_name],
            )
            result.type = ir.TensorType(ir.DataType.INT64)
        else:
            raw_out = builder.Shape(
                arr_val,
                _outputs=[ctx.fresh_name("shape_raw")],
            )
            raw_out.type = ir.TensorType(ir.DataType.INT64)
            result = builder.Cast(
                raw_out,
                _outputs=[out_name],
                to=int(desired.value),
            )
            result.type = ir.TensorType(desired)
        _stamp_type_and_shape(result, target_shape)
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jnp.shape not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike) -> ArrayLike:
                arr = jnp.asarray(a)
                return cls._PRIM.bind(arr)

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


@JnpShapePlugin._PRIM.def_impl
def _shape_impl(a: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpShapePlugin._PRIM, JnpShapePlugin._FUNC_NAME)
    return orig(a)


JnpShapePlugin._PRIM.def_abstract_eval(JnpShapePlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _shape_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpShapePlugin._PRIM.bind(operand)
        return out, batching.not_mapped

    batch_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, batch_size)

    full_shape = JnpShapePlugin._PRIM.bind(operand)
    slice_shape = lax.slice_in_dim(full_shape, 1, operand.ndim, axis=0)
    broadcast = jnp.broadcast_to(slice_shape, (batch_size,) + slice_shape.shape)
    return broadcast, 0


batching.primitive_batchers[JnpShapePlugin._PRIM] = _shape_batch_rule
