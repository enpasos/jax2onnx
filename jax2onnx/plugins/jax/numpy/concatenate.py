# jax2onnx/plugins/jax/numpy/concatenate.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _to_tuple(arrays: Iterable[jnp.ndarray]) -> tuple:
    if isinstance(arrays, (list, tuple)):
        return tuple(arrays)
    return tuple(arrays)


def _normalize_axis(axis: int, rank: int) -> int:
    if rank == 0:
        return 0
    ax = int(axis)
    return ax % rank if ax < 0 else ax


def _promote_dtype(dtypes: Sequence[np.dtype]) -> np.dtype:
    result = dtypes[0]
    for dt in dtypes[1:]:
        result = np.promote_types(result, dt)
    return result


def _concat_dynamic_tile(x: jnp.ndarray) -> jnp.ndarray:
    """Mimic the legacy concat-with-token pattern used in Transformer blocks."""

    # x : (B, N, D)
    d_feature = x.shape[2]
    token = jnp.zeros((1, 1, d_feature), dtype=x.dtype)
    tiled = jnp.broadcast_to(token, (x.shape[0], 1, d_feature))
    return jnp.concatenate([tiled, x], axis=1)


_CONCAT_PRIM: Final = make_jnp_primitive("jax.numpy.concatenate")


@register_primitive(
    jaxpr_primitive=_CONCAT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate_basic",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concatenate_mixed_dtypes",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
            "input_dtypes": [np.float32, np.int32],
        },
        {
            "testcase": "concatenate_with_explicit_dtype",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0, dtype=jnp.float64),
            "input_shapes": [(3,), (3,)],
            "input_dtypes": [np.float32, np.int32],
            "enable_double_precision": True,
        },
        {
            "testcase": "concatenate_with_explicit_dtype_casts_inputs",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1, dtype=jnp.float32),
            "input_shapes": [(5, 1), (5, 1)],
            "input_dtypes": [np.int32, np.int32],
            "expected_output_shapes": [(5, 2)],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "concatenate_abstract_middle_dim",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1),
            "input_shapes": [("B", 1, 8), ("B", 10, 8)],
            "expected_output_shapes": [("B", 11, 8)],
        },
        {
            "testcase": "concatenate_tile_and_symbolic",
            "callable": _concat_dynamic_tile,
            "input_shapes": [("B", 49, 256)],
            "expected_output_shapes": [("B", 50, 256)],
        },
    ],
)
class JnpConcatenatePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _CONCAT_PRIM
    _FUNC_NAME: ClassVar[str] = "concatenate"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _canonicalize_call(*args, **kwargs):
        if not args:
            raise TypeError(
                "concatenate() missing required positional argument 'arrays'"
            )

        arrays = args[0]
        if len(args) > 1:
            axis = args[1]
        else:
            axis = kwargs.pop("axis", 0)
        if len(args) > 2:
            dtype = args[2]
        else:
            dtype = kwargs.pop("dtype", None)
        if len(args) > 3 or kwargs:
            raise TypeError("concatenate() received unexpected arguments")

        arrays_tuple = _to_tuple(arrays)
        if not arrays_tuple:
            raise ValueError("need at least one array to concatenate")
        return arrays_tuple, axis, dtype

    @staticmethod
    def abstract_eval(*arrays, axis=0, dtype=None):
        if not arrays:
            raise ValueError("concatenate requires at least one operand")
        rank = len(arrays[0].shape)
        norm_axis = _normalize_axis(axis, rank)

        axis_sizes = []
        other_dims = arrays[0].shape
        for aval in arrays:
            if len(aval.shape) != rank:
                raise ValueError("all arrays must have the same rank")
            for i, (dim_ref, dim_cur) in enumerate(zip(other_dims, aval.shape)):
                if i == norm_axis:
                    continue
                if dim_ref != dim_cur:
                    raise ValueError("all non-concatenated dimensions must match")
            axis_sizes.append(aval.shape[norm_axis])

        out_shape = list(other_dims)
        if all(isinstance(sz, (int, np.integer)) for sz in axis_sizes):
            out_shape[norm_axis] = int(sum(int(sz) for sz in axis_sizes))
        else:
            out_shape[norm_axis] = axis_sizes[0]
        if dtype is not None:
            out_dtype = np.dtype(dtype)
        else:
            out_dtype = _promote_dtype([np.dtype(a.dtype) for a in arrays])
        return jax.core.ShapedArray(tuple(out_shape), out_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        out_var = eqn.outvars[0]
        in_vars = list(eqn.invars)
        params = getattr(eqn, "params", {})

        axis = params.get("axis", 0)
        dtype_param = params.get("dtype", None)

        first_shape = tuple(getattr(in_vars[0].aval, "shape", ()))
        rank = len(first_shape)
        norm_axis = _normalize_axis(axis, rank)

        target_dtype = (
            np.dtype(dtype_param)
            if dtype_param is not None
            else _promote_dtype(
                [np.dtype(getattr(v.aval, "dtype", np.float32)) for v in in_vars]
            )
        )
        target_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)

        inputs: list[ir.Value] = []
        for var in in_vars:
            val = ctx.get_value_for_var(var, name_hint=ctx.fresh_name("jnp_concat_in"))
            var_dtype = np.dtype(getattr(var.aval, "dtype", target_dtype))
            if var_dtype != target_dtype:
                cast_val = ctx.builder.Cast(
                    val,
                    _outputs=[ctx.fresh_name("jnp_concat_cast")],
                    to=int(target_enum.value),
                )
                cast_val.type = ir.TensorType(target_enum)
                _stamp_type_and_shape(cast_val, tuple(getattr(var.aval, "shape", ())))
                _ensure_value_metadata(ctx, cast_val)
                inputs.append(cast_val)
            else:
                inputs.append(val)

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_concat_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Concat")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Concat")

        result = ctx.builder.Concat(
            *inputs,
            axis=int(norm_axis),
            _outputs=[desired_name],
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.concatenate not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(*args, **kwargs):
                arrays, axis, dtype = cls._canonicalize_call(*args, **kwargs)
                axis_int = int(axis)
                return cls._PRIM.bind(*arrays, axis=axis_int, dtype=dtype)

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


@JnpConcatenatePlugin._PRIM.def_impl
def _concatenate_impl(*args, **kwargs):
    orig = get_orig_impl(JnpConcatenatePlugin._PRIM, JnpConcatenatePlugin._FUNC_NAME)
    return orig(*args, **kwargs)
