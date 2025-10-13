# jax2onnx/plugins/jax/numpy/broadcast_to.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax.broadcast_in_dim import _maybe_inline_constant_broadcast
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_BROADCAST_TO_PRIM: Final = make_jnp_primitive("jax.numpy.broadcast_to")


def _shape_tuple(spec: Sequence[int | object]) -> tuple[int | object, ...]:
    if isinstance(spec, tuple):
        return spec
    if isinstance(spec, list):
        return tuple(spec)
    return (spec,)


@register_primitive(
    jaxpr_primitive=_BROADCAST_TO_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html",
    onnx=[{"component": "Expand", "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html"}],
    since="v0.9.1",
    context="primitives.jnp",
    component="broadcast_to",
    testcases=[
        {
            "testcase": "broadcast_to_leading_axis",
            "callable": lambda a: jnp.broadcast_to(a, (3, 5)),
            "input_shapes": [(5,)],
        },
        {
            "testcase": "broadcast_to_dynamic_batch",
            "callable": lambda a: jnp.broadcast_to(a, ("B", 4)),
            "input_shapes": [(4,)],
        },
    ],
)
class JnpBroadcastToPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _BROADCAST_TO_PRIM
    _FUNC_NAME: ClassVar[str] = "broadcast_to"

    @staticmethod
    def abstract_eval(x, *, shape):
        result_shape = _shape_tuple(shape)
        return jax.core.ShapedArray(result_shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for broadcast_to lowering"
            )

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        shape = _shape_tuple(eqn.params.get("shape", ()))
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("broadcast_to_in"))
        op_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        bdims = tuple(range(len(shape) - len(op_shape), len(shape)))

        if _maybe_inline_constant_broadcast(
            ctx, out_var, x_val, shape, bdims, op_shape
        ):
            return

        origin = getattr(ctx, "get_symbolic_dim_origin", None)
        hints = getattr(ctx, "_scatter_window_hints", None)

        def _peek_scatter_hint(axis: int):
            if not isinstance(hints, dict):
                return None
            values = hints.get(axis)
            if not values:
                return None
            return values[-1]

        dim_pieces: list[ir.Value] = []
        for axis, dim in enumerate(shape):
            if hints and axis not in bdims:
                override = _peek_scatter_hint(axis)
                if override is not None:
                    dim_pieces.append(override)
                    continue
            if isinstance(dim, (int, np.integer)):
                dim_pieces.append(
                    _const_i64(
                        ctx,
                        np.asarray([int(dim)], dtype=np.int64),
                        ctx.fresh_name("broadcast_dim"),
                    )
                )
                continue
            if origin is None:
                raise NotImplementedError(
                    "symbolic dims require ctx.get_symbolic_dim_origin"
                )
            src = origin(dim)
            if src is None:
                raise NotImplementedError(f"no origin recorded for symbolic dim '{dim}'")
            src_val, src_axis = src
            src_rank = len(getattr(getattr(src_val, "shape", None), "dims", ()) or ())
            shp = builder.Shape(
                src_val,
                _outputs=[ctx.fresh_name("broadcast_sym_shape")],
            )
            _stamp_type_and_shape(shp, (src_rank,))
            _ensure_value_metadata(ctx, shp)
            idx = _const_i64(
                ctx,
                np.asarray([int(src_axis)], dtype=np.int64),
                ctx.fresh_name("broadcast_sym_idx"),
            )
            dim_val = builder.Gather(
                shp,
                idx,
                axis=0,
                _outputs=[ctx.fresh_name("broadcast_sym_dim")],
            )
            _stamp_type_and_shape(dim_val, (1,))
            _ensure_value_metadata(ctx, dim_val)
            dim_pieces.append(dim_val)

        target_shape_val = builder.Concat(
            *dim_pieces,
            axis=0,
            _outputs=[ctx.fresh_name("broadcast_target_shape")],
        )
        _stamp_type_and_shape(target_shape_val, (len(shape),))
        _ensure_value_metadata(ctx, target_shape_val)

        need_reshape = len(op_shape) > 0 and len(shape) != len(op_shape)
        if need_reshape:
            reshape_dims = [1] * len(shape)
            for i, r_axis in enumerate(bdims):
                if i < len(op_shape) and isinstance(op_shape[i], (int, np.integer)):
                    reshape_dims[r_axis] = int(op_shape[i])
            reshape_dim_vals: list[ir.Value] = []
            for axis, dim in enumerate(reshape_dims):
                override = _peek_scatter_hint(axis) if hints else None
                if override is not None:
                    reshape_dim_vals.append(override)
                    continue
                reshape_dim_vals.append(
                    _const_i64(
                        ctx,
                        np.asarray([int(dim)], dtype=np.int64),
                        ctx.fresh_name("broadcast_reshape_dim"),
                    )
                )
            reshape_shape_val = builder.Concat(
                *reshape_dim_vals,
                axis=0,
                _outputs=[ctx.fresh_name("broadcast_reshape_shape")],
            )
            _stamp_type_and_shape(reshape_shape_val, (len(shape),))
            _ensure_value_metadata(ctx, reshape_shape_val)

            reshaped = builder.Reshape(
                x_val,
                reshape_shape_val,
                _outputs=[ctx.fresh_name("broadcast_reshape_out")],
            )
            _stamp_type_and_shape(reshaped, tuple(reshape_dims))
            if getattr(x_val, "type", None) is not None:
                reshaped.type = x_val.type
            _ensure_value_metadata(ctx, reshaped)
            expand_input = reshaped
        else:
            expand_input = x_val

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("broadcast_out"))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)

        in_dtype = getattr(getattr(expand_input, "type", None), "dtype", None)
        if in_dtype is not None and out_dtype is not None and in_dtype != out_dtype:
            casted = builder.Cast(
                expand_input,
                _outputs=[ctx.fresh_name("broadcast_cast")],
                to=int(out_dtype.value),
            )
            expand_input = casted

        expanded = builder.Expand(
            expand_input,
            target_shape_val,
            _outputs=[ctx.fresh_name("Expand")],
        )
        final_dtype = out_dtype or getattr(
            getattr(expand_input, "type", None), "dtype", None
        )
        if final_dtype is not None:
            expanded.type = ir.TensorType(final_dtype)
        _stamp_type_and_shape(expanded, out_shape)
        _ensure_value_metadata(ctx, expanded)
        ctx.bind_value_for_var(out_var, expanded)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.broadcast_to not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, shape):
                return cls._PRIM.bind(a, shape=_shape_tuple(shape))

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


@JnpBroadcastToPlugin._PRIM.def_impl
def _broadcast_to_impl(*args, **kwargs):
    orig = get_orig_impl(JnpBroadcastToPlugin._PRIM, JnpBroadcastToPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


JnpBroadcastToPlugin._PRIM.def_abstract_eval(JnpBroadcastToPlugin.abstract_eval)
