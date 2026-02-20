# jax2onnx/plugins/jax/numpy/compress.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_COMPRESS_PRIM: Final = make_jnp_primitive("jax.numpy.compress")


def _canonical_axis(axis: int, rank: int) -> int:
    axis_norm = axis if axis >= 0 else axis + rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return axis_norm


def _compress_len(cond: tuple[bool, ...], axis_dim: int) -> int:
    return int(sum(1 for v in cond[:axis_dim] if bool(v)))


@register_primitive(
    jaxpr_primitive=_COMPRESS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.compress.html",
    onnx=[
        {
            "component": "Compress",
            "doc": "https://onnx.ai/onnx/operators/onnx__Compress.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="compress",
    testcases=[
        {
            "testcase": "jnp_compress_axis0",
            "callable": lambda x: jnp.compress(
                np.array([True, False, True]), x, axis=0
            ),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
            "expected_output_shapes": [(2, 4)],
            "post_check_onnx_graph": EG(
                ["Compress:2x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_compress_axis1",
            "callable": lambda x: jnp.compress(
                np.array([True, False, False, True]), x, axis=1
            ),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
            "expected_output_shapes": [(3, 2)],
            "post_check_onnx_graph": EG(
                ["Compress:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "compress_jvp_issue_batch_diff_rules",
            "callable": lambda x: jax.jvp(
                lambda y: jnp.compress(np.array([True, False, True]), y, axis=1),
                (x,),
                (jnp.ones_like(x),),
            )[1],
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 2)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpCompressPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _COMPRESS_PRIM
    _FUNC_NAME: ClassVar[str] = "compress"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.ShapedArray,
        *,
        axis: int,
        cond: tuple[bool, ...],
    ) -> core.ShapedArray:
        rank = len(a.shape)
        axis_norm = _canonical_axis(int(axis), rank)
        axis_dim = a.shape[axis_norm]
        if not isinstance(axis_dim, (int, np.integer)):
            raise NotImplementedError(
                "jnp.compress lowering currently requires static axis dimension"
            )
        out_shape = list(a.shape)
        out_shape[axis_norm] = _compress_len(cond, int(axis_dim))
        return core.ShapedArray(tuple(out_shape), a.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (a_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = dict(getattr(eqn, "params", {}) or {})
        axis_param = int(params["axis"])
        cond_param = tuple(bool(v) for v in params["cond"])

        in_shape = tuple(getattr(getattr(a_var, "aval", None), "shape", ()))
        axis_norm = _canonical_axis(axis_param, len(in_shape))

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("compress_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("compress_out")
        )

        cond_const = ctx.bind_const_for_var(
            object(), np.asarray(cond_param, dtype=np.bool_)
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("compress_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("compress_out")

        result = ctx.builder.Compress(
            a_val,
            cond_const,
            axis=axis_norm,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(a_val, "type", None) is not None:
            result.type = a_val.type
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.compress not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                condition: ArrayLike,
                a: ArrayLike,
                axis: int | None = None,
                out: ArrayLike | None = None,
                *,
                size: int | None = None,
                fill_value: ArrayLike = 0,
            ) -> jax.Array:
                # Only intercept the subset that maps directly to ONNX Compress.
                if out is not None or size is not None or axis is None:
                    return cast(Callable[..., jax.Array], orig)(
                        condition,
                        a,
                        axis=axis,
                        out=out,
                        size=size,
                        fill_value=fill_value,
                    )
                if isinstance(fill_value, core.Tracer):
                    return cast(Callable[..., jax.Array], orig)(
                        condition,
                        a,
                        axis=axis,
                        out=out,
                        size=size,
                        fill_value=fill_value,
                    )
                try:
                    fill_arr = np.asarray(fill_value)
                except Exception:
                    return cast(Callable[..., jax.Array], orig)(
                        condition,
                        a,
                        axis=axis,
                        out=out,
                        size=size,
                        fill_value=fill_value,
                    )
                if not (fill_arr.shape == () and fill_arr.item() == 0):
                    return cast(Callable[..., jax.Array], orig)(
                        condition,
                        a,
                        axis=axis,
                        out=out,
                        size=size,
                        fill_value=fill_value,
                    )
                if isinstance(condition, core.Tracer):
                    return cast(Callable[..., jax.Array], orig)(
                        condition,
                        a,
                        axis=axis,
                        out=out,
                        size=size,
                        fill_value=fill_value,
                    )
                try:
                    cond_arr = np.asarray(condition, dtype=np.bool_).reshape(-1)
                    axis_int = int(axis)
                except Exception:
                    return cast(Callable[..., jax.Array], orig)(
                        condition,
                        a,
                        axis=axis,
                        out=out,
                        size=size,
                        fill_value=fill_value,
                    )
                return cls._PRIM.bind(a, axis=axis_int, cond=tuple(cond_arr.tolist()))

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


@JnpCompressPlugin._PRIM.def_impl
def _compress_impl(
    a: ArrayLike,
    *,
    axis: int,
    cond: tuple[bool, ...],
) -> jax.Array:
    orig = get_orig_impl(JnpCompressPlugin._PRIM, JnpCompressPlugin._FUNC_NAME)
    arr = jnp.asarray(a)
    axis_norm = _canonical_axis(int(axis), arr.ndim)
    axis_dim = arr.shape[axis_norm]
    if not isinstance(axis_dim, (int, np.integer)):
        raise NotImplementedError(
            "jnp.compress lowering currently requires static axis dimension"
        )
    size = _compress_len(cond, int(axis_dim))
    return orig(
        np.asarray(cond, dtype=np.bool_),
        arr,
        axis=axis_norm,
        size=size,
        fill_value=0,
    )


register_jvp_via_jax_jvp(JnpCompressPlugin._PRIM, _compress_impl)


JnpCompressPlugin._PRIM.def_abstract_eval(JnpCompressPlugin.abstract_eval)
