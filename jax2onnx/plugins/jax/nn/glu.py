# jax2onnx/plugins/jax/nn/glu.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, TypeAlias

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_GLU_PRIM: Final[Primitive] = Primitive("jax.nn.glu")
_GLU_PRIM.multiple_results = False
_JAX_GLU_ORIG: Final = jax.nn.glu
BatchDim: TypeAlias = int | None


def _normalize_axis(axis: int, rank: int) -> int:
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += rank
    if axis_i < 0 or axis_i >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return axis_i


@register_primitive(
    jaxpr_primitive=_GLU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.glu.html",
    onnx=[
        {
            "component": "Split",
            "doc": "https://onnx.ai/onnx/operators/onnx__Split.html",
        },
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.12.1",
    context="primitives.nn",
    component="glu",
    testcases=[
        {
            "testcase": "jaxnn_glu_axis_last",
            "callable": lambda x: jax.nn.glu(x, axis=-1),
            "input_shapes": [(2, 4)],
            "post_check_onnx_graph": EG(
                ["Split:2x2 -> Sigmoid:2x2 -> Mul:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_glu_axis1_dynamic",
            "callable": lambda x: jax.nn.glu(x, axis=1),
            "input_shapes": [("B", 6)],
            "expected_output_shapes": [("B", 3)],
            "post_check_onnx_graph": EG(
                ["Split:Bx3 -> Sigmoid:Bx3 -> Mul:Bx3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class GluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.glu`` via ONNX ``Split`` + ``Sigmoid`` + ``Mul``."""

    _PRIM: ClassVar[Primitive] = _GLU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, *, axis: int = -1
    ) -> jax.core.ShapedArray:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = jax.eval_shape(lambda arr: _JAX_GLU_ORIG(arr, axis=axis), spec)
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        axis = int(eqn.params.get("axis", -1))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if not x_shape:
            raise ValueError("jax.nn.glu expects rank >= 1 input")
        axis_norm = _normalize_axis(axis, len(x_shape))

        axis_size = x_shape[axis_norm]
        if not isinstance(axis_size, (int, np.integer)):
            raise NotImplementedError("jax.nn.glu lowering requires static split axis")
        axis_size_i = int(axis_size)
        if axis_size_i % 2 != 0:
            raise ValueError(
                f"jax.nn.glu requires even split axis size, got {axis_size_i}"
            )
        half = axis_size_i // 2

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("glu_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("glu_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("glu_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("glu_out")

        out_type = getattr(out_spec, "type", None) or getattr(x_val, "type", None)
        out_shape = getattr(out_spec, "shape", None)
        if out_shape is None:
            out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))

        split_sizes = _const_i64(
            ctx, np.asarray([half, half], dtype=np.int64), "glu_split_sizes"
        )
        left_name = ctx.fresh_name("glu_left")
        right_name = ctx.fresh_name("glu_right")
        left_val, right_val = ctx.builder.Split(
            x_val,
            split_sizes,
            _outputs=[left_name, right_name],
            axis=axis_norm,
        )
        for split_val in (left_val, right_val):
            if out_type is not None:
                split_val.type = out_type
            if out_shape is not None:
                split_val.shape = out_shape

        sigmoid_val = ctx.builder.Sigmoid(
            right_val,
            _outputs=[ctx.fresh_name("glu_gate")],
        )
        if out_type is not None:
            sigmoid_val.type = out_type
        if out_shape is not None:
            sigmoid_val.shape = out_shape

        result = ctx.builder.Mul(
            left_val,
            sigmoid_val,
            _outputs=[desired_name],
        )
        if out_type is not None:
            result.type = out_type
        if out_shape is not None:
            result.shape = out_shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.glu not found")

            def _patched(x: ArrayLike, axis: int = -1) -> ArrayLike:
                x_arr = jnp.asarray(x)
                is_float_or_complex = jnp.issubdtype(
                    x_arr.dtype, jnp.floating
                ) or jnp.issubdtype(x_arr.dtype, jnp.complexfloating)
                if not is_float_or_complex:
                    return orig(x, axis=axis)

                rank = x_arr.ndim
                if rank == 0:
                    return orig(x, axis=axis)
                try:
                    axis_norm = _normalize_axis(int(axis), rank)
                except ValueError:
                    return orig(x, axis=axis)

                axis_dim = x_arr.shape[axis_norm]
                if (
                    not isinstance(axis_dim, (int, np.integer))
                    or int(axis_dim) % 2 != 0
                ):
                    return orig(x, axis=axis)
                return cls._PRIM.bind(x_arr, axis=axis_norm)

            return _patched

        return [
            AssignSpec("jax.nn", "glu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="glu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="glu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="glu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@GluPlugin._PRIM.def_impl
def _glu_impl(x: ArrayLike, *, axis: int = -1) -> ArrayLike:
    return _JAX_GLU_ORIG(x, axis=axis)


def _glu_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int = -1,
) -> tuple[jax.Array, BatchDim]:
    (x,), (bdim,) = batched_args, batch_dims
    if bdim is None:
        return GluPlugin._PRIM.bind(x, axis=axis), None

    x_front = batching.bdim_at_front(x, bdim, x.shape[bdim])
    slice_rank = x_front.ndim - 1
    axis_norm = _normalize_axis(axis, slice_rank)
    out = GluPlugin._PRIM.bind(x_front, axis=axis_norm + 1)
    return out, 0


batching.primitive_batchers[GluPlugin._PRIM] = _glu_batch_rule
register_jvp_via_jax_jvp(GluPlugin._PRIM, _glu_impl)
