# jax2onnx/plugins/jax/nn/one_hot.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._axis0_utils import _np_dtype_for_enum
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ONE_HOT_PRIM: Final[Primitive] = Primitive("jax.nn.one_hot")
_ONE_HOT_PRIM.multiple_results = False
_JAX_ONE_HOT_ORIG: Final = jax.nn.one_hot


@register_primitive(
    jaxpr_primitive=_ONE_HOT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.one_hot.html",
    onnx=[
        {
            "component": "OneHot",
            "doc": "https://onnx.ai/onnx/operators/onnx__OneHot.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="one_hot",
    testcases=[
        {
            "testcase": "jaxnn_one_hot_default",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=6),
            "input_values": [np.array([0, 2, 4, 5], dtype=np.int32)],
            "expected_output_shapes": [(4, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["OneHot:4x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_one_hot_axis0",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=5, axis=0),
            "input_values": [np.array([1, 3], dtype=np.int32)],
            "expected_output_shapes": [(5, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["OneHot:5x2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class OneHotPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.one_hot`` to ONNX ``OneHot``."""

    _PRIM: ClassVar[Primitive] = _ONE_HOT_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        *,
        num_classes: int,
        dtype: Any = jnp.float32,
        axis: int = -1,
    ) -> jax.core.ShapedArray:
        x_shape = tuple(getattr(x, "shape", ()))
        out_rank = len(x_shape) + 1

        axis_int = int(axis)
        if axis_int < 0:
            axis_int += out_rank
        if axis_int < 0 or axis_int > len(x_shape):
            raise ValueError(
                f"one_hot axis={axis} is out of bounds for input rank {len(x_shape)}"
            )

        out_shape = list(x_shape)
        out_shape.insert(axis_int, int(num_classes))
        return jax.core.ShapedArray(tuple(out_shape), np.dtype(dtype))

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        num_classes = int(eqn.params.get("num_classes"))
        axis = int(eqn.params.get("axis", -1))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("one_hot_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("one_hot_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("one_hot_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("one_hot_out")

        out_dtype_enum = getattr(getattr(out_spec, "type", None), "dtype", None)
        out_dtype = _np_dtype_for_enum(out_dtype_enum)
        if out_dtype is None:
            out_dtype = np.dtype(
                getattr(getattr(out_var, "aval", None), "dtype", np.float32)
            )

        indices_input = x_val
        x_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.int64))
        if x_dtype != np.dtype(np.int64):
            indices_input = ctx.builder.Cast(
                x_val,
                _outputs=[ctx.fresh_name("one_hot_indices_i64")],
                to=int(
                    _dtype_to_ir(
                        np.dtype(np.int64), ctx.builder.enable_double_precision
                    ).value
                ),
            )
            if getattr(x_val, "shape", None) is not None:
                indices_input.shape = x_val.shape

        depth_const = ctx.bind_const_for_var(
            object(),
            np.asarray(num_classes, dtype=np.int64),
        )
        values_const = ctx.bind_const_for_var(
            object(),
            np.asarray([0, 1], dtype=out_dtype),
        )

        result = ctx.builder.OneHot(
            indices_input,
            depth_const,
            values_const,
            _outputs=[desired_name],
            axis=axis,
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
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
                raise RuntimeError("Original jax.nn.one_hot not found")

            def _patched(
                x: ArrayLike,
                num_classes: int,
                *,
                dtype: Any = jnp.float32,
                axis: int = -1,
            ) -> ArrayLike:
                return cls._PRIM.bind(
                    x,
                    num_classes=int(num_classes),
                    dtype=np.dtype(dtype),
                    axis=int(axis),
                )

            return _patched

        return [
            AssignSpec("jax.nn", "one_hot_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="one_hot",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="one_hot",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="one_hot",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@OneHotPlugin._PRIM.def_impl
def _one_hot_impl(
    x: ArrayLike,
    *,
    num_classes: int,
    dtype: Any = jnp.float32,
    axis: int = -1,
) -> ArrayLike:
    return _JAX_ONE_HOT_ORIG(x, num_classes=num_classes, dtype=dtype, axis=axis)


def _one_hot_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    num_classes: int,
    dtype: Any = jnp.float32,
    axis: int = -1,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (bd,) = batch_dims

    out = OneHotPlugin._PRIM.bind(
        x,
        num_classes=num_classes,
        dtype=dtype,
        axis=axis,
    )
    if bd is None:
        return out, None

    out_rank = x.ndim + 1
    axis_int = int(axis)
    if axis_int < 0:
        axis_int += out_rank
    out_bd = bd + 1 if axis_int <= bd else bd
    return out, out_bd


batching.primitive_batchers[OneHotPlugin._PRIM] = _one_hot_batch_rule
