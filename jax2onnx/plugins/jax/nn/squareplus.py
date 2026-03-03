# jax2onnx/plugins/jax/nn/squareplus.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

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
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SQUAREPLUS_PRIM: Final[Primitive] = Primitive("jax.nn.squareplus")
_SQUAREPLUS_PRIM.multiple_results = False
_JAX_SQUAREPLUS_ORIG: Final = jax.nn.squareplus


@register_primitive(
    jaxpr_primitive=_SQUAREPLUS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.squareplus.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Sqrt",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="squareplus",
    testcases=[
        {
            "testcase": "jaxnn_squareplus_default_b",
            "callable": lambda x: jax.nn.squareplus(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Mul:2x5 -> Add:2x5 -> Sqrt:2x5 -> Add:2x5 -> Mul:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_squareplus_broadcast_b",
            "callable": lambda x, b: jax.nn.squareplus(x, b=b),
            "input_shapes": [(2, 5), (1, 5)],
            "post_check_onnx_graph": EG(
                ["Mul:2x5 -> Add:2x5 -> Sqrt:2x5 -> Add:2x5 -> Mul:2x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SquareplusPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.squareplus`` via ONNX elementwise ops."""

    _PRIM: ClassVar[Primitive] = _SQUAREPLUS_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        b: jax.core.AbstractValue,
    ) -> jax.core.ShapedArray:
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        b_spec = jax.ShapeDtypeStruct(b.shape, b.dtype)
        out = jax.eval_shape(
            lambda xv, bv: _JAX_SQUAREPLUS_ORIG(xv, b=bv), x_spec, b_spec
        )
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var, b_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("squareplus_x"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("squareplus_b"))
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("squareplus_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "squareplus_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("squareplus_out")

        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        half_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("squareplus_half"),
            array=np.asarray(0.5, dtype=x_dtype),
        )

        out_type = getattr(out_spec, "type", None) or getattr(x_val, "type", None)
        out_shape = getattr(out_spec, "shape", None) or getattr(x_val, "shape", None)
        x_shape = getattr(x_val, "shape", None)

        x_sq = ctx.builder.Mul(
            x_val,
            x_val,
            _outputs=[ctx.fresh_name("squareplus_x_sq")],
        )
        if out_type is not None:
            x_sq.type = out_type
        if x_shape is not None:
            x_sq.shape = x_shape

        inside = ctx.builder.Add(
            x_sq,
            b_val,
            _outputs=[ctx.fresh_name("squareplus_inside")],
        )
        if out_type is not None:
            inside.type = out_type
        if out_shape is not None:
            inside.shape = out_shape

        sqrt_inside = ctx.builder.Sqrt(
            inside,
            _outputs=[ctx.fresh_name("squareplus_sqrt")],
        )
        if out_type is not None:
            sqrt_inside.type = out_type
        if out_shape is not None:
            sqrt_inside.shape = out_shape

        summed = ctx.builder.Add(
            x_val,
            sqrt_inside,
            _outputs=[ctx.fresh_name("squareplus_sum")],
        )
        if out_type is not None:
            summed.type = out_type
        if out_shape is not None:
            summed.shape = out_shape

        result = ctx.builder.Mul(
            summed,
            half_const,
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
                raise RuntimeError("Original jax.nn.squareplus not found")

            def _patched(x: ArrayLike, b: ArrayLike = 4) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x, b=b)
                b_arr = jnp.asarray(b, dtype=x_arr.dtype)
                return cls._PRIM.bind(x_arr, b_arr)

            return _patched

        return [
            AssignSpec("jax.nn", "squareplus_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="squareplus",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SquareplusPlugin._PRIM.def_impl
def _squareplus_impl(x: ArrayLike, b: ArrayLike) -> ArrayLike:
    return _JAX_SQUAREPLUS_ORIG(x, b=b)


register_jvp_via_jax_jvp(SquareplusPlugin._PRIM, _squareplus_impl)


def _squareplus_batch_rule(
    batched_args: tuple[Any, ...], batch_dims: tuple[Any, ...], **params: Any
) -> Any:
    return broadcast_batcher_compat(
        SquareplusPlugin._PRIM,
        batched_args,
        batch_dims,
        **params,
    )


batching.primitive_batchers[SquareplusPlugin._PRIM] = _squareplus_batch_rule
