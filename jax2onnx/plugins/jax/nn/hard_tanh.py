# jax2onnx/plugins/jax/nn/hard_tanh.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_HARD_TANH_PRIM: Final[Primitive] = Primitive("jax.nn.hard_tanh")
_HARD_TANH_PRIM.multiple_results = False
_JAX_HARD_TANH_ORIG: Final = jax.nn.hard_tanh


@register_primitive(
    jaxpr_primitive=_HARD_TANH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.hard_tanh.html",
    onnx=[
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
    ],
    since="0.12.1",
    context="primitives.nn",
    component="hard_tanh",
    testcases=[
        {
            "testcase": "jaxnn_hard_tanh",
            "callable": lambda x: jax.nn.hard_tanh(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Max:2x5 -> Min:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_hard_tanh_dynamic",
            "callable": lambda x: jax.nn.hard_tanh(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": EG(
                ["Max:Bx4 -> Min:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class HardTanhPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.hard_tanh`` via ONNX ``Max`` + ``Min``."""

    _PRIM: ClassVar[Primitive] = _HARD_TANH_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("hard_tanh_in"))
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("hard_tanh_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "hard_tanh_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("hard_tanh_out")

        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        minus_one_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("hard_tanh_min"),
            array=np.asarray(-1, dtype=x_dtype),
        )
        plus_one_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("hard_tanh_max"),
            array=np.asarray(1, dtype=x_dtype),
        )

        out_type = getattr(out_spec, "type", None) or getattr(x_val, "type", None)
        out_shape = getattr(out_spec, "shape", None) or getattr(x_val, "shape", None)

        max_val = ctx.builder.Max(
            x_val,
            minus_one_const,
            _outputs=[ctx.fresh_name("hard_tanh_maxed")],
        )
        if out_type is not None:
            max_val.type = out_type
        if out_shape is not None:
            max_val.shape = out_shape

        result = ctx.builder.Min(
            max_val,
            plus_one_const,
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
                raise RuntimeError("Original jax.nn.hard_tanh not found")

            def _patched(x: ArrayLike) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if jnp.issubdtype(x_arr.dtype, jnp.bool_) or jnp.issubdtype(
                    x_arr.dtype, jnp.complexfloating
                ):
                    return orig(x)
                return cls._PRIM.bind(x_arr)

            return _patched

        return [
            AssignSpec("jax.nn", "hard_tanh_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="hard_tanh",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="hard_tanh",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="hard_tanh",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@HardTanhPlugin._PRIM.def_impl
def _hard_tanh_impl(x: ArrayLike) -> ArrayLike:
    return _JAX_HARD_TANH_ORIG(x)


register_unary_elementwise_batch_rule(HardTanhPlugin._PRIM)
register_jvp_via_jax_jvp(HardTanhPlugin._PRIM, _hard_tanh_impl)
