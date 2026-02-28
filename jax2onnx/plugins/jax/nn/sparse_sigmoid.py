# jax2onnx/plugins/jax/nn/sparse_sigmoid.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

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


_SPARSE_SIGMOID_PRIM: Final[Primitive] = Primitive("jax.nn.sparse_sigmoid")
_SPARSE_SIGMOID_PRIM.multiple_results = False
_JAX_SPARSE_SIGMOID_ORIG: Final = jax.nn.sparse_sigmoid


@register_primitive(
    jaxpr_primitive=_SPARSE_SIGMOID_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.sparse_sigmoid.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "Less",
            "doc": "https://onnx.ai/onnx/operators/onnx__Less.html",
        },
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="sparse_sigmoid",
    testcases=[
        {
            "testcase": "jaxnn_sparse_sigmoid",
            "callable": lambda x: jax.nn.sparse_sigmoid(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Where:2x5 -> Where:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_sparse_sigmoid_dynamic",
            "callable": lambda x: jax.nn.sparse_sigmoid(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": EG(
                ["Where:Bx4 -> Where:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class SparseSigmoidPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.sparse_sigmoid`` as a piecewise ONNX graph."""

    _PRIM: ClassVar[Primitive] = _SPARSE_SIGMOID_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(
            x_var,
            name_hint=ctx.fresh_name("sparse_sigmoid_in"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("sparse_sigmoid_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "sparse_sigmoid_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("sparse_sigmoid_out")

        x_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))
        one_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("sparse_sigmoid_one"),
            array=np.asarray(1.0, dtype=x_dtype),
        )
        minus_one_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("sparse_sigmoid_minus_one"),
            array=np.asarray(-1.0, dtype=x_dtype),
        )
        zero_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("sparse_sigmoid_zero"),
            array=np.asarray(0.0, dtype=x_dtype),
        )
        half_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("sparse_sigmoid_half"),
            array=np.asarray(0.5, dtype=x_dtype),
        )

        out_type = getattr(out_spec, "type", None) or getattr(x_val, "type", None)
        out_shape = getattr(out_spec, "shape", None) or getattr(x_val, "shape", None)

        x_plus_one = ctx.builder.Add(
            x_val,
            one_const,
            _outputs=[ctx.fresh_name("sparse_sigmoid_x_plus_one")],
        )
        mid_val = ctx.builder.Mul(
            x_plus_one,
            half_const,
            _outputs=[ctx.fresh_name("sparse_sigmoid_mid")],
        )

        low_mask = ctx.builder.Less(
            x_val,
            minus_one_const,
            _outputs=[ctx.fresh_name("sparse_sigmoid_low_mask")],
        )
        low_or_mid = ctx.builder.Where(
            low_mask,
            zero_const,
            mid_val,
            _outputs=[ctx.fresh_name("sparse_sigmoid_low_or_mid")],
        )

        high_mask = ctx.builder.Greater(
            x_val,
            one_const,
            _outputs=[ctx.fresh_name("sparse_sigmoid_high_mask")],
        )
        result = ctx.builder.Where(
            high_mask,
            one_const,
            low_or_mid,
            _outputs=[desired_name],
        )

        for val in (x_plus_one, mid_val, low_or_mid, result):
            if out_type is not None:
                val.type = out_type
            if out_shape is not None:
                val.shape = out_shape
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
                raise RuntimeError("Original jax.nn.sparse_sigmoid not found")

            def _patched(x: ArrayLike) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x)
                return cls._PRIM.bind(x_arr)

            return _patched

        return [
            AssignSpec(
                "jax.nn",
                "sparse_sigmoid_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="sparse_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SparseSigmoidPlugin._PRIM.def_impl
def _sparse_sigmoid_impl(x: ArrayLike) -> ArrayLike:
    return _JAX_SPARSE_SIGMOID_ORIG(x)


register_unary_elementwise_batch_rule(SparseSigmoidPlugin._PRIM)
register_jvp_via_jax_jvp(SparseSigmoidPlugin._PRIM, _sparse_sigmoid_impl)
