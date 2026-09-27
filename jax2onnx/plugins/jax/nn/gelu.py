# jax2onnx/plugins/jax/nn/gelu.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, cast

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    Primitive,
    ShapedArray,
    ad,
    batching,
)
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_rule
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise


_GELU_PRIM: Final[Primitive] = Primitive("jax.nn.gelu")
_GELU_PRIM.multiple_results = False
_JAX_GELU_ORIG: Final = jax.nn.gelu

# ONNX ``Gelu`` was introduced in opset 20.
_GELU_MIN_OPSET: Final[int] = 20
_SQRT_HALF: Final[float] = 0.7071067811865476
_SQRT_2_OVER_PI: Final[float] = 0.7978845608028654
_GELU_TANH_COEFF: Final[float] = 0.044715


def lower_gelu(
    ctx: LoweringContextProtocol, eqn: JaxprEqn, *, approximate: bool
) -> None:
    """Lower GELU to ONNX ``Gelu``, or to its formula below opset 20.

    The decomposition follows ``jax.nn.gelu``: ``0.5*x*(1 + erf(x/sqrt(2)))``
    when exact, ``x*0.5*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x**3)))`` when
    approximate.
    """
    if int(ctx.builder.opset) >= _GELU_MIN_OPSET:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Gelu",
            input_hint="gelu_in",
            output_hint="gelu_out",
            attrs={"approximate": "tanh" if approximate else "none"},
        )
        return

    (x_var,) = eqn.invars
    (y_var,) = eqn.outvars
    x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("gelu_in"))
    out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("gelu_out"))
    x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
    np_dtype = ir_dtype_to_numpy(x_val.dtype, default=None)
    if np_dtype is None:
        np_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("gelu_out")
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name("gelu_out")

    def scalar(value: float) -> ir.Value:
        const = ctx.bind_const_for_var(object(), np.asarray(value, dtype=np_dtype))
        return cast_param_like(ctx, const, x_val, name_hint="gelu_const_cast")

    def node(op_type: str, *inputs: ir.Value, name: str | None = None) -> ir.Value:
        out = cast(
            ir.Value,
            getattr(ctx.builder, op_type)(
                *inputs, _outputs=[name or ctx.fresh_name(f"gelu_{op_type.lower()}")]
            ),
        )
        out.type = x_val.type
        _stamp_type_and_shape(out, x_shape)
        return out

    if approximate:
        x_cubed = node("Mul", node("Mul", x_val, x_val), x_val)
        inner = node("Add", x_val, node("Mul", x_cubed, scalar(_GELU_TANH_COEFF)))
        tanh = node("Tanh", node("Mul", inner, scalar(_SQRT_2_OVER_PI)))
        cdf = node("Mul", node("Add", tanh, scalar(1.0)), scalar(0.5))
        result = node("Mul", x_val, cdf, name=desired_name)
    else:
        half_x = node("Mul", x_val, scalar(0.5))
        erf = node("Erf", node("Mul", x_val, scalar(_SQRT_HALF)))
        result = node("Mul", half_x, node("Add", erf, scalar(1.0)), name=desired_name)

    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    ctx.bind_value_for_var(y_var, result)


@register_primitive(
    jaxpr_primitive=_GELU_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html",
    onnx=[
        {"component": "Gelu", "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html"},
        {"component": "Erf", "doc": "https://onnx.ai/onnx/operators/onnx__Erf.html"},
        {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.7.1",
    context="primitives.nn",
    component="gelu",
    testcases=[
        {
            "testcase": "jaxnn_gelu",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_1",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_approx",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [(3, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:3x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_exact",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_tanh",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [("B", 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gelu:Bx3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_exact_opset18",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(2, 5)],
            "opset_version": 18,
            "check_onnx_load": True,
            # ONNX Runtime has no float64 Erf kernel.
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Erf:2x5 -> Add:2x5 -> Mul:2x5"],
                must_absent=["Gelu"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_gelu_tanh_opset18",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [("B", 3)],
            "opset_version": 18,
            "check_onnx_load": True,
            "post_check_onnx_graph": EG(
                ["Tanh:Bx3 -> Add:Bx3 -> Mul:Bx3 -> Mul:Bx3"],
                symbols={"B": None},
                must_absent=["Gelu"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "gelu_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jax.nn.gelu(y, approximate=False) ** 2)
            )(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class GeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.gelu`` to ONNX ``Gelu`` (opset >= 20) or its formula."""

    _PRIM: ClassVar[Primitive] = _GELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue, approximate: bool = True) -> ShapedArray:
        del approximate
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lower_gelu(ctx, eqn, approximate=bool(eqn.params.get("approximate", True)))

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
                raise RuntimeError("Original jax.nn.gelu not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "gelu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@GeluPlugin._PRIM.def_impl
def _gelu_impl(x: ArrayLike, approximate: bool = True) -> ArrayLike:
    return _JAX_GELU_ORIG(x, approximate=approximate)


def _gelu_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    approximate: bool = True,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (bd,) = batch_dims
    out = GeluPlugin._PRIM.bind(x, approximate=approximate)
    return out, bd


batching.primitive_batchers[GeluPlugin._PRIM] = _gelu_batch_rule


def _gelu_jvp_rule(
    primals: tuple[ArrayLike, ...], tangents: tuple[ArrayLike, ...], **params: object
) -> tuple[ArrayLike, ArrayLike]:
    approximate = bool(params.get("approximate", True))

    (x,) = primals
    (x_dot,) = tangents
    x_dot = ad.instantiate_zeros(x_dot)

    one = jnp.asarray(1.0, dtype=x.dtype)
    half = jnp.asarray(0.5, dtype=x.dtype)

    if approximate:
        c = jnp.asarray(0.044715, dtype=x.dtype)
        k = jnp.asarray(0.7978845608028654, dtype=x.dtype)  # sqrt(2/pi)

        x_sq = jax.lax.mul(x, x)
        x_cu = jax.lax.mul(x_sq, x)
        inner = jax.lax.add(x, jax.lax.mul(c, x_cu))
        u = jax.lax.mul(k, inner)
        tanh_u = jax.lax.tanh(u)

        primal_out = jax.lax.mul(jax.lax.mul(half, x), jax.lax.add(one, tanh_u))

        three_c = jnp.asarray(0.134145, dtype=x.dtype)  # 3*0.044715
        du_dx = jax.lax.mul(k, jax.lax.add(one, jax.lax.mul(three_c, x_sq)))
        sech2 = jax.lax.sub(one, jax.lax.mul(tanh_u, tanh_u))
        dt_dx = jax.lax.mul(sech2, du_dx)
        deriv = jax.lax.add(
            jax.lax.mul(half, jax.lax.add(one, tanh_u)),
            jax.lax.mul(jax.lax.mul(half, x), dt_dx),
        )
    else:
        inv_sqrt2 = jnp.asarray(0.7071067811865475, dtype=x.dtype)
        inv_sqrt_2pi = jnp.asarray(0.3989422804014327, dtype=x.dtype)

        x_sq = jax.lax.mul(x, x)
        cdf = jax.lax.mul(
            half,
            jax.lax.add(one, jax.lax.erf(jax.lax.mul(x, inv_sqrt2))),
        )
        pdf = jax.lax.mul(
            inv_sqrt_2pi,
            jax.lax.exp(jax.lax.neg(jax.lax.mul(half, x_sq))),
        )
        primal_out = jax.lax.mul(x, cdf)
        deriv = jax.lax.add(cdf, jax.lax.mul(x, pdf))

    tangent_out = jax.lax.mul(x_dot, deriv)
    return primal_out, tangent_out


register_jvp_rule(GeluPlugin._PRIM, _gelu_jvp_rule)
