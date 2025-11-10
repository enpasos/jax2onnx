# jax2onnx/plugins/jax/random/random_bits.py

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import onnx_ir as ir

import jax

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.converter.typing_support import LoweringContextProtocol


def _shape_from_params(shape_param: Sequence[int]) -> tuple[int, ...]:
    dims: list[int] = []
    for dim in shape_param:
        if isinstance(dim, (int, np.integer)):
            dims.append(int(dim))
        else:
            raise NotImplementedError(
                "random_bits lowering currently requires static integer shapes"
            )
    return tuple(dims)


def _scalar_constant(ctx: LoweringContextProtocol, value: float) -> ir.Value:
    arr = np.asarray(value, dtype=np.float32)
    return ctx.builder.add_initializer_from_scalar(
        name=ctx.fresh_name("const"), value=arr
    )


@register_primitive(
    jaxpr_primitive="random_bits",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.random.bits.html",
    onnx=[
        {
            "component": "RandomUniform",
            "doc": "https://onnx.ai/onnx/operators/onnx__RandomUniform.html",
        },
        {
            "component": "Floor",
            "doc": "https://onnx.ai/onnx/operators/onnx__Floor.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="v0.7.2",
    context="primitives.random",
    component="random_bits",
    testcases=[
        {
            "testcase": "random_bits_uint32",
            "callable": lambda: jax.random.bits(
                jax.random.PRNGKey(0), (4,), dtype=jax.numpy.uint32
            ),
            "input_shapes": [],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["RandomUniform -> Mul -> Floor -> Cast"],
            ),
        }
    ],
)
class RandomBitsPlugin(PrimitiveLeafPlugin):
    """Lower ``random_bits`` via RandomUniform + scaling + cast."""

    def lower(
        self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn
    ) -> None:  # type: ignore[override]
        key_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        params = getattr(eqn, "params", {})
        bit_width = int(params.get("bit_width", 32))
        shape_param = params.get("shape", ())
        shape = _shape_from_params(shape_param)

        # Force materialisation of the key so upstream RNG nodes stay live.
        ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("rng_key"))

        uniform_val = ctx.builder.RandomUniform(
            low=0.0,
            high=1.0,
            dtype=int(ir.DataType.FLOAT.value),
            shape=shape,
            _outputs=[ctx.fresh_name("rand_bits_uniform")],
        )
        uniform_val.type = ir.TensorType(ir.DataType.FLOAT)
        _stamp_type_and_shape(uniform_val, shape)

        scale = float(math.ldexp(1.0, bit_width))
        scale_const = _scalar_constant(ctx, scale)
        scaled_val = ctx.builder.Mul(
            uniform_val,
            scale_const,
            _outputs=[ctx.fresh_name("rand_bits_scaled")],
        )
        scaled_val.type = ir.TensorType(ir.DataType.FLOAT)
        _stamp_type_and_shape(scaled_val, shape)

        floored_val = ctx.builder.Floor(
            scaled_val, _outputs=[ctx.fresh_name("rand_bits_floor")]
        )
        floored_val.type = ir.TensorType(ir.DataType.FLOAT)
        _stamp_type_and_shape(floored_val, shape)

        target_dtype = ir.DataType.UINT32 if bit_width <= 32 else ir.DataType.UINT64
        out_value = ctx.builder.Cast(
            floored_val,
            _outputs=[ctx.fresh_name("rand_bits")],
            to=int(target_dtype.value),
        )
        out_value.type = ir.TensorType(target_dtype)
        _stamp_type_and_shape(out_value, shape)

        ctx.bind_value_for_var(out_var, out_value)
