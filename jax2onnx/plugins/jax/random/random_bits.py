# file: jax2onnx/plugins/jax/random/random_bits.py
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

import jax

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:
    from jax2onnx.converter.ir_context import IRContext


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


def _scalar_constant(ctx: "IRContext", value: float) -> ir.Value:
    from jax2onnx.converter.ir_context import IRContext  # local import to avoid cycle

    if not isinstance(ctx, IRContext):  # pragma: no cover - defensive
        raise TypeError("ctx must be an IRContext")
    arr = np.asarray(value, dtype=np.float32)
    const = ir.Value(
        name=ctx.fresh_name("const"),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(()),
        const_value=ir.tensor(arr),
    )
    ctx.add_node(
        ir.Node(
            op_type="Constant",
            domain="",
            inputs=[],
            outputs=[const],
            name=ctx.fresh_name("Constant"),
            attributes=[IRAttr("value", IRAttrType.TENSOR, ir.tensor(arr))],
        )
    )
    return const


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

    def lower(self, ctx, eqn):  # type: ignore[override]
        from jax2onnx.converter.ir_context import IRContext  # local import

        if not isinstance(ctx, IRContext):  # pragma: no cover - defensive
            raise TypeError("Expected IRContext")

        key_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        params = getattr(eqn, "params", {})
        bit_width = int(params.get("bit_width", 32))
        shape_param = params.get("shape", ())
        shape = _shape_from_params(shape_param)

        # Force materialisation of the key so upstream RNG nodes stay live.
        ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("rng_key"))

        uniform_val = ir.Value(
            name=ctx.fresh_name("rand_bits_uniform"),
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(shape),
        )
        ctx.add_node(
            ir.Node(
                op_type="RandomUniform",
                domain="",
                inputs=[],
                outputs=[uniform_val],
                name=ctx.fresh_name("RandomUniform"),
                attributes=[
                    IRAttr("low", IRAttrType.FLOAT, 0.0),
                    IRAttr("high", IRAttrType.FLOAT, 1.0),
                    IRAttr("dtype", IRAttrType.INT, int(ir.DataType.FLOAT.value)),
                    IRAttr("shape", IRAttrType.INTS, shape),
                ],
            )
        )

        scale = float(math.ldexp(1.0, bit_width))
        scale_const = _scalar_constant(ctx, scale)
        scaled_val = ir.Value(
            name=ctx.fresh_name("rand_bits_scaled"),
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(shape),
        )
        ctx.add_node(
            ir.Node(
                op_type="Mul",
                domain="",
                inputs=[uniform_val, scale_const],
                outputs=[scaled_val],
                name=ctx.fresh_name("Mul"),
            )
        )

        floored_val = ir.Value(
            name=ctx.fresh_name("rand_bits_floor"),
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(shape),
        )
        ctx.add_node(
            ir.Node(
                op_type="Floor",
                domain="",
                inputs=[scaled_val],
                outputs=[floored_val],
                name=ctx.fresh_name("Floor"),
            )
        )

        target_dtype = ir.DataType.UINT32 if bit_width <= 32 else ir.DataType.UINT64
        out_value = ir.Value(
            name=ctx.fresh_name("rand_bits"),
            type=ir.TensorType(target_dtype),
            shape=ir.Shape(shape),
        )
        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[floored_val],
                outputs=[out_value],
                name=ctx.fresh_name("Cast"),
                attributes=[
                    IRAttr("to", IRAttrType.INT, int(target_dtype.value)),
                ],
            )
        )

        ctx.bind_value_for_var(out_var, out_value)
