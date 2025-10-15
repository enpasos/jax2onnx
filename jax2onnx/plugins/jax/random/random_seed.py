# jax2onnx/plugins/jax/random/random_seed.py

"""Lowering for JAX PRNG seed primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

import jax

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - import guard for typing only
    from jax2onnx.converter.ir_context import IRContext


def _shape_dims(shape: ir.Shape | tuple[int, ...]) -> tuple:
    dims = getattr(shape, "dims", None)
    if dims is None:
        try:
            return tuple(shape)
        except TypeError:
            return ()
    return tuple(dims)


def _const_array(ctx: "IRContext", arr: np.ndarray, *, name_hint: str) -> ir.Value:
    """Materialize a constant initializer that survives both graph + function modes."""

    enum = _dtype_to_ir(arr.dtype, ctx.builder.enable_double_precision)
    value = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(enum),
        shape=ir.Shape(tuple(int(d) for d in arr.shape)),
        const_value=ir.tensor(arr),
    )
    ctx._initializers.append(value)
    return value


def _unsqueeze(ctx: "IRContext", value: ir.Value, axis: int) -> ir.Value:
    axes = _const_array(
        ctx, np.asarray([axis], dtype=np.int64), name_hint="unsqueeze_axes"
    )
    base_dims = _shape_dims(value.shape)
    squeezed_shape = (1,) + tuple(
        int(d) if isinstance(d, (int, np.integer)) else d for d in base_dims
    )
    squeezed = ctx.builder.Unsqueeze(
        value, axes, _outputs=[ctx.fresh_name("unsqueeze")]
    )
    squeezed.type = value.type
    squeezed.shape = ir.Shape(squeezed_shape)
    return squeezed


@register_primitive(
    jaxpr_primitive="random_seed",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        },
    ],
    since="v0.2.0",
    context="primitives.random",
    component="random_seed",
    testcases=[
        {
            "testcase": "random_seed_basic",
            "callable": lambda seed: jax.random.PRNGKey(seed),
            "input_shapes": [()],
            "input_dtypes": [np.int32],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.dtype(np.uint32)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Cast -> Unsqueeze:1 -> Concat:2",
                        "inputs": {0: {"const": 0.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        }
    ],
)
class RandomSeedPlugin(PrimitiveLeafPlugin):
    """Lower ``random_seed`` to a deterministic uint32 key pair [0, seed]."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        seed_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        seed_value = ctx.get_value_for_var(seed_var, name_hint=ctx.fresh_name("seed"))

        cast_target = ctx.builder.Cast(
            seed_value,
            _outputs=[ctx.fresh_name("seed_u32")],
            to=int(ir.DataType.UINT32.value),
        )
        cast_target.type = ir.TensorType(ir.DataType.UINT32)
        cast_target.shape = seed_value.shape

        seed_vector = _unsqueeze(ctx, cast_target, axis=0)

        zero_vector = _const_array(
            ctx, np.asarray([0], dtype=np.uint32), name_hint="prng_zero"
        )

        key_value = ctx.builder.Concat(
            zero_vector,
            seed_vector,
            axis=0,
            _outputs=[ctx.fresh_name("prng_key")],
        )
        key_value.type = ir.TensorType(ir.DataType.UINT32)
        key_value.shape = ir.Shape((2,))

        ctx.bind_value_for_var(out_var, key_value)


@register_primitive(
    jaxpr_primitive="random_unwrap",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.random.key.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.2.0",
    context="primitives.random",
    component="random_unwrap",
    testcases=[],
)
class RandomUnwrapPlugin(PrimitiveLeafPlugin):
    """Forward the uint32 key produced by ``random_seed``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        key_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        key_value = ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("prng"))
        ctx.bind_value_for_var(out_var, key_value)


@register_primitive(
    jaxpr_primitive="random_wrap",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.random.wrap_key.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.7.2",
    context="primitives.random",
    component="random_wrap",
    testcases=[],
)
class RandomWrapPlugin(PrimitiveLeafPlugin):
    """No-op wrapper around PRNG keys."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        key_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        key_value = ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("prng"))
        ctx.bind_value_for_var(out_var, key_value)
