# jax2onnx/plugins/jax/lax/add.py

from typing import TYPE_CHECKING, Optional
import jax
import numpy as np
from jax2onnx.plugins._axis0_utils import (
    maybe_expand_binary_axis0,
    stamp_axis0_binary_result,
)
from jax2onnx.plugins._loop_extent_meta import (
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # for type hints


def lower_add(ctx, eqn) -> None:
    """Shared lowering routine for lax/jnp Add plugins."""

    x_var, y_var = eqn.invars
    out_var = eqn.outvars[0]

    prefer_dt: Optional[np.dtype] = np.dtype(getattr(x_var.aval, "dtype", np.float32))

    a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("add_lhs"))
    b_val = ctx.get_value_for_var(
        y_var, name_hint=ctx.fresh_name("add_rhs"), prefer_np_dtype=prefer_dt
    )
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("add_out"))
    a_val, b_val, override = maybe_expand_binary_axis0(
        ctx, a_val, b_val, out_spec, out_var
    )
    result = ctx.builder.Add(a_val, b_val, _outputs=[out_spec.name])
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    stamp_axis0_binary_result(result, out_var, out_spec, override)
    if override is not None:
        set_axis0_override(result, override)
    propagate_axis0_override(a_val, result)
    propagate_axis0_override(b_val, result)
    ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=jax.lax.add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[
        {
            "component": "Add",
            "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="add",
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x1, x2: x1 + x2,
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Add:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "add_const",
            "callable": lambda x: x + 1.0,
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Add:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "add_complex64",
            "callable": lambda x, y: x + y,
            "input_values": [
                np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64),
                np.array([-1.5 + 0.5j, 2.0 + 1.25j], dtype=np.complex64),
            ],
            "expected_output_dtypes": [np.complex64],
            "skip_numeric_validation": True,  # ORT CPU build lacks native complex arithmetic
            "post_check_onnx_graph": EG(
                ["Add:2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class AddPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lower_add(ctx, eqn)
