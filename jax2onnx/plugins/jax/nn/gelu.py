# file: jax2onnx/plugins/jax/nn/gelu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper
from onnx import TensorProto
import numpy as _np
import math

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.gelu_p = Primitive("jax.nn.gelu")
jax.nn.gelu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.gelu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html",
    onnx=[
        {
            "component": "Gelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="gelu",
    testcases=[
        {
            "testcase": "jaxnn_gelu",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_gelu_1",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(2, 5)],
        },
        {
            "testcase": "jaxnn_gelu_approx",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [(3, 3)],
        },
    ],
)
class JaxGeluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.gelu calls to the ONNX Gelu operator.
    Supports both exact and approximate (tanh-based) variants.
    """

    @staticmethod
    def abstract_eval(x, approximate=True):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        approximate = params.get("approximate", True)
        # ONNX expects 'tanh' for approximate=True, 'none' otherwise
        approximation = "tanh" if approximate else "none"

        # figure out dtype & shape
        dtype = input_var.aval.dtype
        shape = input_var.aval.shape

        if dtype == _np.float32:
            # native Gelu for float32
            gelu_node = helper.make_node(
                "Gelu",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("gelu"),
                approximate=approximation,
            )
            s.add_node(gelu_node)
            s.add_shape_info(output_name, shape, dtype)
        elif dtype == _np.float64:
            # Improved DOUBLE-precision path: explicit subgraph with DOUBLE constants and full VI coverage
            # exact:     0.5*x*(1 + erf(x / sqrt(2)))
            # approximate: 0.5*x*(1 + tanh( sqrt(2/pi)*( x + 0.044715*x^3 ) ))
            #
            # We keep all constants as DOUBLE to avoid f32 roundoff at tight tolerances.

            def _const_scalar(name_hint: str, value: float):
                out = s.get_unique_name(f"{name_hint}_out")
                s.add_node(
                    helper.make_node(
                        "Constant",
                        # scalar(double)
                        inputs=[],
                        outputs=[out],
                        name=s.get_unique_name(f"{name_hint}_const"),
                        value=helper.make_tensor(
                            s.get_unique_name(f"{name_hint}_t"),
                            TensorProto.DOUBLE,
                            [],
                            [float(value)],
                        ),
                    )
                )
                s.add_shape_info(out, (), _np.float64)
                return out

            def _const_i64(name_hint: str, value: int):
                out = s.get_unique_name(f"{name_hint}_out")
                s.add_node(
                    helper.make_node(
                        "Constant",
                        # scalar(int64)
                        inputs=[],
                        outputs=[out],
                        name=s.get_unique_name(f"{name_hint}_const"),
                        value=helper.make_tensor(
                            s.get_unique_name(f"{name_hint}_t"),
                            TensorProto.INT64,
                            [],
                            [int(value)],
                        ),
                    )
                )
                s.add_shape_info(out, (), _np.int64)
                return out

            half = _const_scalar("gelu_half", 0.5)
            one = _const_scalar("gelu_one", 1.0)

            if approximate:
                # y = 0.5 * x * (1 + tanh( sqrt(2/pi) * ( x + 0.044715*x^3 ) ) )
                s2opi = _const_scalar("gelu_s2opi", math.sqrt(2.0 / math.pi))
                k044 = _const_scalar("gelu_k", 0.044715)
                three = _const_i64("gelu_three", 3)

                x3 = s.get_unique_name("gelu_x3")
                s.add_node(
                    helper.make_node(
                        "Pow",
                        inputs=[input_name, three],
                        outputs=[x3],
                        name=s.get_unique_name("gelu_pow3"),
                    )
                )
                s.add_shape_info(x3, shape, _np.float64)

                kx3 = s.get_unique_name("gelu_kx3")
                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[k044, x3],
                        outputs=[kx3],
                        name=s.get_unique_name("gelu_kx3"),
                    )
                )
                s.add_shape_info(kx3, shape, _np.float64)

                inner = s.get_unique_name("gelu_inner")
                s.add_node(
                    helper.make_node(
                        "Add",
                        inputs=[input_name, kx3],
                        outputs=[inner],
                        name=s.get_unique_name("gelu_inner"),
                    )
                )
                s.add_shape_info(inner, shape, _np.float64)

                scaled = s.get_unique_name("gelu_scaled")
                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[s2opi, inner],
                        outputs=[scaled],
                        name=s.get_unique_name("gelu_scale"),
                    )
                )
                s.add_shape_info(scaled, shape, _np.float64)

                th = s.get_unique_name("gelu_tanh")
                s.add_node(
                    helper.make_node(
                        "Tanh",
                        inputs=[scaled],
                        outputs=[th],
                        name=s.get_unique_name("gelu_tanh"),
                    )
                )
                s.add_shape_info(th, shape, _np.float64)

                onep = s.get_unique_name("gelu_one_plus")
                s.add_node(
                    helper.make_node(
                        "Add",
                        inputs=[one, th],
                        outputs=[onep],
                        name=s.get_unique_name("gelu_1p"),
                    )
                )
                s.add_shape_info(onep, shape, _np.float64)

                halfx = s.get_unique_name("gelu_halfx")
                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[half, input_name],
                        outputs=[halfx],
                        name=s.get_unique_name("gelu_halfx"),
                    )
                )
                s.add_shape_info(halfx, shape, _np.float64)

                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[halfx, onep],
                        outputs=[output_name],
                        name=s.get_unique_name("gelu_out"),
                    )
                )
                s.add_shape_info(output_name, shape, _np.float64)
            else:
                # y = 0.5 * x * (1 + erf(x / sqrt(2)))
                # ORT CPU lacks Erf(double), so evaluate Erf in float32 and cast back.
                invs2 = _const_scalar("gelu_inv_sqrt2", 1.0 / math.sqrt(2.0))
                xs = s.get_unique_name("gelu_x_scaled")
                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[input_name, invs2],
                        outputs=[xs],
                        name=s.get_unique_name("gelu_scale"),
                    )
                )
                s.add_shape_info(xs, shape, _np.float64)

                # Cast to float32 → Erf → Cast back to float64
                xs_f32 = s.get_unique_name("gelu_x_scaled_f32")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[xs],
                        outputs=[xs_f32],
                        name=s.get_unique_name("gelu_cast_to_f32"),
                        to=TensorProto.FLOAT,
                    )
                )
                s.add_shape_info(xs_f32, shape, _np.float32)

                erf_f32 = s.get_unique_name("gelu_erf_f32")
                s.add_node(
                    helper.make_node(
                        "Erf",
                        inputs=[xs_f32],
                        outputs=[erf_f32],
                        name=s.get_unique_name("gelu_erf"),
                    )
                )
                s.add_shape_info(erf_f32, shape, _np.float32)

                erf = s.get_unique_name("gelu_erf_f64")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[erf_f32],
                        outputs=[erf],
                        name=s.get_unique_name("gelu_cast_to_f64"),
                        to=TensorProto.DOUBLE,
                    )
                )
                s.add_shape_info(erf, shape, _np.float64)

                onep = s.get_unique_name("gelu_one_plus")
                s.add_node(
                    helper.make_node(
                        "Add",
                        inputs=[one, erf],
                        outputs=[onep],
                        name=s.get_unique_name("gelu_1p"),
                    )
                )
                s.add_shape_info(onep, shape, _np.float64)
                halfx = s.get_unique_name("gelu_halfx")
                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[half, input_name],
                        outputs=[halfx],
                        name=s.get_unique_name("gelu_halfx"),
                    )
                )
                s.add_shape_info(halfx, shape, _np.float64)
                s.add_node(
                    helper.make_node(
                        "Mul",
                        inputs=[halfx, onep],
                        outputs=[output_name],
                        name=s.get_unique_name("gelu_out"),
                    )
                )
                s.add_shape_info(output_name, shape, _np.float64)
        else:
            # Fallback for other types: Cast→Gelu(f32)→Cast back
            cast_in = s.get_unique_name("gelu_cast_in")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_in],
                    name=s.get_unique_name("cast_in"),
                    to=TensorProto.FLOAT,
                )
            )
            s.add_shape_info(cast_in, shape, _np.float32)

            gelu_f32 = s.get_unique_name("gelu_f32")
            s.add_node(
                helper.make_node(
                    "Gelu",
                    inputs=[cast_in],
                    outputs=[gelu_f32],
                    name=s.get_unique_name("gelu"),
                    approximate=approximation,
                )
            )
            s.add_shape_info(gelu_f32, shape, _np.float32)

            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[gelu_f32],
                    outputs=[output_name],
                    name=s.get_unique_name("cast_out"),
                    to=TensorProto.DOUBLE,
                )
            )
            # Best effort: if we landed here, treat as double output for parity with old path
            s.add_shape_info(output_name, shape, _np.float64)

    @staticmethod
    def get_monkey_patch():
        def patched_gelu(x, approximate=True):
            return jax.nn.gelu_p.bind(x, approximate=approximate)

        return patched_gelu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxGeluPlugin.get_monkey_patch(),
            "target_attribute": "gelu",
        }


def gelu_batching_rule(batched_args, batch_dims, *, approximate):
    """
    Batching rule for jax.nn.gelu.
    Since GELU is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.gelu_p.bind(x, approximate=approximate)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.gelu_p.def_abstract_eval(JaxGeluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.gelu_p] = gelu_batching_rule
