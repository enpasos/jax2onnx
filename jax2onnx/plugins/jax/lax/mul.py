from __future__ import annotations

# jax2onnx/plugins/jax/lax/mul.py
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import core, lax
from onnx import helper
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def _np_dt(x) -> np.dtype:
    return x if isinstance(x, np.dtype) else np.dtype(x)


def _cast_to(
    s: "Jaxpr2OnnxConverter",
    name: str,
    cur_dt: np.dtype,
    tgt_dt: np.dtype,
    *,
    ctx: str,
    shape_hint: tuple[Any, ...],
) -> str:
    if cur_dt == tgt_dt:
        return name
    out = s.builder.get_unique_name(f"{ctx}_cast")
    s.add_node(
        helper.make_node(
            "Cast",
            [name],
            [out],
            to=int(s.builder._numpy_dtype_to_onnx(tgt_dt)),
            name=s.builder.get_unique_name(f"{ctx}_Cast"),
        )
    )
    s.add_shape_info(out, shape_hint, tgt_dt)
    return out


@register_primitive(
    jaxpr_primitive=lax.mul_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.mul.html",
    onnx=[{"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"}],
    since="v0.1.0",
    context="primitives.lax",
    component="mul",
    testcases=[
        {
            "testcase": "mul_test1",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "mul_test2",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(2, 2), (2, 2)],
        },
        {
            "testcase": "mul_pyfloat_promotes_to_array_dtype_f64",
            "callable": lambda x: x * 1.5,
            "input_values": [np.array([1.0, 2.0], dtype=np.float64)],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "mul_scalar_broadcast_promote_to_f64",
            "callable": lambda x: (x.astype(np.float64)) * 1.5,
            "input_values": [np.array([1.0, 2.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
    ],
)
class MulPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(x: core.ShapedArray, y: core.ShapedArray, **params):
        out_dtype = np.promote_types(x.dtype, y.dtype)
        return core.ShapedArray(np.broadcast_shapes(x.shape, y.shape), out_dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        x_v, y_v = node_inputs
        out_v = node_outputs[0]

        x_name = s.get_name(x_v)
        y_name = s.get_name(y_v)
        out_name = s.get_name(out_v)

        # JAX-style dtype promotion
        x_dt = _np_dt(x_v.aval.dtype)
        y_dt = _np_dt(y_v.aval.dtype)
        tgt = _np_dt(np.promote_types(x_dt, y_dt))

        # Cast both inputs to target dtype (via aval dtypes)
        x_cast = _cast_to(s, x_name, x_dt, tgt, ctx="Mul", shape_hint=x_v.aval.shape)
        y_cast = _cast_to(s, y_name, y_dt, tgt, ctx="Mul", shape_hint=y_v.aval.shape)

        # Rely on ONNX implicit broadcasting
        s.add_node(
            helper.make_node(
                "Mul",
                [x_cast, y_cast],
                [out_name],
                name=s.builder.get_unique_name("Mul"),
            )
        )
        s.add_shape_info(out_name, out_v.aval.shape, tgt)
