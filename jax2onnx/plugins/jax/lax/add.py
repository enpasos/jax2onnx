# jax2onnx/plugins/jax/lax/add.py
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any, Tuple
import numpy as np
from jax import core, lax
from onnx import helper, mapping as onnx_map
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

def _np_dt(x) -> np.dtype:
    return x if isinstance(x, np.dtype) else np.dtype(x)

def _aval_dtype_shape(s: "Jaxpr2OnnxConverter", v: Any) -> Tuple[np.dtype, Tuple[Any, ...]]:
    # Works for Vars and Literals
    if hasattr(v, "aval"):  # normal JAX Var
        return _np_dt(v.aval.dtype), tuple(v.aval.shape)
    # Literal or already-named value: fall back to builder metadata
    name = s.get_name(v)
    meta = s.builder.value_info_metadata.get(name)
    if meta is not None:
        shp, dt = meta
        np_dt = _np_dt(onnx_map.TENSOR_TYPE_TO_NP_TYPE[dt]) if isinstance(dt, int) else _np_dt(dt)
        return np_dt, tuple(shp)
    # Ultimate fallback
    return np.dtype(np.float32), ()

def _cast_to(s: "Jaxpr2OnnxConverter", name: str, cur_dt: np.dtype, tgt_dt: np.dtype, *,
             ctx: str, shape_hint: tuple[Any, ...]) -> str:
    if cur_dt == tgt_dt:
        return name
    out = s.builder.get_unique_name(f"{ctx}_cast")
    s.add_node(helper.make_node(
        "Cast", [name], [out],
        to=int(s.builder._numpy_dtype_to_onnx(tgt_dt)),
        name=s.builder.get_unique_name(f"{ctx}_Cast"),
    ))
    s.add_shape_info(out, shape_hint, tgt_dt)
    return out

@register_primitive(
    jaxpr_primitive=lax.add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[{"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"}],
    since="v0.2.0",
    context="primitives.lax",
    component="add",
    testcases=[{"testcase": "add", "callable": lambda x1, x2: x1 + x2, "input_shapes": [(3,), (3,)]}],
)
class AddPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(x: core.ShapedArray, y: core.ShapedArray, **params):
        out_dtype = np.promote_types(x.dtype, y.dtype)
        return core.ShapedArray(np.broadcast_shapes(x.shape, y.shape), out_dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter",
                node_inputs: Sequence[Any], node_outputs: Sequence[Any], params: dict[str, Any]):
        x_v, y_v = node_inputs
        out_v = node_outputs[0]

        x_name = s.get_name(x_v)
        y_name = s.get_name(y_v)
        out_name = s.get_name(out_v)

        # Robust dtype/shape collection (works for Vars and Literals)
        x_dt, x_shape = _aval_dtype_shape(s, x_v)
        y_dt, y_shape = _aval_dtype_shape(s, y_v)

        # Let the builder coerce *initializers* to a common numeric dtype when mixed
        x_name, y_name, tgt_enum = s.builder.coerce_binop_literals(x_name, y_name)

        # Decide target dtype: prefer builder’s choice, otherwise NumPy promotion
        if tgt_enum not in (None, 0):
            tgt_np = _np_dt(onnx_map.TENSOR_TYPE_TO_NP_TYPE[tgt_enum])
        else:
            tgt_np = _np_dt(np.promote_types(x_dt, y_dt))

        # Cast both sides to the target (Add in ONNX requires exact type match)
        x_cast = _cast_to(s, x_name, x_dt, tgt_np, ctx="Add", shape_hint=x_shape)
        y_cast = _cast_to(s, y_name, y_dt, tgt_np, ctx="Add", shape_hint=y_shape)

        s.add_node(helper.make_node("Add", [x_cast, y_cast], [out_name],
                                    name=s.builder.get_unique_name("Add")))
        s.add_shape_info(out_name, np.broadcast_shapes(x_shape, y_shape), tgt_np)
