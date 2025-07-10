# file: jax2onnx/plugins/jax/lax/select.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
import jax.numpy as jnp
from jax import core, lax
from jax.extend.core import Var
from onnx import helper, TensorProto
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.select")

@register_primitive(
    jaxpr_primitive="select",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html",
    onnx=[{"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}],
    since="v0.7.1",
    context="primitives.lax",
    component="select",
    testcases=[
        {
            "testcase": "select_simple",
            "callable": lambda c, x, y: lax.select(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "input_dtypes": [jnp.bool_, jnp.float32, jnp.float32],
            "expected_output_shapes": [(3,)],
        },
        {
            # scalar else-branch scenario (e.g., GPT attention masking)
            "testcase": "select_mask_scores_literal_else",
            "callable": lambda mask, scores: lax.select(mask, scores, -1e9),
            "input_shapes": [("B", 1, "T", "T"), ("B", 12, "T", "T")],
            "input_dtypes": [jnp.bool_, jnp.float32],
            "expected_output_shapes": [("B", 12, "T", "T")],
        },
        {
            # Reproduces GPT attention masking: cond ? scores : -1e9
            "testcase": "select_scalar_else_pyfloat",
            "callable": lambda c, x, y: lax.select(c, x, y),
            "input_values": [
                np.random.choice([True, False], size=(2, 4, 5)),
                np.random.randn(2, 4, 5).astype(np.float32),
                np.full((2, 4, 5), -1e9, dtype=np.float32),
            ],
            "expected_output_shapes": [(2, 4, 5)],
            "expected_output_dtypes": [np.float32],
        },
    ],
)
class SelectPlugin(PrimitiveLeafPlugin):
    """Lower lax.select to ONNX Where, handling scalar broadcasts."""

    # Abstract evaluation
    @staticmethod
    def abstract_eval(
        cond_av: core.AbstractValue,
        x_av: core.AbstractValue,
        y_av: core.AbstractValue,
        **kwargs,
    ) -> core.AbstractValue:
        promoted_dtype = jnp.promote_types(x_av.dtype, y_av.dtype)
        shape_xy = np.broadcast_shapes(x_av.shape, y_av.shape)
        out_shape = np.broadcast_shapes(cond_av.shape, shape_xy)
        return core.ShapedArray(out_shape, promoted_dtype)

    # ONNX Conversion
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        cond_v, x_v, y_v = node_inputs
        out_v = node_outputs[0]

        cond_name = s.get_name(cond_v)
        x_name = s.get_name(x_v)
        y_name = s.get_name(y_v)
        out_name = s.get_name(out_v)

        # Ensure BOOL condition in ONNX
        if cond_v.aval.dtype != np.bool_:
            cast_out = s.builder.get_unique_name("select_cond_cast")
            s.builder.add_node(
                helper.make_node("Cast", [cond_name], [cast_out], to=TensorProto.BOOL)
            )
            s.add_shape_info(cast_out, cond_v.aval.shape, np.bool_)
            cond_name = cast_out

        # Scalar literal broadcasting for y
        if "y" in params and np.isscalar(params["y"]):
            scalar_val = params["y"]
            shape_of_x = s.builder.get_unique_name("shape_of_x")
            y_broadcasted = s.builder.get_unique_name("y_broadcasted")

            # Shape of true branch (x)
            s.builder.add_node(helper.make_node("Shape", [x_name], [shape_of_x]))

            # Constant scalar tensor
            scalar_tensor = helper.make_tensor(
                name=s.builder.get_unique_name("scalar_tensor"),
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[scalar_val],
            )

            # Broadcast scalar
            s.builder.add_node(
                helper.make_node(
                    "ConstantOfShape",
                    [shape_of_x],
                    [y_broadcasted],
                    value=scalar_tensor,
                )
            )
            y_name = y_broadcasted

        # Emit ONNX Where
        s.add_node(helper.make_node("Where", [cond_name, x_name, y_name], [out_name]))
        s.add_shape_info(out_name, out_v.aval.shape, out_v.aval.dtype)

    # --------------------------------------------------------------------- #
    # Runtime patch so that `lax.select(mask, scores, -1e9)` works even when
    # the else-branch is a scalar.  Nothing here touches `select_n_p`.
    # --------------------------------------------------------------------- #
    @staticmethod
    def patch_info():
        """
        Returns the instructions that the jax-to-onnx plugin system uses to
        monkey-patch `jax.lax.select` at import time.
        """

        def _to_array_if_scalar(val, dtype):
            # Promote python / NumPy scalar or 0-D array to 0-D jax array
            if np.isscalar(val) or (isinstance(val, jnp.ndarray) and val.ndim == 0):
                val = jnp.asarray(val, dtype=dtype)
            return val

        def patched_select(pred, x, y):
            """
            Drop-in replacement for `lax.select`:

              • If either branch is a scalar, turn it into a 0-D array whose
                dtype matches the other branch.
              • Delegate to `jnp.where`, which already supports full
                NumPy-style broadcasting, so shapes like
                pred:(B,1,T,T)  x/y:(B,12,T,T) are handled naturally.
            """
            x = _to_array_if_scalar(x, y.dtype if hasattr(y, "dtype") else None)
            y = _to_array_if_scalar(y, x.dtype if hasattr(x, "dtype") else None)
            return jnp.where(pred, x, y)

        return {
            "patch_targets": [lax],
            "target_attribute": "select",
            "patch_function": lambda _orig: patched_select,
        }
