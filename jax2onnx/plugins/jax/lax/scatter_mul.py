# jax2onnx/plugins/jax/lax/scatter_mul.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import lax, core
from jax.lax import ScatterDimensionNumbers, GatherScatterMode
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=lax.scatter_mul_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_mul.html",
    onnx=[{"component": "ScatterND", "attributes": ["reduction='mul'"]}],
    since="v0.5.3",
    context="primitives.lax",
    component="scatter_mul",
    testcases=[
        # ... (testcases omitted for brevity)
    ],
)
class ScatterMulPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,
        update_consts,
        *,
        dimension_numbers: ScatterDimensionNumbers,
        indices_are_sorted: bool,
        unique_indices: bool,
        mode: GatherScatterMode | None,
    ):
        return core.ShapedArray(operand.shape, operand.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        operand_v, indices_v, updates_v = node_inputs
        out_v = node_outputs[0]

        operand_name = s.get_name(operand_v)
        out_name = s.get_name(out_v)
        operand_shape = tuple(operand_v.aval.shape)
        operand_dtype_np = np.dtype(operand_v.aval.dtype)

        # Prepare indices and updates
        idx_name, idx_depth, num_updates = self._prepare_indices(s, indices_v)
        upd_name, window_shape = self._prepare_updates(
            s, updates_v, params, num_updates
        )
        dnums: ScatterDimensionNumbers = params["dimension_numbers"]

        # If windowed scatter, implement as GatherND -> Mul -> ScatterND
        if dnums.update_window_dims:
            # Gather existing slices
            gathered = s.get_unique_name("gathered_slices")
            s.add_node(
                helper.make_node(
                    "GatherND",
                    inputs=[operand_name, idx_name],
                    outputs=[gathered],
                    name=s.get_unique_name("gathernd_scatter_mul"),
                )
            )
            s.add_shape_info(gathered, (num_updates, *window_shape), operand_dtype_np)

            # Multiply
            multiplied = s.get_unique_name("multiplied_slices")
            s.add_node(
                helper.make_node(
                    "Mul",
                    inputs=[gathered, upd_name],
                    outputs=[multiplied],
                    name=s.get_unique_name("mul_scatter_mul"),
                )
            )
            s.add_shape_info(multiplied, (num_updates, *window_shape), operand_dtype_np)

            # Scatter multiplied slices back (replace)
            scatter_node = helper.make_node(
                "ScatterND",
                inputs=[operand_name, idx_name, multiplied],
                outputs=[out_name],
                name=s.get_unique_name("scatter_mul_windowed"),
            )
            s.add_node(scatter_node)
            s.add_shape_info(out_name, operand_shape, operand_dtype_np)
        else:
            # Direct scatter_mul via reduction attribute
            scatter_node = helper.make_node(
                "ScatterND",
                inputs=[operand_name, idx_name, upd_name],
                outputs=[out_name],
                name=s.get_unique_name("scatter_mul"),
                reduction="mul",
            )
            s.add_node(scatter_node)
            s.add_shape_info(out_name, operand_shape, operand_dtype_np)

    @staticmethod
    def _prepare_indices(s: "Jaxpr2OnnxConverter", indices_v):
        name = s.get_name(indices_v)
        shape = list(indices_v.aval.shape)
        dtype_np = np.dtype(indices_v.aval.dtype)
        # Cast to int64
        if dtype_np != np.int64:
            casted = s.get_unique_name(f"{name}_int64")
            s.add_node(
                helper.make_node(
                    "Cast",
                    [name],
                    [casted],
                    to=TensorProto.INT64,
                    name=s.get_unique_name("cast_to_int64"),
                )
            )
            name = casted
        # Flatten to (num_updates, index_depth)
        index_depth = shape[-1]
        batch_dims = shape[:-1]
        num_updates = int(np.prod(batch_dims)) if batch_dims else 1
        target_shape = (num_updates, index_depth)
        if tuple(shape) != target_shape:
            reshaped = s.get_unique_name(f"{name}_reshaped")
            shape_const = np.array(target_shape, dtype=np.int64)
            const_name = s.get_constant_name(shape_const)
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [name, const_name],
                    [reshaped],
                    name=s.get_unique_name("reshape_indices"),
                )
            )
            name = reshaped
        return name, index_depth, num_updates

    @staticmethod
    def _prepare_updates(
        s: "Jaxpr2OnnxConverter",
        updates_v,
        params,
        num_updates,
    ):
        name = s.get_name(updates_v)
        shape = list(updates_v.aval.shape)
        dnums: ScatterDimensionNumbers = params["dimension_numbers"]
        batch_dims = len(shape) - len(dnums.update_window_dims)
        window_shape = shape[batch_dims:]
        target_shape = (num_updates, *window_shape)
        if tuple(shape) != target_shape:
            reshaped = s.get_unique_name("updates_reshaped")
            shape_const = np.array(target_shape, dtype=np.int64)
            const_name = s.get_constant_name(shape_const)
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [name, const_name],
                    [reshaped],
                    name=s.get_unique_name("reshape_updates"),
                )
            )
            name = reshaped
        return name, tuple(window_shape)
