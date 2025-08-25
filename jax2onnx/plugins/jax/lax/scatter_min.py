from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
import jax.numpy as jnp
from jax import lax, core
from jax.lax import ScatterDimensionNumbers, GatherScatterMode

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from .scatter_converters import convert_lax_scatter_min
import logging

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter_min")


@register_primitive(
    jaxpr_primitive=lax.scatter_min_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_min.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
            "attributes": ["reduction='min'"],
        }
    ],
    since="v0.7.5",
    context="primitives.lax",
    component="scatter_min",
    testcases=[
        {
            "testcase": "scatter_min_simple_1d",
            "callable": lambda operand, indices, updates: lax.scatter_min(
                operand,
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                np.array([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                np.array([[1], [3]], dtype=np.int32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
        },
        {
            "testcase": "scatter_min_window_2d_operand_1d_indices",
            "callable": lambda operand, indices, updates: lax.scatter_min(
                operand,
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(1,),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                np.array([[5.0, 5.0, 5.0], [9.0, 9.0, 9.0]], dtype=np.float32),
                np.array([[0]], dtype=np.int32),
                np.array([[4.0, 3.0, 2.0]], dtype=np.float32),
            ],
        },
        {
            "testcase": "scatter_min_batch_updates_1d_operand",
            "callable": lambda operand, indices, updates: lax.scatter_min(
                operand,
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                np.full((5,), 10.0, dtype=np.float32),
                np.array([[[0], [1]], [[0], [2]]], dtype=np.int32),
                np.array([[7.0, 8.0], [1.0, 2.0]], dtype=np.float32),
            ],
        },
        # fp64 regression-style checks (shape/dtype paths), mirroring scatter_add coverage
        {
            "testcase": "scatter_min_fp64_dtype_path_check",
            "callable": (
                lambda: lax.scatter_min(
                    jnp.zeros((4, 3), dtype=jnp.float64),
                    jnp.array([[0, 0], [2, 1]], dtype=jnp.int32),
                    jnp.array([9.0, 8.0], dtype=jnp.float64),
                    dimension_numbers=lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0, 1),
                        scatter_dims_to_operand_dims=(0, 1),
                    ),
                )
            ),
            "input_shapes": [],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "scatter_min_depth2_helper_regression_fp64",
            "callable": (
                lambda: lax.scatter_min(
                    jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                    jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),  # (N,2)
                    jnp.ones((2, 4, 5), dtype=jnp.float64),  # window (4,5)
                    dimension_numbers=lax.ScatterDimensionNumbers(
                        update_window_dims=(1, 2),
                        inserted_window_dims=(0, 1),
                        scatter_dims_to_operand_dims=(0, 1),
                    ),
                )
            ),
            "input_shapes": [],
            "run_only_f64_variant": True,
        },
    ],
)
class ScatterMinPlugin(PrimitiveLeafPlugin):
    """
    ONNX conversion for the lax.scatter_min primitive.
    """

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
        logger.info(
            "Converting lax.scatter_min with dimension_numbers: %s",
            params.get("dimension_numbers"),
        )

        # Style B: delegate to the shared converter
        class _Eqn:
            # mypy: declare the attribute so assignments are type-checked
            params: dict[str, Any]

        _e = _Eqn()
        _e.params = params
        convert_lax_scatter_min(
            s,
            _e,
            (operand_v, indices_v, updates_v),
            (out_v,),
        )
        logger.debug(
            "[ScatterMinPlugin] Emitted ScatterND(reduction='min') → %s",
            s.get_name(out_v),
        )
