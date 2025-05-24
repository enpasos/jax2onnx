# jax2onnx/plugins/jax/lax/scatter.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any

import numpy as np
import jax.numpy as jnp  # Keep for potential use in test cases or future needs
from jax import ShapeDtypeStruct, lax, core
from jax.lax import (
    ScatterDimensionNumbers,
    GatherScatterMode,  # Keep for parameters, though ScatterND has limited mode support
)
from onnx import (
    helper,
)  # TensorProto might be needed by the utility if not passed via s.

# Import the new utility function
from .scatter_utils import (
    _prepare_scatter_inputs_for_onnx,
)

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


import logging

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter")


scatter_p = lax.scatter_p


@register_primitive(
    jaxpr_primitive=scatter_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html",
    onnx=[
        # The primary target is now ScatterND
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        # ScatterElements is no longer the direct fallback from this plugin's core logic
    ],
    since="v0.4.4",  # Consider updating if behavior significantly changes/improves
    context="primitives.lax",
    component="scatter",
    testcases=[  # Existing testcases are kept as per instruction
        {
            "testcase": "scatter_set_axis0",
            "callable": lambda x: x.at[0].set(-100.0),
            "input_shapes": [(1, 1)],
        },
        {
            "testcase": "scatter_set_middle",
            "callable": lambda x: x.at[1].set(42.0),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "scatter_correct_axis_determination",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_updates_slice_needed_axis0",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_from_user_warning_shapes_valid_jax",
            "callable": lambda operand, indices, updates_sliced_scalar_batch: lax.scatter(
                operand,
                indices,
                jnp.reshape(updates_sliced_scalar_batch, indices.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_user_error_scenario_precise",  # This test will be critical
            "callable": lambda operand, indices, updates: lax.scatter(
                operand,
                indices,
                updates,
                ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
                mode=GatherScatterMode.FILL_OR_DROP,
                unique_indices=False,
                indices_are_sorted=False,
            ),
            "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
    ],
)
class ScatterPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,  # JAX's scatter_p has this
        *,
        dimension_numbers: ScatterDimensionNumbers,  # type: ignore
        indices_are_sorted: bool,  # type: ignore
        unique_indices: bool,  # type: ignore
        mode: GatherScatterMode | str | None,  # type: ignore
        **params,
    ):
        # Output shape and dtype match the operand
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
        out_name = s.get_name(out_v)

        # Original operand shape and dtype for final output registration
        operand_aval = operand_v.aval
        operand_shape = tuple(operand_aval.shape)
        operand_dtype_np = np.dtype(operand_aval.dtype)

        dimension_numbers: ScatterDimensionNumbers = params["dimension_numbers"]
        # The `mode` parameter from JAX (e.g., FILL_OR_DROP, CLIP) might influence
        # ops *around* ScatterND if ScatterND doesn't directly support it.
        # The utility function prepares data; mode handling beyond reduction is for the plugin.
        # lax.scatter implies reduction="none".
        # mode_param = params.get("mode", GatherScatterMode.PROMISE_IN_BOUNDS)
        # mode_enum = mode_param
        # if not isinstance(mode_param, GatherScatterMode):
        #     mode_enum = GatherScatterMode.from_any(str(mode_param))
        # TODO: Handle JAX modes like CLIP or FILL_OR_DROP if ScatterND reduction='none'
        # does not cover them. This might require additional ONNX ops around ScatterND.
        # For now, we assume the utility prepares for standard ScatterND behavior.

        logger.info(
            f"Preparing inputs for ONNX ScatterND for JAX scatter primitive with "
            f"dimension_numbers: {dimension_numbers}"
        )

        # 1. Call the common utility function to prepare inputs for ScatterND
        # The utility function handles casting indices, reshaping indices and updates.
        final_operand_name, final_indices_name, final_updates_name = (
            _prepare_scatter_inputs_for_onnx(
                s,
                operand_v,
                indices_v,
                updates_v,
                dimension_numbers,
            )
        )

        # 2. Create the ONNX ScatterND node
        # For lax.scatter, the reduction mode is 'none'.
        reduction_attribute = "none"

        # Opset version check for reduction='none' (available since opset 16 for ScatterND)
        # ScatterND was introduced in opset 11, but reduction attribute was added later.
        # If opset < 11, ScatterND isn't available.
        # If 11 <= opset < 16, ScatterND is available but 'reduction' attribute is not.
        #    In this case, 'none' is the default behavior, so we might not need to set the attribute.
        # If opset >= 16, 'reduction' attribute is available.

        node_attributes = {}
        # Access opset_version via s.builder.opset
        if s.builder.opset >= 11:  # ScatterND exists
            if s.builder.opset >= 16:  # reduction attribute exists
                node_attributes["reduction"] = reduction_attribute
            elif (
                reduction_attribute != "none"
            ):  # opset 11-15, reduction not supported, but requested non-default
                raise NotImplementedError(
                    f"ScatterND with reduction='{reduction_attribute}' requires ONNX opset 16+. "
                    f"Current opset: {s.builder.opset}"  # Corrected here
                )
            # If opset is 11-15 and reduction_attribute is 'none', it's the default, so no attribute needed.
        else:  # opset < 11
            raise NotImplementedError(
                f"ScatterND requires ONNX opset 11+. Current opset: {s.builder.opset}"  # Corrected here
            )

        s.add_node(
            helper.make_node(
                "ScatterND",
                inputs=[final_operand_name, final_indices_name, final_updates_name],
                outputs=[out_name],
                name=s.get_unique_name(f"scatter_nd_{out_name}"),
                **node_attributes,
            )
        )

        # 3. Register final output shape and dtype robustly
        # operand_shape and operand_dtype_np are for the output, derived from operand_v.aval

        sds_out = ShapeDtypeStruct(operand_shape, operand_dtype_np)
        s.shape_env[out_name] = (
            sds_out  # Explicitly populate s.shape_env with the consistently imported ShapeDtypeStruct
        )
        s.add_shape_info(
            out_name, operand_shape, operand_dtype_np
        )  # Ensure builder's value_info is also updated

        logger.debug(
            f"[{self.__class__.__name__}] Ensured s.shape_env and called add_shape_info for ScatterND output '{out_name}' with {sds_out}"
        )

        # --- Logging Block for Inputs to ScatterND ---
        # Ensure these logging calls robustly check for presence in shape_env and attribute existence
        for role, name_to_check in [
            ("operand", final_operand_name),
            ("indices", final_indices_name),
            ("updates", final_updates_name),
        ]:
            info = s.shape_env.get(name_to_check)
            if info is not None and hasattr(info, "shape") and hasattr(info, "dtype"):
                logger.debug(
                    f"[ScatterPlugin] Input '{role}' ('{name_to_check}') for ScatterND: "
                    f"shape={info.shape}, dtype={info.dtype}"
                )
            else:
                if info is None:
                    logger.warning(
                        f"[ScatterPlugin] Input '{role}' ('{name_to_check}') for ScatterND: "
                        f"Info is None in shape_env."
                    )
                else:
                    logger.warning(
                        f"[ScatterPlugin] Input '{role}' ('{name_to_check}') for ScatterND: "
                        f"Info not a valid ShapeDtypeStruct in shape_env (type: {type(info)})."
                    )
        # --- End Logging Block ---

        # More robust verification using hasattr (duck typing)
        output_info_final_check = s.shape_env.get(out_name)
        if output_info_final_check is None:
            logger.error(
                f"CRITICAL ERROR in {self.__class__.__name__}: Output info for '{out_name}' is None in shape_env AFTER explicit set."
            )
        elif not (
            hasattr(output_info_final_check, "shape")
            and hasattr(output_info_final_check, "dtype")
            and output_info_final_check.shape is not None
            and output_info_final_check.dtype is not None
        ):
            logger.error(
                f"CRITICAL ERROR in {self.__class__.__name__}: Output info for '{out_name}' (type: {type(output_info_final_check)}) "
                f"in shape_env AFTER explicit set is not ShapeDtypeStruct-like (missing .shape or .dtype)."
            )
        else:
            # Only check .shape and .dtype if output_info_final_check is not None and has those attributes
            if not (
                output_info_final_check.shape == operand_shape
                and np.dtype(output_info_final_check.dtype) == operand_dtype_np
            ):
                logger.warning(
                    f"[{self.__class__.__name__}] Final verification mismatch for {out_name}. "
                    f"Env: {output_info_final_check.shape}/{output_info_final_check.dtype}, "
                    f"Expected: {operand_shape}/{operand_dtype_np}. This might be due to symbolic shapes if dynamically resolved."
                )
            else:
                logger.debug(
                    f"[{self.__class__.__name__}] Output info for '{out_name}' verified in shape_env."
                )
