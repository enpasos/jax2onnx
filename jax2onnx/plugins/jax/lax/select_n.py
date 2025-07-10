# file: jax2onnx/plugins/jax/lax/select_n.py

from typing import TYPE_CHECKING, List, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from onnx import TensorProto, helper
from jax.extend.core import Var

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.select_n_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html",
    onnx=[  # This ONNX mapping is accurate for the 2-case boolean predicate scenario
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="select_n",
    testcases=[
        {
            "testcase": "select_n_bool_predicate_two_cases_float",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array([True, False, True], dtype=jnp.bool_),
                jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),  # on_false (case 0)
                jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),  # on_true (case 1)
            ],
            "description": "Test lax.select_n with a boolean predicate and two float choices.",
        },
        {
            "testcase": "select_n_bool_predicate_two_cases_int",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array([[True, False], [False, True]], dtype=jnp.bool_),
                jnp.array([[10, 20], [30, 40]], dtype=jnp.int32),  # on_false (case 0)
                jnp.array([[50, 60], [70, 80]], dtype=jnp.int32),  # on_true (case 1)
            ],
            "description": "Test lax.select_n with a boolean predicate and two integer choices.",
        },
        {
            "testcase": "select_n_bool_predicate_scalar_broadcast",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array(True, dtype=jnp.bool_),  # Scalar predicate
                jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),  # on_false (case 0)
                jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),  # on_true (case 1)
            ],
            "description": "Test lax.select_n with a scalar boolean predicate broadcasting over array choices.",
        },
        # --- Test cases for integer indices and multiple choices ---
        # These are expected to FAIL or produce incorrect results with the current plugin
        # as it's tailored for the 2-case boolean scenario.
        {
            "testcase": "select_n_int_indices_three_cases",
            "callable": lambda indices, c0, c1, c2: jax.lax.select_n(
                indices, c0, c1, c2
            ),
            "input_values": [
                jnp.array([0, 1, 2, 0], dtype=jnp.int32),  # indices
                jnp.array([10, 11, 12, 13], dtype=jnp.float32),  # case 0
                jnp.array([20, 21, 22, 23], dtype=jnp.float32),  # case 1
                jnp.array([30, 31, 32, 33], dtype=jnp.float32),  # case 2
            ],
            "description": "Test lax.select_n with integer indices and three choices. Current plugin may not support this correctly.",
        },
        {
            "testcase": "select_n_int_indices_four_cases",
            "callable": lambda indices, c0, c1, c2, c3: jax.lax.select_n(
                indices, c0, c1, c2, c3
            ),
            "input_values": [
                jnp.array([0, 1, 2, 3, 1, 0], dtype=jnp.int32),  # indices
                jnp.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=jnp.float32),  # case 0
                jnp.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5], dtype=jnp.float32),  # case 1
                jnp.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5], dtype=jnp.float32),  # case 2
                jnp.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5], dtype=jnp.float32),  # case 3
            ],
            "description": "Test lax.select_n with integer indices and four choices. Current plugin may not support this correctly.",
        },
    ],
)
class SelectNPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.select_n to ONNX.
    Handles both the 2-case boolean predicate scenario (using ONNX Where)
    and the general N-way select with integer indices (using ONNX GatherND).
    """

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: List["Var"],
        node_outputs: List["Var"],
        params: Dict[str, Any],
    ):
        condition_or_indices_var = node_inputs[0]
        cases_vars = node_inputs[1:]
        output_var = node_outputs[0]

        condition_or_indices_name = s.get_name(condition_or_indices_var)
        output_name = s.get_name(output_var)
        output_aval = output_var.aval

        # Check if it's the 2-case boolean predicate
        is_boolean_predicate = condition_or_indices_var.aval.dtype == jnp.bool_

        if is_boolean_predicate and len(cases_vars) == 2:
            # Handles select_n(pred, on_false, on_true) -> Where(pred, on_true, on_false)
            s.logger.info(
                f"select_n: Handling 2-case boolean predicate for output '{output_name}'."
            )
            on_false_var = cases_vars[0]
            on_true_var = cases_vars[1]

            # For ONNX Where(condition, X, Y): X is if true, Y is if false.
            x_name_onnx = s.get_name(on_true_var)
            y_name_onnx = s.get_name(on_false_var)

            node = helper.make_node(
                "Where",
                inputs=[condition_or_indices_name, x_name_onnx, y_name_onnx],
                outputs=[output_name],
                name=s.get_unique_name(f"where_from_select_n_{output_name}"),
            )
            s.add_node(node)
            s.add_shape_info(output_name, output_aval.shape, output_aval.dtype)

        elif not is_boolean_predicate and jnp.issubdtype(
            condition_or_indices_var.aval.dtype, jnp.integer
        ):
            s.logger.info(
                f"select_n: Handling N-case integer indices for output '{output_name}'."
            )

            if not cases_vars:
                raise ValueError(
                    "select_n with integer indices requires at least one case."
                )

            # 1. Stack all cases
            # Each case tensor (e.g., shape (6,)) needs to be unsqueezed to (1, 6)
            # Then concatenated along axis=0 to form (num_cases, 6)

            unsqueezed_case_names = []
            case_shape = cases_vars[
                0
            ].aval.shape  # Assuming all cases have the same shape
            axes_for_unsqueeze_name = s.get_constant_name(np.array([0], dtype=np.int64))

            for i, case_var in enumerate(cases_vars):
                case_name = s.get_name(case_var)
                unsqueezed_name = s.get_unique_name(
                    f"unsqueezed_case_{i}_{output_name}"
                )

                # Ensure consistent dtypes for stacking if necessary (ONNX Concat requires same dtype)
                # For now, assume JAX ensures cases have compatible dtypes for select_n result.

                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[case_name, axes_for_unsqueeze_name],
                        outputs=[unsqueezed_name],
                        name=s.get_unique_name(f"unsqueeze_op_case_{i}_{output_name}"),
                    )
                )
                s.add_shape_info(
                    unsqueezed_name, (1,) + case_shape, case_var.aval.dtype
                )
                unsqueezed_case_names.append(unsqueezed_name)

            stacked_cases_name = s.get_unique_name(f"stacked_cases_{output_name}")
            s.add_node(
                helper.make_node(
                    "Concat",
                    inputs=unsqueezed_case_names,
                    outputs=[stacked_cases_name],
                    axis=0,  # Stack along the new leading dimension
                    name=s.get_unique_name(f"concat_op_cases_{output_name}"),
                )
            )
            num_cases = len(cases_vars)
            stacked_cases_shape = (num_cases,) + case_shape
            s.add_shape_info(
                stacked_cases_name, stacked_cases_shape, cases_vars[0].aval.dtype
            )

            # 2. Prepare JAX indices for GatherND (to select full case slices)
            # JAX indices shape: e.g., (6,)
            # ONNX GatherND indices for desired output: (6, 1) if we want to pick rows from (num_cases, 6)
            #   Each element [idx] in the JAX indices will correspond to [idx] in ONNX indices.

            indices_name_int64 = condition_or_indices_name
            if condition_or_indices_var.aval.dtype != np.int64:
                indices_name_int64 = s.get_unique_name(
                    f"{condition_or_indices_name}_int64"
                )
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[condition_or_indices_name],
                        outputs=[indices_name_int64],
                        to=TensorProto.INT64,
                    )
                )
                s.add_shape_info(
                    indices_name_int64, condition_or_indices_var.aval.shape, np.int64
                )

            # For GatherND, indices need to select elements from stacked_cases.
            # If stacked_cases is (num_cases, D1, D2, ...) and JAX indices is (N1, N2, ...),
            # output[i1,i2,...,j1,j2,...] = stacked_cases[ JAX_indices[i1,i2,...], j1, j2, ... ]
            # We need to form an ONNX indices tensor of shape (N1, N2, ..., 1)
            # where the last dimension contains the value from JAX_indices.

            indices_rank = len(condition_or_indices_var.aval.shape)
            axes_for_unsqueeze_indices_name = s.get_constant_name(
                np.array([indices_rank], dtype=np.int64)
            )  # Unsqueeze at the end

            prepared_gather_indices_name = s.get_unique_name(
                f"prepared_gather_indices_{output_name}"
            )
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[indices_name_int64, axes_for_unsqueeze_indices_name],
                    outputs=[prepared_gather_indices_name],
                )
            )
            prepared_indices_shape = condition_or_indices_var.aval.shape + (1,)
            s.add_shape_info(
                prepared_gather_indices_name, prepared_indices_shape, np.int64
            )

            # 3. GatherND Operation (Intermediate step)
            # This selects entire cases based on JAX indices, resulting in shape (S1,..,Sk, D1,..,Dm)
            # e.g., if JAX_indices is (6,) and cases are (6,), this gives (6,6)
            # output[k,j] = (case_selected_by_JAX_indices[k])[j]
            intermediate_gathered_name = s.get_unique_name(
                f"intermediate_gathered_select_n_{output_name}"
            )
            s.add_node(
                helper.make_node(
                    "GatherND",
                    inputs=[stacked_cases_name, prepared_gather_indices_name],
                    outputs=[intermediate_gathered_name],
                    batch_dims=0,
                    name=s.get_unique_name(
                        f"gathernd_intermediate_for_select_n_{output_name}"
                    ),
                )
            )
            # Shape of intermediate_gathered_name is JAX_indices.shape + case_shape
            # For test: (6,) + (6,) -> (6,6)
            intermediate_shape = (
                condition_or_indices_var.aval.shape + cases_vars[0].aval.shape
            )
            s.add_shape_info(
                intermediate_gathered_name, intermediate_shape, output_aval.dtype
            )

            # 4. Perform diagonal selection for element-wise pick
            # JAX behavior for select_n(indices, c0, c1...): output[k] = (case_selected_by_indices[k])[k]
            # If JAX indices shape is (S1, ..., Sn) and case element shape is (D1, ..., Dm)
            # The JAX output shape is (S1, ..., Sn).
            # The intermediate_gathered_name has shape (S1, ..., Sn, D1, ..., Dm).
            # We need to select elements such that the indices into (D1,...,Dm) match (S1,...,Sn) if they are compatible.

            # In the current test case:
            # JAX indices shape is (6,) -> S1=6
            # Case shape is (6,) -> D1=6
            # JAX output shape is (6,)
            # intermediate_gathered_name is (6, 6)
            # We need output[i] = intermediate_gathered_name[i, i]

            # Check if shapes allow for this diagonal extraction.
            # This diagonal extraction is valid if JAX_indices_shape == case_shape_prefix
            # and the desired output is JAX_indices_shape.
            # For now, assume the test case (indices (6,), cases (6,), output (6,)) is the target.

            if (
                len(condition_or_indices_var.aval.shape) == 1
                and len(cases_vars[0].aval.shape) == 1
                and condition_or_indices_var.aval.shape[0]
                == cases_vars[0].aval.shape[0]
                and output_aval.shape == condition_or_indices_var.aval.shape
            ):

                s.logger.debug(
                    f"select_n: Applying diagonal extraction for element-wise selection on '{output_name}'."
                )

                # Create indices for diagonal: [0, 1, ..., N-1] where N is shape[0] of JAX indices
                num_elements = condition_or_indices_var.aval.shape[0]

                start_val_name = s.get_constant_name(np.array(0, dtype=np.int64))
                limit_val_name = s.get_constant_name(
                    np.array(num_elements, dtype=np.int64)
                )
                delta_val_name = s.get_constant_name(np.array(1, dtype=np.int64))

                diag_indices_flat_name = s.get_unique_name(
                    f"diag_indices_flat_{output_name}"
                )
                s.add_node(
                    helper.make_node(
                        "Range",
                        inputs=[start_val_name, limit_val_name, delta_val_name],
                        outputs=[diag_indices_flat_name],
                    )
                )
                s.add_shape_info(diag_indices_flat_name, (num_elements,), np.int64)

                # Unsqueeze to make it (N, 1) for GatherElements axis=1
                axes_for_unsqueeze_diag_name = s.get_constant_name(
                    np.array([1], dtype=np.int64)
                )
                diag_indices_unsqueezed_name = s.get_unique_name(
                    f"diag_indices_unsqueezed_{output_name}"
                )
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[diag_indices_flat_name, axes_for_unsqueeze_diag_name],
                        outputs=[diag_indices_unsqueezed_name],
                    )
                )
                s.add_shape_info(
                    diag_indices_unsqueezed_name, (num_elements, 1), np.int64
                )

                # GatherElements to pick the diagonal
                # output[i,0] = intermediate_gathered_name[i, diag_indices_unsqueezed_name[i,0]]
                # which is intermediate_gathered_name[i,i]
                gathered_elements_name = s.get_unique_name(
                    f"gathered_elements_{output_name}"
                )
                s.add_node(
                    helper.make_node(
                        "GatherElements",
                        inputs=[
                            intermediate_gathered_name,
                            diag_indices_unsqueezed_name,
                        ],
                        outputs=[gathered_elements_name],
                        axis=1,  # Select along the second dimension of the (N,N) intermediate tensor
                    )
                )
                s.add_shape_info(
                    gathered_elements_name, (num_elements, 1), output_aval.dtype
                )

                # Squeeze to final (N,) shape
                axes_for_squeeze_final_name = s.get_constant_name(
                    np.array([1], dtype=np.int64)
                )
                s.add_node(
                    helper.make_node(
                        "Squeeze",
                        inputs=[gathered_elements_name, axes_for_squeeze_final_name],
                        outputs=[output_name],
                    )
                )
                s.add_shape_info(output_name, output_aval.shape, output_aval.dtype)
            else:
                # If not the specific diagonal case, assume the GatherND output is what's expected
                # This might happen if JAX output is indeed (S1..Sk, D1..Dm)
                # This path needs more careful thought for general cases.
                # For now, if the JAX output shape matches the intermediate_gathered_name shape,
                # then no further processing is needed.
                if output_aval.shape == intermediate_shape:
                    s.logger.warning(
                        f"select_n: Output shape {output_aval.shape} matches GatherND output. "
                        f"Skipping diagonal extraction for {output_name}. Ensure this is intended."
                    )
                    s.add_node(
                        helper.make_node(  # Identity op
                            "Identity",
                            inputs=[intermediate_gathered_name],
                            outputs=[output_name],
                        )
                    )
                    s.add_shape_info(output_name, output_aval.shape, output_aval.dtype)
                else:
                    raise NotImplementedError(
                        f"select_n: General N-case integer index selection for JAX output shape {output_aval.shape} "
                        f"from JAX indices {condition_or_indices_var.aval.shape} and case shape {cases_vars[0].aval.shape} "
                        f"(intermediate ONNX shape {intermediate_shape}) is not fully implemented beyond simple diagonal. "
                        f"ONNX Runtime reported ONNX shape {intermediate_shape}, JAX expected {output_aval.shape}."
                    )
        else:
            # This case should ideally not be hit if the JAX types are consistent
            # or if the two main cases (bool predicate or int indices) cover all valid uses.
            raise NotImplementedError(
                f"lax.select_n received an unexpected combination of inputs: "
                f"first input dtype {condition_or_indices_var.aval.dtype}, number of cases {len(cases_vars)+1}. "
                f"A general handler for this specific configuration might be needed."
            )
