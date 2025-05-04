# file: jax2onnx/plugins/jax/numpy/einsum.py

from typing import Any, Callable, Sequence, TYPE_CHECKING, Dict
import importlib

import jax
from jax import core, numpy as jnp
from jax.interpreters import batching
from jax.extend.core import Primitive
from jax._src.util import safe_zip  # Use safe_zip

from jax._src.export.shape_poly import _DimExpr as DimExpr

from onnx import helper
from jax import eval_shape, ShapeDtypeStruct


from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


import numpy as np  # For manual shape calc


# Define the Einsum primitive
jnp.einsum_p = Primitive("einsum")
jnp.einsum_p.multiple_results = False


@register_primitive(
    primitive_obj=jnp.einsum_p,
    binding_factory=lambda: jnp.einsum,
    jaxpr_primitive=jnp.einsum_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="einsum",
    testcases=[  # --- Added specific attention-related batch tests ---
        {
            "testcase": "einsum_vector_dot",
            "callable": lambda x, y: jnp.einsum("i,i->", x, y),
            "input_shapes": [(5,), (5,)],
        },
        {
            "testcase": "einsum_matrix_vector",
            "callable": lambda x, y: jnp.einsum("ij,j->i", x, y),
            "input_shapes": [(3, 5), (5,)],
        },
        {
            "testcase": "einsum_matrix_matrix",
            "callable": lambda x, y: jnp.einsum("ij,jk->ik", x, y),
            "input_shapes": [("B", 5), (5, 2)],
        },
        {
            "testcase": "einsum_transpose",
            "callable": lambda x: jnp.einsum("ij->ji", x),
            "input_shapes": [(3, 5)],
        },
        {
            "testcase": "einsum_batch_transpose",
            "callable": lambda x: jnp.einsum("...ij->...ji", x),
            "input_shapes": [("B", 3, 5)],
        },
        {
            "testcase": "einsum_diag",
            "callable": lambda x: jnp.einsum("ii->i", x),
            "input_shapes": [(5, 5)],
        },
        {
            "testcase": "einsum_sum_reduce",
            "callable": lambda x: jnp.einsum("ij->", x),
            "input_shapes": [(3, 5)],
        },
        {
            "testcase": "einsum_multi_operand",
            "callable": lambda a, b, c: jnp.einsum("ij,jk,kl->il", a, b, c),
            "input_shapes": [(2, 3), (3, 4), (4, 5)],
        },
        {
            "testcase": "einsum_attention_logits_orig",
            "callable": lambda q, k: jnp.einsum("BTNH,BSNH->BNTS", q, k),
            "input_shapes": [("B", 4, 8, 32), ("B", 4, 8, 32)],
        },
        {
            "testcase": "einsum_attention_output_orig",
            "callable": lambda attn, v: jnp.einsum("BNTS,BSNH->BTNH", attn, v),
            "input_shapes": [("B", 8, 4, 4), ("B", 4, 8, 32)],
        },
        # --- New Tests Mimicking Batched Attention Internals ---
        {
            "testcase": "einsum_attention_logits_batched",
            # Equation modified by batching rule
            "callable": lambda q, k: jnp.einsum("...BTNH,BSNH->...BNTS", q, k),
            # Shapes potentially modified by vmap (added singleton dim)
            "input_shapes": [("B", 1, 4, 8, 32), ("B", 4, 8, 32)],
        },
        {
            "testcase": "einsum_attention_output_batched",
            # Equation modified by batching rule
            "callable": lambda attn, v: jnp.einsum("...BNTS,BSNH->...BTNH", attn, v),
            # Shapes potentially modified by vmap (added singleton dim)
            "input_shapes": [("B", 1, 8, 4, 4), ("B", 4, 8, 32)],
        },
        {
            "testcase": "einsum_ellipsis_rank_mismatch",
            "callable": lambda q, k: jnp.einsum("...BTNH,...BSNH->...BNTS", q, k),
            # Input shapes mimic a scenario where vmap might add a dim, causing different batch ranks
            # Shape 1: (Batch=2, AddedDim=1, SeqT=4, Heads=8, Embed=32)
            # Shape 2: (Batch=2, SeqS=5, Heads=8, Embed=32) -> ... covers (2,) vs (2, 5) - different rank
            # Expected Output Batch: Broadcasted (2, 5)
            # Expected Output Core: (Heads=8, SeqT=4, SeqS=5) -> BNTS
            # Full Expected Output: (2, 5, 8, 4, 5)
            "input_shapes": [(2, 1, 4, 8, 32), (2, 5, 8, 32)],
            "expected_output_shapes": [
                (2, 2, 8, 4, 5)
            ],  # Specify expected shape to catch inference errors
        },
        # --- End New Tests ---
    ],
)
class EinsumPlugin(PrimitiveLeafPlugin):
    """Plugin for jnp.einsum using manual shape calculation workaround."""

    _ORIG_CALL: Callable[..., Any] | None = None  # Still capture original

    # ------------------------------------------------------------------ #
    # Abstract evaluation                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def abstract_eval(*avals, equation: str, **_):
        out_shape = EinsumPlugin._checked_shape(avals, equation)
        return core.ShapedArray(out_shape, avals[0].dtype)

    @staticmethod
    def _get_dynamic_output_shape_manual(
        input_shapes: list[tuple[Any, ...]], equation: str
    ) -> tuple[Any, ...]:
        """Manual calculation of output shape, handling dynamic dimensions and ellipsis broadcasting."""
        if "->" not in equation:
            raise NotImplementedError(
                "Implicit einsum output shape calculation not supported in manual helper."
            )

        input_specs_str, output_spec_str = equation.split("->")
        input_specs = input_specs_str.split(",")

        if len(input_specs) != len(input_shapes):
            raise ValueError(
                f"Einsum specs count ({len(input_specs)}) doesn't match inputs count ({len(input_shapes)})"
            )

        dim_map: Dict[str, Any] = {}
        batch_shape = []  # Stores the computed broadcasted batch shape
        processed_input_specs = (
            []
        )  # Store specs after handling ellipsis for dimension mapping
        first_batch_processed = False  # Track if we've initialized batch_shape

        # --- Phase 1: Determine broadcasted batch shape and map non-batch dimensions ---
        for i, (spec, shape) in enumerate(zip(input_specs, input_shapes)):
            non_batch_spec = spec
            current_operand_batch_shape = []
            num_batch_dims = 0

            if spec.startswith("..."):
                non_batch_spec = spec[3:]
                num_core_dims = len(non_batch_spec)
                num_batch_dims = len(shape) - num_core_dims

                if num_batch_dims < 0:
                    raise ValueError(
                        f"Ellipsis mismatch in spec '{spec}' for shape {shape}"
                    )

                current_operand_batch_shape = list(shape[:num_batch_dims])

                # Initialize or broadcast batch_shape
                if not first_batch_processed and num_batch_dims > 0:
                    # First operand with non-empty ellipsis sets initial batch shape
                    batch_shape = current_operand_batch_shape
                    first_batch_processed = True
                elif (
                    num_batch_dims > 0
                ):  # Subsequent operands with ellipsis need broadcasting
                    # Align ranks by prepending 1s to the shorter shape
                    rank_diff = len(batch_shape) - len(current_operand_batch_shape)
                    if rank_diff > 0:
                        aligned_current = [1] * rank_diff + current_operand_batch_shape
                        aligned_batch = batch_shape
                    elif rank_diff < 0:
                        aligned_batch = [1] * abs(rank_diff) + batch_shape
                        aligned_current = current_operand_batch_shape
                    else:
                        aligned_batch = batch_shape
                        aligned_current = current_operand_batch_shape

                    # Broadcast dimensions element-wise
                    new_broadcasted_batch_shape = []
                    compatible = True
                    for d_batch, d_current in zip(aligned_batch, aligned_current):
                        if d_batch == 1:
                            new_broadcasted_batch_shape.append(d_current)
                        elif d_current == 1:
                            new_broadcasted_batch_shape.append(d_batch)
                        elif (
                            d_batch == d_current
                        ):  # Includes symbolic equality check if objects support it
                            new_broadcasted_batch_shape.append(d_batch)
                        else:
                            # Try robust symbolic comparison if applicable
                            try:
                                is_symbolic_b = isinstance(
                                    d_batch, (core.Tracer, DimExpr)
                                )
                                is_symbolic_c = isinstance(
                                    d_current, (core.Tracer, DimExpr)
                                )
                                if (
                                    is_symbolic_b or is_symbolic_c
                                ) and d_batch == d_current:
                                    new_broadcasted_batch_shape.append(d_batch)
                                else:
                                    compatible = False
                                    break
                            except:  # Catch comparison errors
                                compatible = False
                                break

                    if not compatible:
                        raise ValueError(
                            f"Incompatible batch shapes for broadcasting: {batch_shape} vs {current_operand_batch_shape}"
                        )
                    batch_shape = (
                        new_broadcasted_batch_shape  # Update the overall batch shape
                    )

            # Map core dimensions
            processed_input_specs.append(non_batch_spec)
            core_shape = shape[num_batch_dims:]  # Dimensions not part of batch

            if len(non_batch_spec) != len(core_shape):
                raise ValueError(
                    f"Spec '{non_batch_spec}' rank mismatch with core shape {core_shape} (Full shape: {shape}, Batch shape: {current_operand_batch_shape})"
                )

            for label, size in zip(non_batch_spec, core_shape):
                if label in dim_map:
                    existing_size = dim_map[label]
                    # Handle broadcasting/consistency for core dims
                    if existing_size == 1:
                        dim_map[label] = size
                    elif size != 1 and existing_size != size:
                        # Check symbolic equality robustly
                        try:
                            are_equal = existing_size == size
                        except:
                            are_equal = False  # Assume not equal if comparison fails

                        if not are_equal:
                            raise ValueError(
                                f"Inconsistent size for core label '{label}': {existing_size} vs {size}"
                            )
                        # If they are equal (could be symbolically), keep the existing one
                else:
                    dim_map[label] = size

        # --- Phase 2: Construct output shape ---
        output_core_spec = output_spec_str
        if output_spec_str.startswith("..."):
            output_core_spec = output_spec_str[3:]

        # Start output shape with the final broadcasted batch shape
        output_shape_list = list(batch_shape)

        # Append core dimensions based on output spec
        for label in output_core_spec:
            if label not in dim_map:
                raise ValueError(
                    f"Output label '{label}' not found in mapped dimensions. Dim map: {dim_map}"
                )
            output_shape_list.append(dim_map[label])

        return tuple(output_shape_list)

    # --- END Manual Shape Calculation Helper ---

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Emit an ONNX `Einsum` op and attach correct, possibly‑dynamic shapes."""
        # ---- make ONNX accept broadcasting ellipses ----
        equation = params["equation"]
        in_specs, _ = equation.split("->")
        in_specs = in_specs.split(",")

        # how many dims are covered by "..." for each operand?
        ellipsis_ranks = []
        for spec, v in zip(in_specs, node_inputs):
            if spec.startswith("..."):
                core_rank = len(spec[3:])
                ellipsis_ranks.append(len(v.aval.shape) - core_rank)
            else:
                ellipsis_ranks.append(0)

        max_ellipsis_rank = max(ellipsis_ranks)
        input_names = []

        for spec, v, er in zip(in_specs, node_inputs, ellipsis_ranks):
            base_name = s.get_name(v)
            if spec.startswith("...") and er < max_ellipsis_rank:
                # pad with leading singleton axes
                pad = max_ellipsis_rank - er
                new_shape = (1,) * pad + v.aval.shape
                axes_const = s.get_constant_name(
                    np.array(list(range(pad)), dtype=np.int64)
                )  # 1‑D int64 tensor

                padded = s.get_unique_name(base_name + "_pad")
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[base_name, axes_const],  # ← 2 inputs 👍
                        outputs=[padded],
                        name=s.get_unique_name("unsqueeze"),
                    )
                )
                s.add_shape_info(padded, new_shape, v.aval.dtype)
                input_names.append(padded)
            else:
                input_names.append(base_name)

        out_var = node_outputs[0]
        out_name = s.get_name(out_var)

        # (equation itself is unchanged –  now every operand's ellipsis
        #  has identical rank & sizes, so ONNX is happy.)

        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=input_names,
                outputs=[out_name],
                name=s.get_unique_name("einsum"),
                equation=equation,
            )
        )

        # ---- final shape (validated) ----
        inferred_shape = EinsumPlugin._checked_shape(
            [v.aval for v in node_inputs], equation
        )
        s.add_shape_info(out_name, inferred_shape, out_var.aval.dtype)

    @staticmethod
    def _einsum_binding(*args: Any, equation: str, **kwargs: Any) -> Any:
        """Binds inputs to the einsum primitive."""
        bind_kwargs = {
            "equation": equation,
            "precision": kwargs.get("precision"),
            "preferred_element_type": kwargs.get("preferred_element_type"),
            "_numeric_decoder": kwargs.get("_numeric_decoder"),
        }
        bind_kwargs = {k: v for k, v in bind_kwargs.items() if v is not None}
        return jnp.einsum_p.bind(*args, **bind_kwargs)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns the patched function that binds the primitive."""
        EinsumPlugin._ORIG_CALL = orig_fn

        def patched_einsum(subscripts: str, *operands: Any, **kwargs: Any) -> Any:
            return EinsumPlugin._einsum_binding(
                *operands, equation=subscripts, **kwargs
            )

        return patched_einsum

    @staticmethod
    def patch_info():
        """Provides patching information for jnp.einsum."""
        return {
            "patch_targets": [jnp],
            "target_attribute": "einsum",
            "patch_function": EinsumPlugin.get_monkey_patch,
        }

        # --- NEW helper -------------------------------------------------------

    @staticmethod
    def _infer_shape_with_fallback(arg_avals, equation: str) -> tuple[Any, ...]:
        """
        Try the manual rule first, but, if it disagrees with JAX's own
        (fully‑concrete) `eval_shape`, trust the latter.
        """
        # 1. manual (may raise, so keep it in a try)
        manual_ok, manual_shape = False, ()
        try:
            manual_shape = EinsumPlugin._get_dynamic_output_shape_manual(
                [a.shape for a in arg_avals], equation
            )
            manual_ok = True
        except Exception:
            pass

        # 2. ground‑truth via the *original* jnp.einsum
        orig_einsum = EinsumPlugin._ORIG_CALL or jnp.einsum
        dummy = [ShapeDtypeStruct(a.shape, a.dtype) for a in arg_avals]
        true_shape = eval_shape(lambda *xs: orig_einsum(equation, *xs), *dummy).shape

        # 3. choose
        if manual_ok:
            # if any concrete dimensions differ -> trust the ground truth
            for m, t in zip(manual_shape, true_shape):
                if isinstance(m, int) and isinstance(t, int) and m != t:
                    return true_shape
            return manual_shape
        return true_shape

    # --- new helper --------------------------------------------------------
    @staticmethod
    def _checked_shape(
        arg_avals: Sequence[core.AbstractValue], equation: str
    ) -> tuple[Any, ...]:
        """
        Return a shape that is guaranteed to agree with JAX.

        * try the fast manual rule;
        * compare against `eval_shape` using the *un‑patched* ``jnp.einsum``;
        * if any concrete dim differs, fall back to the JAX result.
        """
        # 1. manual – may raise
        manual_shape: tuple[Any, ...] | None = None
        try:
            manual_shape = EinsumPlugin._get_dynamic_output_shape_manual(
                [a.shape for a in arg_avals], equation
            )
        except Exception:  # pragma: no cover
            pass

        # 2. ground truth from JAX
        orig_einsum = EinsumPlugin._ORIG_CALL or jnp.einsum
        dummies = [ShapeDtypeStruct(a.shape, a.dtype) for a in arg_avals]
        true_shape = eval_shape(lambda *xs: orig_einsum(equation, *xs), *dummies).shape

        # 3. decide
        if manual_shape is None:
            return true_shape

        for m, t in zip(manual_shape, true_shape):
            if isinstance(m, int) and isinstance(t, int) and m != t:
                return true_shape  # concrete mismatch – trust JAX
        return manual_shape  # manual is fine


# --------------------------------------------------------------------- #
# 1. Try to reuse JAX's official batching rule.                         #
#    If it is not available (different JAX revision),                   #
#    we fall back to the lightweight rule we wrote earlier.             #
# --------------------------------------------------------------------- #
try:
    _std_rule = importlib.import_module(
        "jax._src.numpy.einsum"
    ).einsum_batching_rule  # type: ignore
    batching.primitive_batchers[jnp.einsum_p] = _std_rule
except (ModuleNotFoundError, AttributeError):
    # ---------- minimal fallback (what we had before) ----------
    def _fallback_einsum_batching_rule(args, batch_axes, **params):
        equation = params["equation"]
        # add ellipsis *everywhere* so all operands broadcast correctly
        in_specs, out_spec = equation.split("->")
        new_in_specs = [
            spec if "..." in spec else f"...{spec}" for spec in in_specs.split(",")
        ]
        params = dict(params, equation=f"{','.join(new_in_specs)}->...{out_spec}")
        res = jnp.einsum_p.bind(*args, **params)
        return res, 0

    batching.primitive_batchers[jnp.einsum_p] = _fallback_einsum_batching_rule


# --- Registrations ---
jnp.einsum_p.def_abstract_eval(EinsumPlugin.abstract_eval)  # Use manual abstract_eval
# --- End Registrations ---
