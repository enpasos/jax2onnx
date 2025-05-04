# file: jax2onnx/plugins/jax/numpy/einsum.py

from typing import Any, Callable, Sequence, TYPE_CHECKING, Dict

import jax
from jax import core, numpy as jnp
from jax.interpreters import batching
from jax.extend.core import Primitive
from jax._src.util import safe_zip  # Use safe_zip

# Assuming DimExpr might be part of shapes handled
from jax._src.export.shape_poly import _DimExpr as DimExpr

from onnx import helper
from jax import eval_shape, ShapeDtypeStruct


from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

    # from jax._src.export.shape_poly import _DimExpr as DimExpr # Already imported

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
        # --- End New Tests ---
    ],
)
class EinsumPlugin(PrimitiveLeafPlugin):
    """Plugin for jnp.einsum using manual shape calculation workaround."""

    _ORIG_CALL: Callable[..., Any] | None = None  # Still capture original

    # INSIDE EinsumPlugin.abstract_eval  – replace the whole manual section

    @staticmethod
    def abstract_eval(*args_avals, equation: str, **kwargs):
        """Ask JAX itself what the output aval is."""
        # Use the original (pre‑patch) jnp.einsum so we don’t recurse.
        orig_einsum = EinsumPlugin._ORIG_CALL or jnp.einsum

        dummy_args = [ShapeDtypeStruct(a.shape, a.dtype) for a in args_avals]
        out_aval = eval_shape(lambda *xs: orig_einsum(equation, *xs), *dummy_args)
        return core.ShapedArray(out_aval.shape, out_aval.dtype)

    @staticmethod
    def _get_dynamic_output_shape_manual(
        input_shapes: list[tuple[Any, ...]], equation: str
    ) -> tuple[Any, ...]:
        """Manual calculation of output shape, handling dynamic dimensions and ellipsis."""
        # (Keep implementation from previous response - manual calculation with ellipsis fix)
        if "->" not in equation:
            return ()  # Basic handling for implicit output
        input_specs_str, output_spec_str = equation.split("->")
        input_specs = input_specs_str.split(",")
        if len(input_specs) != len(input_shapes):
            raise ValueError(f"Einsum specs/inputs mismatch")
        dim_map: Dict[str, Any] = {}
        batch_shape = []
        processed_specs = []  # Store specs after handling ellipsis

        for i, (spec, shape) in enumerate(zip(input_specs, input_shapes)):
            non_batch_spec = spec
            num_batch_dims = 0
            if spec.startswith("..."):
                non_batch_spec = spec[3:]
                num_batch_dims = len(shape) - len(non_batch_spec)
                if num_batch_dims < 0:
                    raise ValueError(f"Ellipsis mismatch: spec '{spec}' shape {shape}")
                current_batch_shape = list(shape[:num_batch_dims])
                if not batch_shape:
                    batch_shape = current_batch_shape
                elif batch_shape != current_batch_shape:
                    if len(batch_shape) != len(current_batch_shape):
                        raise ValueError("Inconsistent batch ranks")
                    new_b = []
                    for d1, d2 in zip(batch_shape, current_batch_shape):
                        if d1 == 1:
                            new_b.append(d2)
                        elif d2 == 1:
                            new_b.append(d1)
                        elif d1 == d2:
                            new_b.append(d1)
                        else:
                            raise ValueError(
                                f"Inconsistent batch shapes: {batch_shape} vs {current_batch_shape}"
                            )
                    batch_shape = new_b
            processed_specs.append(non_batch_spec)
            shape_to_process = shape[num_batch_dims:]
            if len(non_batch_spec) != len(shape_to_process):
                raise ValueError(
                    f"Spec/Shape rank mismatch after ellipsis: '{non_batch_spec}' vs {shape_to_process}"
                )
            for label, size in zip(non_batch_spec, shape_to_process):
                if label in dim_map:
                    existing_size = dim_map[label]
                    # Allow matching concrete, matching symbolic, or concrete vs symbolic
                    is_existing_symbolic = isinstance(
                        existing_size, (core.Tracer, DimExpr)
                    )
                    is_current_symbolic = isinstance(size, (core.Tracer, DimExpr))
                    if existing_size == 1 and not is_current_symbolic:
                        dim_map[label] = size
                    elif size == 1 and not is_existing_symbolic:
                        pass  # keep existing larger or symbolic size
                    elif existing_size != size and not (
                        is_existing_symbolic or is_current_symbolic
                    ):
                        raise ValueError(
                            f"Inconsistent size for label '{label}': {existing_size} vs {size}"
                        )
                    elif is_current_symbolic and not is_existing_symbolic:
                        dim_map[label] = size  # Prefer symbolic
                else:
                    dim_map[label] = size

        output_shape_list = list(batch_shape)
        if output_spec_str.startswith("..."):
            output_spec_str = output_spec_str[3:]
        for label in output_spec_str:
            if label not in dim_map:
                raise ValueError(f"Output label '{label}' not found.")
            output_shape_list.append(dim_map[label])
        return tuple(output_shape_list)

    # --- END Manual Shape Calculation Helper ---

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of the einsum primitive to ONNX Einsum op."""
        input_names = [s.get_name(var) for var in node_inputs]
        output_var = node_outputs[0]
        output_name = s.get_name(output_var)
        output_aval = output_var.aval
        equation = params["equation"]
        einsum_node = helper.make_node(
            "Einsum",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("einsum"),
            equation=equation,
        )
        s.add_node(einsum_node)
        s.add_shape_info(output_name, output_aval.shape, output_aval.dtype)

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


# --- Batching Rule (Keep fixed version) ---
def einsum_batching_rule(args, batch_axes, **params):
    """Batching rule for einsum, handles ellipsis."""
    equation = params["equation"]
    batch_axes_filtered = [ax for ax in batch_axes if ax is not None]
    if not batch_axes_filtered:
        return jnp.einsum_p.bind(*args, **params), None
    if len(set(batch_axes_filtered)) > 1:
        raise NotImplementedError("Einsum batching rule requires same batch axis.")
    if "..." not in equation:
        input_specs, output_spec = equation.split("->")
        batched_input_specs = [
            f"...{spec}" if batch_axes[i] is not None else spec
            for i, spec in enumerate(input_specs.split(","))
        ]
        batched_equation = f"{','.join(batched_input_specs)}->...{output_spec}"
    else:
        batched_equation = equation
    new_params = params.copy()
    new_params["equation"] = batched_equation
    result = jnp.einsum_p.bind(*args, **new_params)  # Direct bind call
    return result, 0


# --- Registrations ---
jnp.einsum_p.def_abstract_eval(EinsumPlugin.abstract_eval)  # Use manual abstract_eval
batching.primitive_batchers[jnp.einsum_p] = einsum_batching_rule
# --- End Registrations ---

# --- Debugging: Print Registered Batching Rules ---
print("Registered Batching Rules:")
for primitive, batcher in batching.primitive_batchers.items():
    print(f"  {primitive}: {batcher}")
