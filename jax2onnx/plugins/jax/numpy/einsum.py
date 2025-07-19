# file: jax2onnx/plugins/jax/numpy/einsum.py

from typing import (
    Any,
    Callable,
    Sequence,
    TYPE_CHECKING,
    Dict,
    Tuple,
)
import importlib
import numpy as np

from jax import core, numpy as jnp
from jax.interpreters import batching
from jax.extend.core import Primitive
from onnx import helper
from jax import eval_shape, ShapeDtypeStruct

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


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
    testcases=[
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
        {
            "testcase": "einsum_attention_logits_batched",
            "callable": lambda q, k: jnp.einsum("...BTNH,BSNH->...BNTS", q, k),
            "input_shapes": [("B", 1, 4, 8, 32), ("B", 4, 8, 32)],
        },
        {
            "testcase": "einsum_attention_output_batched",
            "callable": lambda attn, v: jnp.einsum("...BNTS,BSNH->...BTNH", attn, v),
            "input_shapes": [("B", 1, 8, 4, 4), ("B", 4, 8, 32)],
        },
        {
            "testcase": "einsum_ellipsis_rank_mismatch",
            "callable": lambda q, k: jnp.einsum("...BTNH,...BSNH->...BNTS", q, k),
            "input_shapes": [(2, 1, 4, 8, 32), (2, 5, 8, 32)],
            "expected_output_shapes": [(2, 2, 8, 4, 5)],
        },
        {
            "testcase": "einsum_attention_logits_batched_rank_mismatch",
            "callable": lambda q, k: jnp.einsum("...BTNH,BSNH->...BNTS", q, k),
            "input_shapes": [(2, 1, 4, 8, 16), (2, 5, 8, 16)],
            "expected_output_shapes": [(2, 2, 8, 4, 5)],
        },
    ],
)
class EinsumPlugin(PrimitiveLeafPlugin):
    _ORIG_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(*avals, equation: str, **_):
        out_shape = EinsumPlugin._checked_shape(avals, equation)
        return core.ShapedArray(out_shape, avals[0].dtype)

    @staticmethod
    def _get_dynamic_output_shape_manual(
        input_shapes: list[tuple[Any, ...]], equation: str
    ) -> tuple[Any, ...]:
        if "->" not in equation:
            raise NotImplementedError(
                "Implicit einsum output shape calculation is not supported."
            )

        input_specs_str, output_spec_str = equation.split("->")
        input_specs = input_specs_str.split(",")

        if len(input_specs) != len(input_shapes):
            raise ValueError(
                f"Einsum specs count ({len(input_specs)}) mismatches inputs count ({len(input_shapes)})."
            )

        dim_map: Dict[str, Any] = {}
        batch_shapes = []

        for spec, shape in zip(input_specs, input_shapes):
            core_spec = spec.replace("...", "")

            if "..." in spec:
                num_core_dims = len(core_spec)
                num_batch_dims = len(shape) - num_core_dims
                if num_batch_dims < 0:
                    raise ValueError(
                        f"Ellipsis mismatch in spec '{spec}' for shape {shape}."
                    )
                batch_shapes.append(shape[:num_batch_dims])
                core_shape = shape[num_batch_dims:]
            else:
                batch_shapes.append(())
                core_shape = shape

            if len(core_spec) != len(core_shape):
                raise ValueError(
                    f"Core spec '{core_spec}' rank mismatches core shape {core_shape}."
                )

            for label, size in zip(core_spec, core_shape):
                if label in dim_map and dim_map[label] != size:
                    if dim_map[label] == 1:
                        dim_map[label] = size
                    elif size != 1:
                        try:
                            if dim_map[label] == size:
                                continue
                        except Exception:
                            pass
                        raise ValueError(
                            f"Inconsistent size for label '{label}': {dim_map[label]} vs {size}."
                        )
                else:
                    dim_map[label] = size

        broadcasted_batch_shape = ()
        non_empty_batch_shapes = [bs for bs in batch_shapes if bs]
        if non_empty_batch_shapes:
            broadcasted_batch_shape = np.broadcast_shapes(*non_empty_batch_shapes)

        output_core_spec = output_spec_str.replace("...", "")
        output_core_shape = [dim_map[label] for label in output_core_spec]

        return tuple(broadcasted_batch_shape) + tuple(output_core_shape)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        # Original equation and its parts
        equation = params["equation"]
        lhs, rhs = equation.split("->")
        in_specs = lhs.split(",")

        ellipsis_ranks = []
        for spec, v in zip(in_specs, node_inputs):
            core_spec = spec.replace("...", "")
            if "..." in spec:
                core_rank = len(core_spec)
                ellipsis_ranks.append(len(v.aval.shape) - core_rank)
            else:
                ellipsis_ranks.append(0)

        max_ellipsis_rank = max(ellipsis_ranks) if ellipsis_ranks else 0

        # --- NEW: rewrite equation so that any spec we pad also gets '...' ---
        new_specs = []
        for spec, er in zip(in_specs, ellipsis_ranks):
            if ("..." not in spec) and (er < max_ellipsis_rank):
                # we're about to pad this operand, so give it an ellipsis
                new_specs.append("..." + spec)
            else:
                new_specs.append(spec)
        new_equation = ",".join(new_specs) + "->" + rhs

        input_names = []

        for spec, v, er in zip(in_specs, node_inputs, ellipsis_ranks):
            base_name = s.get_name(v)
            if er < max_ellipsis_rank:
                pad = max_ellipsis_rank - er
                # compute the statically-known padded shape
                new_shape = (1,) * pad + v.aval.shape
                # make a constant tensor [0,1,2,...] for the Unsqueeze‐axes input
                axes_const = s.get_constant_name(np.arange(pad, dtype=np.int64))
                padded = s.get_unique_name(base_name + "_pad")
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[base_name, axes_const],
                        outputs=[padded],
                        name=s.get_unique_name("unsqueeze"),
                    )
                )
                # *** KEY FIX *** register both metadata & value_info so ONNX shape‐inference
                # can see that padded has rank = max_ellipsis_rank + core_rank
                s.builder.register_value_info_metadata(
                    padded, shape=new_shape, dtype=v.aval.dtype
                )
                s.builder.add_value_info(padded, shape=new_shape, dtype=v.aval.dtype)
                input_names.append(padded)
            else:
                input_names.append(base_name)

        out_var = node_outputs[0]
        out_name = s.get_name(out_var)

        # emit the Einsum with the adjusted equation
        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=input_names,
                outputs=[out_name],
                name=s.get_unique_name("einsum"),
                equation=new_equation,
            )
        )
        inferred_shape = EinsumPlugin._checked_shape(
            [v.aval for v in node_inputs], equation
        )
        s.add_shape_info(out_name, inferred_shape, out_var.aval.dtype)

    @staticmethod
    def _einsum_binding(*args: Any, equation: str, **kwargs: Any) -> Any:
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
        EinsumPlugin._ORIG_CALL = orig_fn

        def patched_einsum(subscripts: str, *operands: Any, **kwargs: Any) -> Any:
            return EinsumPlugin._einsum_binding(
                *operands, equation=subscripts, **kwargs
            )

        return patched_einsum

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "target_attribute": "einsum",
            "patch_function": EinsumPlugin.get_monkey_patch,
        }

    @staticmethod
    def _checked_shape(
        arg_avals: Sequence[core.AbstractValue], equation: str
    ) -> Tuple[Any, ...]:
        try:
            return EinsumPlugin._get_dynamic_output_shape_manual(
                [a.shape for a in arg_avals], equation
            )
        except Exception:
            orig_einsum = EinsumPlugin._ORIG_CALL or jnp.einsum
            dummies = [ShapeDtypeStruct(a.shape, a.dtype) for a in arg_avals]
            return eval_shape(lambda *xs: orig_einsum(equation, *xs), *dummies).shape


try:
    _std_rule = importlib.import_module("jax._src.numpy.einsum").einsum_batching_rule
    batching.primitive_batchers[jnp.einsum_p] = _std_rule
except (ModuleNotFoundError, AttributeError):

    def _fallback_einsum_batching_rule(args, batch_axes, **params):
        equation = params["equation"]
        in_specs, out_spec = equation.split("->")
        new_in_specs = [
            spec if spec.startswith("...") else f"...{spec}"
            for spec in in_specs.split(",")
        ]
        out_spec_new = out_spec
        if any(s.startswith("...") for s in new_in_specs) and not out_spec.startswith(
            "..."
        ):
            out_spec_new = f"...{out_spec}"
        params = dict(params, equation=f"{','.join(new_in_specs)}->{out_spec_new}")
        res = jnp.einsum_p.bind(*args, **params)
        return res, 0

    batching.primitive_batchers[jnp.einsum_p] = _fallback_einsum_batching_rule

jnp.einsum_p.def_abstract_eval(EinsumPlugin.abstract_eval)
