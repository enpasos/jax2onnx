from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_EINSUM_PRIM = make_jnp_primitive("jax.numpy.einsum")


def _einsum_shape(avals, equation: str):
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
    orig = getattr(_EINSUM_PRIM, "__orig_impl__einsum", jnp.einsum)
    result = jax.eval_shape(lambda *args: orig(equation, *args), *specs)
    return result.shape, result.dtype


@register_primitive(
    jaxpr_primitive=_EINSUM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        }
    ],
    since="v0.9.0",
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
class JnpEinsumPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EINSUM_PRIM
    _FUNC_NAME: ClassVar[str] = "einsum"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *avals,
        equation: str,
        precision: Any | None = None,
        optimize: Any | None = None,
        preferred_element_type: Any | None = None,
    ):
        shape, dtype = _einsum_shape(avals, equation)
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        equation = params["equation"]

        input_vals = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("einsum_in"))
            for var in eqn.invars
        ]
        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("einsum_out"))

        lhs, rhs = equation.split("->")
        in_specs = lhs.split(",")

        var_shapes = [tuple(getattr(var.aval, "shape", ())) for var in eqn.invars]

        ellipsis_ranks: list[int] = []
        for spec, shape in zip(in_specs, var_shapes):
            core_spec = spec.replace("...", "")
            core_rank = len(core_spec)
            ellipsis_rank = len(shape) - core_rank
            if ellipsis_rank < 0:
                raise ValueError(
                    f"Einsum spec '{spec}' expects rank >= {core_rank} but got shape {shape}"
                )
            ellipsis_ranks.append(ellipsis_rank if "..." in spec else 0)

        max_ellipsis_rank = max(ellipsis_ranks) if ellipsis_ranks else 0

        adjusted_specs: list[str] = []
        for spec, er in zip(in_specs, ellipsis_ranks):
            if "..." not in spec and er < max_ellipsis_rank:
                adjusted_specs.append("..." + spec)
            else:
                adjusted_specs.append(spec)

        adjusted_inputs: list[ir.Value] = []
        for spec, val, shape, er in zip(
            in_specs, input_vals, var_shapes, ellipsis_ranks
        ):
            if er < max_ellipsis_rank:
                pad = max_ellipsis_rank - er
                if pad > 0:
                    axes = np.arange(pad, dtype=np.int64)
                    axes_const = _const_i64(ctx, axes, "einsum_pad_axes")
                    padded_shape = tuple([1] * pad + list(shape))
                    padded_val = ir.Value(
                        name=ctx.fresh_name("einsum_pad"),
                        type=ir.TensorType(val.type.dtype),
                    )
                    ctx.add_node(
                        ir.Node(
                            op_type="Unsqueeze",
                            domain="",
                            inputs=[val, axes_const],
                            outputs=[padded_val],
                            name=ctx.fresh_name("Unsqueeze"),
                        )
                    )
                    _stamp_type_and_shape(padded_val, padded_shape)
                    _ensure_value_info(ctx, padded_val)
                    adjusted_inputs.append(padded_val)
                    continue
            adjusted_inputs.append(val)

        adjusted_equation = ",".join(adjusted_specs) + "->" + rhs

        ctx.add_node(
            ir.Node(
                op_type="Einsum",
                domain="",
                inputs=adjusted_inputs,
                outputs=[out_val],
                name=ctx.fresh_name("Einsum"),
                attributes=[IRAttr("equation", IRAttrType.STRING, adjusted_equation)],
            )
        )
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        allowed_kwargs = {"precision", "optimize", "preferred_element_type"}

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.einsum not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(equation, *operands, **kwargs):
                unknown = set(kwargs) - allowed_kwargs
                if unknown:
                    raise NotImplementedError(
                        f"Unsupported kwargs for jnp.einsum: {tuple(sorted(unknown))}"
                    )

                bind_kwargs = {"equation": str(equation)}
                for key in allowed_kwargs:
                    value = kwargs.get(key, None)
                    if value is not None:
                        bind_kwargs[key] = value

                return cls._PRIM.bind(*operands, **bind_kwargs)

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpEinsumPlugin._PRIM.def_impl
def _einsum_impl(
    equation,
    *operands,
    precision=None,
    optimize=None,
    preferred_element_type=None,
):
    orig = get_orig_impl(JnpEinsumPlugin._PRIM, JnpEinsumPlugin._FUNC_NAME)
    kwargs = {}
    if precision is not None:
        kwargs["precision"] = precision
    if optimize is not None:
        kwargs["optimize"] = optimize
    if preferred_element_type is not None:
        kwargs["preferred_element_type"] = preferred_element_type
    return orig(equation, *operands, **kwargs)


JnpEinsumPlugin._PRIM.def_abstract_eval(JnpEinsumPlugin.abstract_eval)
