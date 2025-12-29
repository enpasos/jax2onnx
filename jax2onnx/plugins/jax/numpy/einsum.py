# jax2onnx/plugins/jax/numpy/einsum.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from numpy.typing import ArrayLike
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_EINSUM_PRIM: Final = make_jnp_primitive("jax.numpy.einsum")
_JNP_EINSUM_ORIG: Final = jnp.einsum


def _einsum_shape(
    avals: Sequence[core.AbstractValue], equation: str
) -> tuple[tuple[Any, ...], np.dtype[Any]]:
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
    since="v0.1.0",
    context="primitives.jnp",
    component="einsum",
    testcases=[
        {
            "testcase": "einsum_vector_dot",
            "callable": lambda x, y: jnp.einsum("i,i->", x, y),
            "input_shapes": [(5,), (5,)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_matrix_vector",
            "callable": lambda x, y: jnp.einsum("ij,j->i", x, y),
            "input_shapes": [(3, 5), (5,)],
            "post_check_onnx_graph": EG(
                ["Einsum:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_matrix_matrix",
            "callable": lambda x, y: jnp.einsum("ij,jk->ik", x, y),
            "input_shapes": [("B", 5), (5, 2)],
            "post_check_onnx_graph": EG(
                ["Einsum:Bx2"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_transpose",
            "callable": lambda x: jnp.einsum("ij->ji", x),
            "input_shapes": [(3, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum:5x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_batch_transpose",
            "callable": lambda x: jnp.einsum("...ij->...ji", x),
            "input_shapes": [("B", 3, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum:Bx5x3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_diag",
            "callable": lambda x: jnp.einsum("ii->i", x),
            "input_shapes": [(5, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_sum_reduce",
            "callable": lambda x: jnp.einsum("ij->", x),
            "input_shapes": [(3, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_multi_operand",
            "callable": lambda a, b, c: jnp.einsum("ij,jk,kl->il", a, b, c),
            "input_shapes": [(2, 3), (3, 4), (4, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_attention_logits_orig",
            "callable": lambda q, k: jnp.einsum("BTNH,BSNH->BNTS", q, k),
            "input_shapes": [("B", 4, 8, 32), ("B", 4, 8, 32)],
            "post_check_onnx_graph": EG(
                ["Einsum:Bx8x4x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_attention_output_orig",
            "callable": lambda attn, v: jnp.einsum("BNTS,BSNH->BTNH", attn, v),
            "input_shapes": [("B", 8, 4, 4), ("B", 4, 8, 32)],
            "post_check_onnx_graph": EG(
                ["Einsum:Bx4x8x32"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_attention_logits_batched",
            "callable": lambda q, k: jnp.einsum("...BTNH,BSNH->...BNTS", q, k),
            "input_shapes": [("B", 1, 4, 8, 32), ("B", 4, 8, 32)],
            "post_check_onnx_graph": EG(
                ["Einsum:BxBx8x4x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_attention_output_batched",
            "callable": lambda attn, v: jnp.einsum("...BNTS,BSNH->...BTNH", attn, v),
            "input_shapes": [("B", 1, 8, 4, 4), ("B", 4, 8, 32)],
            "post_check_onnx_graph": EG(
                ["Einsum:BxBx4x8x32"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_ellipsis_rank_mismatch",
            "callable": lambda q, k: jnp.einsum("...BTNH,...BSNH->...BNTS", q, k),
            "input_shapes": [(2, 1, 4, 8, 32), (2, 5, 8, 32)],
            "expected_output_shapes": [(2, 2, 8, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum:2x2x8x4x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "einsum_attention_logits_batched_rank_mismatch",
            "callable": lambda q, k: jnp.einsum("...BTNH,BSNH->...BNTS", q, k),
            "input_shapes": [(2, 1, 4, 8, 16), (2, 5, 8, 16)],
            "expected_output_shapes": [(2, 2, 8, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Einsum:2x2x8x4x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpEinsumPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EINSUM_PRIM
    _FUNC_NAME: ClassVar[str] = "einsum"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *avals: core.AbstractValue,
        equation: str,
        precision: Any | None = None,
        optimize: Any | None = None,
        preferred_element_type: Any | None = None,
        _dot_general: Any | None = None,
    ) -> core.ShapedArray:
        del _dot_general
        shape, dtype = _einsum_shape(avals, equation)
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        equation = params["equation"]

        input_vals = []
        for var in eqn.invars:
            val = ctx.get_value_for_var(var, name_hint=ctx.fresh_name("einsum_in"))
            input_vals.append(val)
        out_var = eqn.outvars[0]
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("einsum_out")
        )

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
                    axes_const = _const_i64(
                        ctx, axes, ctx.fresh_name("einsum_pad_axes")
                    )
                    _stamp_type_and_shape(axes_const, tuple(axes.shape))
                    _ensure_value_metadata(ctx, axes_const)
                    padded_shape = tuple([1] * pad + list(shape))
                    padded_val = ctx.builder.Unsqueeze(
                        val,
                        axes_const,
                        _outputs=[ctx.fresh_name("einsum_pad")],
                    )
                    val_dtype = getattr(getattr(val, "type", None), "dtype", None)
                    if val_dtype is not None:
                        padded_val.type = ir.TensorType(val_dtype)
                    _stamp_type_and_shape(padded_val, padded_shape)
                    _ensure_value_metadata(ctx, padded_val)
                    adjusted_inputs.append(padded_val)
                    continue
            adjusted_inputs.append(val)

        adjusted_equation = ",".join(adjusted_specs) + "->" + rhs

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("einsum_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("einsum_out")

        result = ctx.builder.Einsum(
            *adjusted_inputs,
            equation=adjusted_equation,
            _outputs=[desired_name],
        )
        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            result.type = spec_type
        else:
            inferred_dtype = next(
                (
                    getattr(getattr(v, "type", None), "dtype", None)
                    for v in adjusted_inputs
                    if getattr(getattr(v, "type", None), "dtype", None) is not None
                ),
                None,
            )
            if inferred_dtype is not None:
                result.type = ir.TensorType(inferred_dtype)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        allowed_kwargs = {
            "precision",
            "optimize",
            "preferred_element_type",
            "_dot_general",
        }

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.einsum not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                equation: str,
                *operands: ArrayLike,
                **kwargs: Any,
            ) -> jax.Array:
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
    equation: str,
    *operands: ArrayLike,
    precision: Any | None = None,
    optimize: Any | None = None,
    preferred_element_type: Any | None = None,
    _dot_general: Any | None = None,
) -> jax.Array:
    try:
        orig = get_orig_impl(JnpEinsumPlugin._PRIM, JnpEinsumPlugin._FUNC_NAME)
    except RuntimeError:
        orig = _JNP_EINSUM_ORIG
    kwargs = {}
    if precision is not None:
        kwargs["precision"] = precision
    if optimize is not None:
        kwargs["optimize"] = optimize
    if preferred_element_type is not None:
        kwargs["preferred_element_type"] = preferred_element_type
    if _dot_general is not None:
        kwargs["_dot_general"] = _dot_general
    return orig(equation, *operands, **kwargs)


JnpEinsumPlugin._PRIM.def_abstract_eval(JnpEinsumPlugin.abstract_eval)


def _einsum_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    equation: str,
    precision: Any | None = None,
    optimize: Any | None = None,
    preferred_element_type: Any | None = None,
) -> tuple[jax.Array, int | None]:
    kwargs: dict[str, Any] = {}
    if precision is not None:
        kwargs["precision"] = precision
    if optimize is not None:
        kwargs["optimize"] = optimize
    if preferred_element_type is not None:
        kwargs["preferred_element_type"] = preferred_element_type

    try:
        orig = get_orig_impl(JnpEinsumPlugin._PRIM, JnpEinsumPlugin._FUNC_NAME)
    except RuntimeError:
        orig = _JNP_EINSUM_ORIG

    processed_args = []
    batch_size = None
    for arg, bdim in zip(batched_args, batch_dims):
        if bdim is None:
            processed_args.append(arg)
            continue
        if bdim != 0:
            arg = jnp.moveaxis(arg, bdim, 0)
        if batch_size is None:
            batch_size = arg.shape[0]
        processed_args.append(arg)

    if batch_size is None:
        # No batched operands; fall back to primitive bind.
        result = JnpEinsumPlugin._PRIM.bind(
            *batched_args,
            equation=equation,
            precision=precision,
            optimize=optimize,
            preferred_element_type=preferred_element_type,
        )
        return result, None

    broadcasted_args = []
    for arg, bdim in zip(processed_args, batch_dims):
        if bdim is None:
            expanded = jnp.expand_dims(arg, 0)
            broadcasted = jnp.broadcast_to(expanded, (batch_size,) + arg.shape)
            broadcasted_args.append(broadcasted)
        else:
            broadcasted_args.append(arg)

    def _einsum_no_batch(*xs):
        return orig(equation, *xs, **kwargs)

    result = jax.vmap(_einsum_no_batch)(*broadcasted_args)
    return result, 0


jax.interpreters.batching.primitive_batchers[JnpEinsumPlugin._PRIM] = _einsum_batch_rule
