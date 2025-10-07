# jax2onnx/plugins/jax/lax/fori_loop.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import tree_util
from jax.extend.core import Primitive

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_info,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    lower_jaxpr_eqns,
    make_subgraph_context,
    relax_value_to_rank_only,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from jax2onnx.converter.ir_context import IRContext


try:  # pragma: no cover - older jax without config.read
    from jax import config as _jax_config
except Exception:  # pragma: no cover
    _jax_config = None


_USE_INT64 = bool(getattr(_jax_config, "read", lambda _k: False)("jax_enable_x64"))


def _canon_int(value: int | np.integer) -> np.integer:
    return np.int64(value) if _USE_INT64 else np.int32(value)


def model_fn(x):
    steps = 5

    def body_func(index, args):
        x_val, counter = args
        x_val = x_val + 0.1 * x_val**2
        counter = counter + 1
        return x_val, counter

    return jax.lax.fori_loop(0, steps, body_func, (x, 0))


def _build_body_graph(
    parent_ctx: "IRContext",
    closed_jaxpr: Any,
    state_prototypes: "Sequence[ir.Value]",
) -> ir.Graph:
    body_ctx = make_subgraph_context(parent_ctx, prefix="fori_body")
    builder = getattr(body_ctx, "builder", None)
    if builder is None:
        raise AttributeError("Subgraph context missing builder for fori_loop body")

    iter_input = ir.Value(
        name=body_ctx.fresh_name("loop_iter"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(()),
    )
    cond_input = ir.Value(
        name=body_ctx.fresh_name("loop_cond_in"),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(()),
    )
    body_ctx.builder.inputs.extend([iter_input, cond_input])

    jaxpr = closed_jaxpr.jaxpr

    # Bind constants inside the loop body context.
    for const_var, const_value in zip(
        jaxpr.constvars, getattr(closed_jaxpr, "consts", ())
    ):
        body_ctx.bind_const_for_var(const_var, np.asarray(const_value))

    # Bind the iteration index, casting to the requested dtype when needed.
    iter_var = jaxpr.invars[0]
    iter_dtype = np.dtype(getattr(iter_var.aval, "dtype", np.int64))
    iter_enum = _dtype_to_ir(iter_dtype, builder.enable_double_precision)
    iter_value = iter_input
    if iter_enum != ir.DataType.INT64:
        cast_iter = builder.Cast(
            iter_input,
            _outputs=[body_ctx.fresh_name("loop_iter_cast")],
            to=int(iter_enum.value),
        )
        cast_iter.type = ir.TensorType(iter_enum)
        _stamp_type_and_shape(cast_iter, ())
        _ensure_value_info(body_ctx, cast_iter)
        iter_value = cast_iter
    body_ctx.bind_value_for_var(iter_var, iter_value)

    # Bind loop-carried state inputs (skip the iteration index).
    for inner_var, prototype in zip(jaxpr.invars[1:], state_prototypes):
        aval = getattr(inner_var, "aval", None)
        dtype_enum = _dtype_to_ir(
            np.dtype(getattr(aval, "dtype", getattr(prototype.type, "dtype", None))),
            builder.enable_double_precision,
        )
        shape_tuple = tuple(getattr(aval, "shape", ()))
        shape_dims = tuple(_to_ir_dim_for_shape(dim) for dim in shape_tuple)
        state_input = ir.Value(
            name=body_ctx.fresh_name("loop_state_in"),
            type=ir.TensorType(dtype_enum),
            shape=ir.Shape(shape_dims),
        )
        body_ctx.builder.inputs.append(state_input)
        body_ctx.bind_value_for_var(inner_var, state_input)
        for axis, dim in enumerate(shape_tuple):
            if isinstance(dim, (int, np.integer)):
                continue
            try:
                body_ctx._sym_origin[dim] = (state_input, axis)
            except TypeError:
                pass
            body_ctx._sym_origin_str[str(dim)] = (state_input, axis)
        _stamp_type_and_shape(state_input, shape_tuple)
        _ensure_value_info(body_ctx, state_input)
        relax_value_to_rank_only(state_input)

    # Lower body equations inside the nested context.
    lower_jaxpr_eqns(body_ctx, jaxpr)

    # Collect loop-carried outputs from the body jaxpr.
    loop_outputs: list[ir.Value] = []
    for out_var in jaxpr.outvars:
        out_val = body_ctx.get_value_for_var(
            out_var, name_hint=body_ctx.fresh_name("loop_state_out")
        )
        aval = getattr(out_var, "aval", None)
        if aval is not None:
            out_shape = tuple(getattr(aval, "shape", ()))
            out_dtype = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                builder.enable_double_precision,
            )
            out_val.type = ir.TensorType(out_dtype)
            _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(body_ctx, out_val)
        relax_value_to_rank_only(out_val)
        loop_outputs.append(out_val)

    # Propagate the loop condition unchanged (fixed-trip Loop).
    cond_out = builder.Identity(
        cond_input,
        _outputs=[body_ctx.fresh_name("loop_cond_out")],
    )
    cond_out.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(cond_out, ())
    _ensure_value_info(body_ctx, cond_out)

    body_ctx.builder.outputs = [cond_out, *loop_outputs]

    return ir.Graph(
        inputs=list(body_ctx.builder.inputs),
        outputs=list(body_ctx.builder.outputs),
        nodes=list(body_ctx.builder.nodes),
        initializers=list(body_ctx.builder.initializers),
        name=parent_ctx.fresh_name("fori_body"),
        opset_imports={"": getattr(parent_ctx.builder, "opset", 21)},
    )


@register_primitive(
    jaxpr_primitive="lax.fori_loop",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html",
    onnx=[
        {
            "component": "Loop",
            "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html",
        }
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="fori_loop",
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": lambda: jax.lax.fori_loop(0, 5, lambda i, v: v + 1, 0),
            "input_shapes": [],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "fori_loop_zero",
            "callable": lambda: jax.lax.fori_loop(0, 0, lambda i, v: v + 1, 42),
            "input_shapes": [],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "fori_loop_vector",
            "callable": lambda: jax.lax.fori_loop(
                0,
                3,
                lambda i, v: v.at[i].set(i),
                jnp.zeros((3,), dtype=jnp.int32),
            ),
            "input_shapes": [],
            "expected_output_shapes": [(3,)],
        },
        {
            "testcase": "fori_loop_example",
            "callable": lambda: jax.lax.fori_loop(
                0,
                5,
                lambda i, args: (args[0] + 0.1 * args[0] ** 2, args[1] + 1),
                (jnp.array([1.0], dtype=jnp.float32), 0),
            )[0],
            "input_shapes": [],
            "expected_output_shapes": [(1,)],
        },
        {
            "testcase": "fori_loop_test",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(2,), ()],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "fori_loop_test_f64",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float64],
            "expected_output_shapes": [(2,), ()],
            "run_only_f64_variant": True,
        },
    ],
)
class ForiLoopPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("lax.fori_loop")
    _PRIM.multiple_results = True
    _ORIG_FORI_LOOP: ClassVar[Any] = None

    @classmethod
    def binding_specs(cls):
        def _patch(orig):
            cls._ORIG_FORI_LOOP = orig

            def _wrapped(lower, upper, body_fun, init_val):
                return cls._fori_loop_binding(lower, upper, body_fun, init_val)

            return _wrapped

        return [
            AssignSpec("jax.lax", "fori_loop_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec("jax.lax", "fori_loop", _patch, delete_if_missing=False),
        ]

    @staticmethod
    def abstract_eval(*in_avals, **__):
        return tuple(in_avals)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = getattr(eqn, "params", {})
        closed = params.get("body_jaxpr")
        trip_count = int(params.get("trip_count", 0))
        lower = int(params.get("lower", 0) or 0)
        if closed is None:
            raise ValueError("fori_loop lowering requires 'body_jaxpr' parameter")
        if lower != 0:
            raise NotImplementedError(
                "fori_loop with non-zero lower bound is unsupported"
            )

        state_vals = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("fori_state"))
            for var in eqn.invars
        ]

        body_graph = _build_body_graph(ctx, closed, state_vals)

        trip_val = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("fori_trip_count"),
            value=np.asarray(trip_count, dtype=np.int64),
        )
        cond_val = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("fori_cond_init"),
            value=np.asarray(True, dtype=np.bool_),
        )

        loop_inputs = [trip_val, cond_val, *state_vals]

        output_names = [ctx.fresh_name("fori_out") for _ in eqn.outvars]
        loop_outputs = ctx.builder.Loop(
            *loop_inputs,
            body=body_graph,
            _outputs=output_names,
        )

        if not isinstance(loop_outputs, tuple):
            loop_outputs = (loop_outputs,)

        for var, val in zip(eqn.outvars, loop_outputs):
            aval = getattr(var, "aval", None)
            if aval is None:
                continue
            out_shape = tuple(getattr(aval, "shape", ()))
            out_dtype = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            val.type = ir.TensorType(out_dtype)
            _stamp_type_and_shape(val, out_shape)
            _ensure_value_info(ctx, val)
            ctx.bind_value_for_var(var, val)

    @classmethod
    def _fori_loop_binding(cls, lower, upper, body_fun, init_val):
        if int(lower) != 0:
            raise NotImplementedError("fori_loop plugin supports lower == 0 only")

        leaves, treedef = tree_util.tree_flatten(init_val)
        leaves = [
            _canon_int(leaf) if isinstance(leaf, (int, np.integer)) else leaf
            for leaf in leaves
        ]

        def body_flat(i, *state_leaves):
            state = tree_util.tree_unflatten(treedef, state_leaves)
            new_state = body_fun(i, state)
            new_leaves, new_def = tree_util.tree_flatten(new_state)
            if new_def != treedef:
                raise TypeError("fori_loop body must preserve state structure")
            return new_leaves

        body_closed = jax.make_jaxpr(body_flat)(0, *leaves)
        trip_count = int(np.asarray(upper).item()) - int(np.asarray(lower).item())

        flat_result = cls._PRIM.bind(
            *leaves,
            body_jaxpr=body_closed,
            trip_count=trip_count,
            lower=int(lower),
        )
        return tree_util.tree_unflatten(treedef, flat_result)
