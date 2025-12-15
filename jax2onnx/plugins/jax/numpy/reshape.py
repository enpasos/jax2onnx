# jax2onnx/plugins/jax/numpy/reshape.py

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
from jax.interpreters import batching
import numpy as np
from numpy.typing import ArrayLike
import onnx_ir as ir
from jax import core

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.utils.shape_poly import (
    dim_expr_constant_value,
    symbolic_dim_eq,
)


_RESHAPE_PRIM: Final = make_jnp_primitive("jax.numpy.reshape")


def _iter_newshape(newshape: Sequence[int | object] | int | object) -> Iterable:
    if isinstance(newshape, Sequence):
        return newshape
    return (newshape,)


def _find_axis_for_dim(dim: object, input_shape: Sequence[object]) -> int | None:
    for idx, src in enumerate(input_shape):
        if dim is src:
            return idx
        if symbolic_dim_eq(dim, src):
            return idx
        if hasattr(dim, "_hashable_content") and hasattr(src, "_hashable_content"):
            if dim._hashable_content() == src._hashable_content():  # type: ignore[attr-defined]
                return idx
    return None


def _depth_to_space_issue_144(inputs: jax.Array, block_size: int) -> jax.Array:
    height, width, channels = inputs.shape
    new_depth = channels // (block_size * block_size)
    outputs = jnp.reshape(inputs, [height, width, block_size, block_size, new_depth])
    outputs = jnp.transpose(outputs, [0, 2, 1, 3, 4])
    return jnp.reshape(outputs, [height * block_size, width * block_size, new_depth])


@register_primitive(
    jaxpr_primitive=_RESHAPE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape_1",
            "callable": lambda a: jnp.reshape(a, (2, 6)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:2x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_2",
            "callable": lambda a: jnp.reshape(a, (-1, 2)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:6x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_3",
            "callable": lambda a: jnp.reshape(a, (2, -1)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:2x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_4",
            "callable": lambda a: jnp.reshape(a, (a.shape[0], -1)),
            "input_shapes": [("B", 3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:Bx12"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_to_scalar",
            "callable": lambda: jnp.reshape(np.array([7.0], dtype=np.float32), ()),
            "input_values": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {0: {"const": 7.0}},
                        "path": "Reshape",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_from_scalar",
            "callable": lambda: jnp.reshape(np.array(3.0, dtype=np.float32), (1,)),
            "input_values": [],
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 3.0},
                            1: {"const": 1.0},
                        },
                        "path": "Reshape:1",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_cnn",
            "callable": lambda x: x.reshape(x.shape[0], -1),
            "input_shapes": [("B", 64, 14, 14)],
            "post_check_onnx_graph": EG(
                ["Reshape:Bx12544"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_valid_flatten_trailing",
            "callable": lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])),
            "input_shapes": [(201, 1, 5)],
            "post_check_onnx_graph": EG(
                ["Reshape:201x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_with_target_shape_from_symbolic_dim_computation",
            "callable": lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])),
            "input_shapes": [("N", 3, 5)],
            "post_check_onnx_graph": EG(
                ["Reshape:Nx15"],
                symbols={"N": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_basic",
            "callable": lambda a: jnp.reshape(a, (2, 6)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:2x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_infer",
            "callable": lambda a: jnp.reshape(a, (-1, 2)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:6x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_symbolic_flatten",
            "callable": lambda a: jnp.reshape(a, (a.shape[0], -1)),
            "input_shapes": [("B", 8, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:Bx32"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reshape_vmap_batching_issue_144",
            "callable": lambda x: jax.vmap(
                _depth_to_space_issue_144, in_axes=(0, None)
            )(x, 2),
            "input_shapes": [(1, 4, 4, 4)],
            "expected_output_shapes": [(1, 8, 8, 1)],
            "post_check_onnx_graph": EG(
                ["Reshape:1x4x4x2x2x1 -> Transpose:1x4x2x4x2x1 -> Reshape:1x8x8x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpReshapePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _RESHAPE_PRIM
    _FUNC_NAME: ClassVar[str] = "reshape"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        newshape: Sequence[int | object] | int | object,
        order: str | None = "C",
    ) -> core.ShapedArray:
        if order not in (None, "C"):
            raise NotImplementedError("Only C-order reshape is supported")
        storage_slot = f"__orig_impl___{JnpReshapePlugin._FUNC_NAME}"
        orig = getattr(JnpReshapePlugin._PRIM, storage_slot, jnp.reshape)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result = jax.eval_shape(lambda arr: orig(arr, newshape, order=order), spec)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        newshape_param = params.get("new_sizes", params.get("newshape"))
        order = params.get("order", "C")
        if order not in (None, "C"):
            raise NotImplementedError(
                "jnp.reshape order other than 'C' is not supported"
            )
        if newshape_param is None:
            raise KeyError("reshape parameters missing 'newshape' or 'new_sizes'")

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("reshape_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reshape_out")
        )
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for reshape lowering"
            )

        input_shape = tuple(getattr(arr_var.aval, "shape", ()))
        target_shape = tuple(getattr(out_var.aval, "shape", ()))

        newshape_elems = tuple(_iter_newshape(newshape_param))
        shape_components: list[ir.Value] = []
        shape_tensor_rank = len(newshape_elems)

        shape_value: ir.Value | None = None

        def ensure_shape_value() -> ir.Value:
            nonlocal shape_value
            if shape_value is not None:
                return shape_value
            shape_value = builder.Shape(
                arr_val,
                _outputs=[ctx.fresh_name("reshape_input_shape")],
            )
            shape_value.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(shape_value, (len(input_shape),))
            _ensure_value_metadata(ctx, shape_value)
            return shape_value

        def gather_axis(idx: int) -> ir.Value:
            shape_val = ensure_shape_value()
            axis_const = _const_i64(
                ctx, np.asarray(idx, dtype=np.int64), "reshape_axis"
            )
            gather_val = builder.Gather(
                shape_val,
                axis_const,
                axis=0,
                _outputs=[ctx.fresh_name("reshape_dim")],
            )
            gather_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(gather_val, ())
            _ensure_value_metadata(ctx, gather_val)
            axes_const = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "reshape_unsq_axes"
            )
            unsqueezed = builder.Unsqueeze(
                gather_val,
                axes_const,
                _outputs=[ctx.fresh_name("reshape_dim_unsq")],
            )
            unsqueezed.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(unsqueezed, (1,))
            _ensure_value_metadata(ctx, unsqueezed)
            return unsqueezed

        for idx, dim in enumerate(newshape_elems):
            if isinstance(dim, (int, np.integer)):
                int_dim = int(dim)
                val = _const_i64(
                    ctx,
                    np.asarray([int_dim], dtype=np.int64),
                    f"reshape_dim_const_{idx}",
                )
                shape_components.append(val)
                continue

            const_val = dim_expr_constant_value(dim)
            if const_val is not None:
                val = _const_i64(
                    ctx,
                    np.asarray([const_val], dtype=np.int64),
                    f"reshape_dim_const_{idx}",
                )
                shape_components.append(val)
                continue

            axis_idx = _find_axis_for_dim(dim, input_shape)
            if axis_idx is None:
                if const_val == -1:
                    val = _const_i64(
                        ctx,
                        np.asarray([-1], dtype=np.int64),
                        f"reshape_dim_const_{idx}",
                    )
                    shape_components.append(val)
                    continue
                raise TypeError(
                    "reshape with symbolic dimensions requires mapping to input axes"
                )
            shape_components.append(gather_axis(axis_idx))

        if shape_tensor_rank == 0:
            shape_tensor = _const_i64(
                ctx, np.asarray([], dtype=np.int64), "reshape_empty_shape"
            )
        elif len(shape_components) == 1:
            shape_tensor = shape_components[0]
        else:
            const_parts: list[np.ndarray] = []
            all_static = True
            for component in shape_components:
                payload = getattr(component, "const_value", None)
                if payload is None:
                    all_static = False
                    break
                try:
                    const_parts.append(np.asarray(payload, dtype=np.int64).reshape(-1))
                except Exception:
                    all_static = False
                    break
            if all_static:
                combined = (
                    np.concatenate(const_parts).astype(np.int64, copy=False)
                    if const_parts
                    else np.asarray([], dtype=np.int64)
                )
                shape_tensor = builder.add_initializer_from_array(
                    name=ctx.fresh_name("reshape_target_shape_c"),
                    array=combined,
                )
            else:
                shape_tensor = builder.Concat(
                    *shape_components,
                    axis=0,
                    _outputs=[ctx.fresh_name("reshape_target_shape")],
                )
                shape_tensor.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(shape_tensor, (shape_tensor_rank,))
                _ensure_value_metadata(ctx, shape_tensor)

        reshape_out = None
        arr_const = getattr(arr_val, "const_value", None)
        shape_const = getattr(shape_tensor, "const_value", None)
        inline_allowed = getattr(ctx, "_inside_function_scope", False)
        if inline_allowed and arr_const is not None and shape_const is not None:
            try:
                reshaped_array = np.asarray(arr_const).reshape(
                    tuple(np.asarray(shape_const, dtype=np.int64).tolist())
                )
                reshape_out = builder.add_initializer_from_array(
                    name=getattr(out_spec, "name", None) or ctx.fresh_name("Reshape"),
                    array=reshaped_array,
                )
            except Exception:
                reshape_out = None

        if reshape_out is None:
            reshape_out = builder.Reshape(
                arr_val,
                shape_tensor,
                allowzero=0,
                _outputs=[getattr(out_spec, "name", None) or ctx.fresh_name("Reshape")],
            )
        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            reshape_out.type = spec_type
        else:
            arr_dtype = getattr(getattr(arr_val, "type", None), "dtype", None)
            if arr_dtype is not None:
                reshape_out.type = ir.TensorType(arr_dtype)
        _stamp_type_and_shape(reshape_out, target_shape)
        _ensure_value_metadata(ctx, reshape_out)
        ctx.bind_value_for_var(out_var, reshape_out)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl___{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.reshape not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                newshape: Sequence[int | object] | int | object,
                order: str | None = "C",
            ) -> jax.Array:
                if order not in (None, "C"):
                    raise NotImplementedError("Only C-order reshape is supported")
                return cls._PRIM.bind(
                    a, newshape=tuple(_iter_newshape(newshape)), order=order
                )

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


@JnpReshapePlugin._PRIM.def_impl
def _reshape_impl(
    a: ArrayLike,
    newshape: Sequence[int | object] | int | object,
    order: str | None = "C",
) -> jax.Array:
    orig = get_orig_impl(JnpReshapePlugin._PRIM, JnpReshapePlugin._FUNC_NAME)
    return orig(a, newshape, order=order)


def _reshape_batch(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | type(batching.not_mapped), ...],
    *,
    newshape: Sequence[int | object] | int | object,
    order: str | None = "C",
) -> tuple[jax.Array, int | type(batching.not_mapped)]:
    (a,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        return _reshape_impl(a, newshape, order=order), batching.not_mapped

    batch_size = a.shape[bdim]
    a = batching.bdim_at_front(a, bdim, batch_size)

    orig = get_orig_impl(JnpReshapePlugin._PRIM, JnpReshapePlugin._FUNC_NAME)
    bs = tuple(_iter_newshape(newshape))
    res = orig(a, (a.shape[0],) + bs, order=order)
    return res, 0


batching.primitive_batchers[JnpReshapePlugin._PRIM] = _reshape_batch


JnpReshapePlugin._PRIM.def_abstract_eval(JnpReshapePlugin.abstract_eval)
