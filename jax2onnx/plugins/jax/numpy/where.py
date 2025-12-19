# jax2onnx/plugins/jax/numpy/where.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.interpreters import batching

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_WHERE_PRIM: Final = make_jnp_primitive("jax.numpy.where")

# Deterministic fixtures used by regression-style testcases. Keep them small so
# they stay easy to reason about while still exercising the broadcaster logic
# that previously regressed when generated randomly.
_WHERE_A_COND: Final[np.ndarray] = np.array(
    [[[True]], [[False]], [[True]], [[False]]], dtype=np.bool_
)
_WHERE_A_DATA: Final[np.ndarray] = np.arange(4 * 1 * 4, dtype=np.float32).reshape(
    4, 1, 4
)
_WHERE_B_COND: Final[np.ndarray] = np.array(
    [[[False]], [[True]], [[True]], [[False]]], dtype=np.bool_
)
_WHERE_B_DATA: Final[np.ndarray] = (
    np.arange(4 * 1 * 4, dtype=np.int32).reshape(4, 1, 4) * 2
)


def _create_problematic_where_sequence(cond_input, data_input):
    true_val = jnp.array(1.0, dtype=data_input.dtype)
    false_val = jnp.array(0.0, dtype=data_input.dtype)
    where_output = jnp.where(cond_input, true_val, false_val)
    return data_input * where_output


@register_primitive(
    jaxpr_primitive=_WHERE_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="where",
    testcases=[
        {
            "testcase": "where_simple",
            "callable": lambda: jnp.where(
                jnp.array([True, False, True]),
                jnp.array([1, 2, 3], dtype=jnp.float32),
                jnp.array([-1, -2, -3], dtype=jnp.float32),
            ),
            "post_check_onnx_graph": EG(
                ["Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_broadcast",
            "callable": lambda: jnp.where(
                jnp.array([True, False, True, False])[:, None],
                jnp.array(np.arange(20, dtype=np.float32).reshape(4, 5)),
                -jnp.array(np.arange(20, dtype=np.float32).reshape(4, 5)),
            ),
            "post_check_onnx_graph": EG(
                ["Neg:4x5 -> Where:4x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_gpt_mask_scores_literal_else",
            "callable": lambda mask, scores: jnp.where(mask, scores, -1e9),
            "input_shapes": [("B", 1, "T", "T"), ("B", 12, "T", "T")],
            "input_dtypes": [np.bool_, np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": -1_000_000_000.0}},
                        "path": "Where:Bx12xTxT",
                    }
                ],
                symbols={"B": None, "T": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_multidim_condition_scalar_branches_broadcast",
            "callable": lambda: jnp.where(
                jnp.array([[[True]], [[False]], [[True]]], dtype=np.bool_),
                5.0,
                -5.0,
            ),
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 5.0}, 2: {"const": -5.0}},
                        "path": "Where:3x1x1",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_A",
            "callable": lambda: _create_problematic_where_sequence(
                jnp.array(_WHERE_A_COND, dtype=jnp.bool_),
                jnp.array(_WHERE_A_DATA, dtype=jnp.float32),
            ),
            "post_check_onnx_graph": EG(
                ["Where:4x1x1 -> Mul:4x1x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_B",
            "callable": lambda: _create_problematic_where_sequence(
                jnp.array(_WHERE_B_COND, dtype=jnp.bool_),
                jnp.array(_WHERE_B_DATA, dtype=jnp.int32),
            ),
            "post_check_onnx_graph": EG(
                ["Where:4x1x1 -> Mul:4x1x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_gpt_mask_scores_scalar_else",
            "callable": lambda mask, scores: jnp.where(mask, scores, -1e9),
            "input_shapes": [("B", 1, "T", "T"), ("B", 12, "T", "T")],
            "input_dtypes": [np.bool_, np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": -1_000_000_000.0}},
                        "path": "Where:Bx12xTxT",
                    }
                ],
                symbols={"B": None, "T": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_int_condition_cast",
            "callable": lambda: jnp.where(
                jnp.array([1, 0, 2], dtype=np.int32),
                jnp.array([1.0, 2.0, 3.0], dtype=np.float32),
                jnp.array([0.0, 0.0, 0.0], dtype=np.float32),
            ),
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_literal_else_pyfloat",
            "callable": lambda: jnp.where(
                jnp.array([[True, False], [False, True]]),
                jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                -1e9,
            ),
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": -1_000_000_000.0}},
                        "path": "Where:2x2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_jax_int_literals_broadcast_f64_mode",
            "callable": lambda: jnp.where(
                jnp.array([[True], [False], [True]], dtype=np.bool_),
                jnp.array(1, dtype=np.int64),
                jnp.array(0, dtype=np.int64),
            ),
            "enable_double_precision": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}, 2: {"const": 0.0}},
                        "path": "Where:3x1",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_dtype_mismatch_f64_vs_i32_promote",
            "callable": lambda: jnp.where(
                jnp.array([1.0, -2.0, 3.0], dtype=np.float64) > 0,
                jnp.array([1.0, -2.0, 3.0], dtype=np.float64),
                jnp.array([1, 2, 3], dtype=np.int32),
            ),
            "enable_double_precision": True,
            "post_check_onnx_graph": EG(
                ["Greater:3 -> Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_where_basic",
            "callable": lambda c, x, y: jnp.where(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_where_broadcast",
            "callable": lambda c, x, y: jnp.where(c[:, None], x, y),
            "input_shapes": [(4,), (4, 5), (4, 5)],
            "post_check_onnx_graph": EG(
                ["Reshape:4x1 -> Expand:4x1 -> Cast:4x1 -> Where:4x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_where_scalar_else",
            "callable": lambda c, x: jnp.where(c, x, -1e9),
            "input_shapes": [(2, 2), (2, 2)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": -1_000_000_000.0}},
                        "path": "Cast:2x2 -> Where:2x2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "where_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.where(y > 0, y, -y))(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpWherePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _WHERE_PRIM
    _FUNC_NAME: ClassVar[str] = "where"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(cond, x, y, **_):
        if not all(isinstance(av, jax.core.ShapedArray) for av in (cond, x, y)):
            raise TypeError("jnp.where expects ShapedArray inputs")
        promoted = np.promote_types(x.dtype, y.dtype)
        out_shape = jnp.broadcast_shapes(cond.shape, x.shape, y.shape)
        return jax.core.ShapedArray(out_shape, promoted)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        cond_var, x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        cond_val = ctx.get_value_for_var(
            cond_var, name_hint=ctx.fresh_name("where_cond")
        )
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("where_x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("where_y"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("where_out"))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for where lowering")

        cond_dtype = np.dtype(getattr(cond_var.aval, "dtype", np.bool_))
        if cond_dtype != np.bool_:
            cond_val = builder.Cast(
                cond_val,
                _outputs=[ctx.fresh_name("where_cond_cast")],
                to=int(ir.DataType.BOOL.value),
            )
            cond_val.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(cond_val, tuple(getattr(cond_var.aval, "shape", ())))
            _ensure_value_metadata(ctx, cond_val)

        target_dtype = np.promote_types(
            np.dtype(getattr(x_var.aval, "dtype", np.float32)),
            np.dtype(getattr(y_var.aval, "dtype", np.float32)),
        )
        if (
            not ctx.builder.enable_double_precision
            and np.issubdtype(target_dtype, np.floating)
            and target_dtype == np.float64
        ):

            def _is_value_double(val: ir.Value) -> bool:
                enum = getattr(getattr(val, "type", None), "dtype", None)
                return enum == ir.DataType.DOUBLE

            if not _is_value_double(x_val) and not _is_value_double(y_val):
                target_dtype = np.float32
        dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)

        x_cast = self._maybe_cast(ctx, x_val, x_var, target_dtype, "x")
        y_cast = self._maybe_cast(ctx, y_val, y_var, target_dtype, "y")

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("Where")
        result = builder.Where(
            cond_val,
            x_cast,
            y_cast,
            _outputs=[out_name],
        )
        result.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, result)

    @staticmethod
    def _maybe_cast(
        ctx: "IRContext", value: ir.Value, var, target_dtype: np.dtype, tag: str
    ) -> ir.Value:
        current_dtype = np.dtype(getattr(var.aval, "dtype", target_dtype))
        if current_dtype == target_dtype:
            dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
            value_enum = getattr(getattr(value, "type", None), "dtype", None)
            if value_enum == dtype_enum:
                return value

        dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
        const_payload = getattr(value, "const_value", None)
        if const_payload is not None:
            try:
                arr = const_payload.numpy()
            except Exception:
                arr = None
            if arr is not None and arr.dtype != target_dtype:
                arr = arr.astype(target_dtype, copy=False)
                value.const_value = ir.tensor(arr)
                value.type = ir.TensorType(dtype_enum)
                return value

        dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for where lowering")
        cast_val = builder.Cast(
            value,
            _outputs=[ctx.fresh_name(f"where_{tag}_cast")],
            to=int(dtype_enum.value),
        )
        cast_val.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(cast_val, tuple(getattr(var.aval, "shape", ())))
        _ensure_value_metadata(ctx, cast_val)
        return cast_val

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.where not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(condition, x=None, y=None):
                if x is None or y is None:
                    raise NotImplementedError(
                        "jnp.where with fewer than three arguments is not supported"
                    )
                return cls._PRIM.bind(condition, x, y)

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


@JnpWherePlugin._PRIM.def_impl
def _where_impl(condition, x=None, y=None):
    if x is None or y is None:
        raise NotImplementedError(
            "jnp.where with fewer than three arguments is not supported"
        )
    orig = get_orig_impl(JnpWherePlugin._PRIM, JnpWherePlugin._FUNC_NAME)
    return orig(condition, x, y)


JnpWherePlugin._PRIM.def_abstract_eval(JnpWherePlugin.abstract_eval)


def _where_batch_rule(args, dims, **params):
    return broadcast_batcher_compat(JnpWherePlugin._PRIM, args, dims, **params)


batching.primitive_batchers[JnpWherePlugin._PRIM] = _where_batch_rule
